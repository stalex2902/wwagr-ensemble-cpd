# ------------------------------------------------------------------------------------------------------------#
#                             From https://github.com/gpleiss/temperature_scaling                             #
# ------------------------------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from tqdm import tqdm
from betacal import BetaCalibration
from sklearn.calibration import calibration_curve
from src.metrics.metrics_utils import collect_model_predictions_on_set, get_models_predictions


def ece(y_test, preds, strategy="uniform"):
    df = pd.DataFrame({"target": y_test, "proba": preds, "bin": np.nan})

    if strategy == "uniform":
        lim_inf = np.linspace(0, 0.9, 10)
        for idx, lim in enumerate(lim_inf):
            df.loc[df["proba"] >= lim, "bin"] = idx

    elif strategy == "quantile":
        pass

    df_bin_groups = pd.concat(
        [df.groupby("bin").mean(), df["bin"].value_counts()], axis=1
    )
    df_bin_groups["ece"] = (df_bin_groups["target"] - df_bin_groups["proba"]).abs() * (
        df_bin_groups["bin"] / df.shape[0]
    )
    return df_bin_groups["ece"].sum()


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model, lr=1e-2, max_iter=50, device="cpu", verbose=True):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        # self.preprocessor = preprocessor
        self.lr = lr
        self.max_iter = max_iter
        self.device = device
        
        self.verbose = verbose

        self.loss_history = []

    def get_logits(self, input):
        # if self.preprocessor:
        #     input = self.preprocessor(input.float())
        #     input = input.transpose(1, 2).flatten(2)
        logits = self.model(input)
        return logits

    def forward(self, input):
        logits = self.get_logits(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(
            logits.size(0), logits.size(1)
        )
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def fit(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in tqdm(valid_loader, disable=not self.verbose):
                input = input.cuda()
                _logits = self.get_logits(input)

                label = label.long().flatten()  # UPDATE
                _logits = _logits.flatten()  # UPDATE

                num_samples = len(label)
                logits = torch.empty(num_samples, 2)
                logits[:, 0] = 1 - _logits
                logits[:, 1] = _logits

                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()

        if self.verbose:
            print(
                "Before temperature - NLL: %.3f, ECE: %.3f"
                % (before_temperature_nll, before_temperature_ece)
            )

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS(
            [self.temperature], lr=self.lr, max_iter=self.max_iter
        )  # QQQ: parameters??

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()

            self.loss_history.append(round(loss.item(), 4))

            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(
            self.temperature_scale(logits), labels
        ).item()
        after_temperature_ece = ece_criterion(
            self.temperature_scale(logits), labels
        ).item()
        temp = self.temperature.item()

        if self.verbose:
            print("Optimal temperature: %.3f" % temp)
            print(
                "After temperature - NLL: %.3f, ECE: %.3f"
                % (after_temperature_nll, after_temperature_ece)
            )

        # update temperature after calibration
        self.model.temperature = temp

        return self

    def get_predictions(self, inputs):  # seq2seq ONLY
        # if self.preprocessor:
        #     inputs = self.preprocessor(inputs.float())
        #     inputs = inputs.transpose(1, 2).flatten(2)
        cal_preds = self.model(inputs)
        return cal_preds

    def predict_all_models(self, dataloader, verbose=False):
        test_out_bank, _, test_labels_bank = collect_model_predictions_on_set(
            self.model,
            dataloader,
            model_type=self.model.args["model_type"],
            device=self.device,
            verbose=verbose,
            # preprocessor=self.preprocessor,
        )
        preds_cal_flat = torch.vstack(test_out_bank).flatten()
        labels_flat = torch.vstack(test_labels_bank).flatten()

        return preds_cal_flat, labels_flat

class ModelBeta:
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model, parameters="abm", device="cpu"):
        super(ModelBeta, self).__init__()
        self.model = model

        self.calibrator = BetaCalibration(parameters)
        self.device = device

        try:
            self.window_1 = model.window_1
            self.window_2 = model.window_2
        except:
            pass

    def eval(self):
        self.model.eval()

    def to(self, device: str = "cpu"):
        self.model.to(device)

    # This function probably should live outside of this class, but whatever
    def fit(self, dataoader, verbose=True):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        test_out_bank, _, test_labels_bank = collect_model_predictions_on_set(
            self.model,
            dataoader,
            model_type=self.model.args["model_type"],
            device=self.device,
            verbose=verbose,
        )

        test_out_flat = torch.vstack(test_out_bank).flatten().numpy()
        test_labels_flat = torch.vstack(test_labels_bank).flatten().numpy()

        self.calibrator.fit(test_out_flat.reshape(-1, 1), test_labels_flat)

        return self

    def get_predictions(self, inputs):
        model_type = self.model.args["model_type"]

        if model_type in ["tscp", "ts2vec"]:
            step, alpha = self.model.args["predictions"].values()
        else:
            step, alpha = None, None

        preds, _, _ = get_models_predictions(
            inputs=inputs,
            labels=None,
            model=self.model,
            model_type=model_type,
            device=self.device,
            step=step,
            alpha=alpha,
        )
        preds = preds.detach().cpu()

        cal_preds = self.calibrator.predict(preds.flatten()).reshape(preds.shape)

        return torch.from_numpy(cal_preds)

    def predict_all_models(self, dataloader, verbose=False):
        test_out_bank, _, test_labels_bank = collect_model_predictions_on_set(
            self.model,
            dataloader,
            model_type=self.model.args["model_type"],
            device=self.device,
            verbose=verbose,
        )

        preds_flat = torch.vstack(test_out_bank).flatten()
        labels_flat = torch.vstack(test_labels_bank).flatten()

        preds_cal_flat = self.calibrator.predict(preds_flat)

        return preds_cal_flat, labels_flat


# ------------------------------------------------------------------------------------------------------------#
#                                         Utils for calibration                                               #
# ------------------------------------------------------------------------------------------------------------#


def calibrate_single_model(
    cpd_model,
    val_dataloader,
    cal_type="beta",
    parameters_beta="abm",
    lr=1e-2,
    max_iter=50,
    verbose=True,
    device="cpu",
):
    assert cal_type in ["beta", "temperature"], f"Unknown calibration type {cal_type}"

    cpd_model.to(device)
    
    if cal_type == "temperature":
        cpd_model.model.return_logits = True
        scaled_model = ModelWithTemperature(
            cpd_model,
            # preprocessor=preprocessor,
            lr=lr,
            max_iter=max_iter,
            device=device,
            verbose=verbose,
        )
        scaled_model.fit(val_dataloader)
        cpd_model.model.return_logits = False
        
    else:
        scaled_model = ModelBeta(
            cpd_model,
            parameters=parameters_beta,
            # preprocessor=preprocessor,
            device=device,
        )
        scaled_model.fit(val_dataloader, verbose=verbose)

    return scaled_model


def calibrate_all_models_in_ensemble(
    ensemble_model,
    val_dataloader,
    cal_type,
    lr=1e-2,
    max_iter=50,
    verbose=True,
    device="cpu",
):
    cal_models = []
    for cpd_model in ensemble_model.models_list:
        cal_model = calibrate_single_model(
            cpd_model,
            val_dataloader,
            cal_type,
            lr=lr,
            max_iter=max_iter,
            verbose=verbose,
            device=device,
        )
        cal_models.append(cal_model)

    ensemble_model.models_list = cal_models
    ensemble_model.calibrated = True

    return cal_models



def plot_calibration_curves(
    ens_model,
    test_dataloader,
    model_type="seq2seq",
    calibrated=True,
    device="cpu",
    n_bins=10,
    evaluate=False,
    fontsize=12,
    title=None,
    verbose=False,
    savename=None,
    model_num=None,
):
    if not model_num:
        model_num = len(ens_model.models_list)

    x_ideal = np.linspace(0, 1, n_bins)

    plt.figure(figsize=(8, 6))
    plt.plot(
        x_ideal,
        x_ideal,
        linestyle="--",
        label="Ideal",
        c="black",
        linewidth=2,
    )

    if evaluate:
        ece_list = []

    for i, model in enumerate(ens_model.models_list[:model_num]):
        if calibrated:
            test_out_flat, test_labels_flat = model.predict_all_models(
                test_dataloader, verbose=verbose
            )
        else:
            test_out_bank, _, test_labels_bank = collect_model_predictions_on_set(
                model,
                test_dataloader,
                model_type=model_type,
                device=device,
                verbose=verbose,
            )
            test_out_flat = torch.vstack(test_out_bank).flatten()
            test_labels_flat = torch.vstack(test_labels_bank).flatten()

        if evaluate:
            try:
                ece_list.append(ece(test_labels_flat.numpy(), test_out_flat.numpy()))
            except AttributeError:
                ece_list.append(ece(test_labels_flat, test_out_flat))
        prob_true, prob_pred = calibration_curve(
            test_labels_flat, test_out_flat, n_bins=n_bins
        )

        plt.plot(
            prob_pred,
            prob_true,
            linestyle="--",
            marker="o",
            markersize=4,
            linewidth=1,
            label=f"Model {i}",
        )
    if evaluate:
        bbox = dict(boxstyle="round", fc="blanchedalmond", ec="orange", alpha=0.5)
        plt.text(
            x=0.49,
            y=0.00,
            s="Calibration Error = {:.4f}".format(np.round(np.mean(ece_list), 4)),  # noqa: F523
            fontsize=fontsize,
            bbox=bbox,
        )
    if title:
        plt.title(title, fontsize=fontsize + 2)
    plt.xlabel("Predicted probability", fontsize=fontsize)
    plt.ylabel("Fraction of positives", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize - 1)
    plt.tight_layout()
    if savename:
        plt.savefig(f"pictures/calibration/curves/{savename}.pdf", dpi=300)
    plt.show()
