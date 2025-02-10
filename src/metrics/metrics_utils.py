import gc
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from src.baselines import prediction_utils
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------------------------------------------------------------------------------------------------#
#                                         Calculate CPD metrics                                               #
# ------------------------------------------------------------------------------------------------------------#


def find_first_change(mask: np.array) -> np.array:
    """Find first change in batch of predictions.

    :param mask:
    :return: mask with -1 on first change
    """
    change_ind = torch.argmax(mask.int(), axis=1)
    no_change_ind = torch.sum(mask, axis=1)
    change_ind[torch.where(no_change_ind == 0)[0]] = -1
    return change_ind


def calculate_errors(
    real: torch.Tensor, pred: torch.Tensor, seq_len: int
) -> Tuple[int, int, int, int, List[float], List[float]]:
    """Calculate confusion matrix, detection delay and time to false alarms.

    :param real: real true change points idxs for a batch
    :param pred: predicted change point idxs for a batch
    :param seq_len: length of sequence
    :return: tuple of
        - TN, FP, FN, TP
        - array of times to false alarms
        - array of detection delays
    """
    FP_delay = torch.zeros_like(real, requires_grad=False)
    delay = torch.zeros_like(real, requires_grad=False)

    tn_mask = torch.logical_and(real == pred, real == -1)
    fn_mask = torch.logical_and(real != pred, pred == -1)
    tp_mask = torch.logical_and(real <= pred, real != -1)
    fp_mask = torch.logical_or(
        torch.logical_and(torch.logical_and(real > pred, real != -1), pred != -1),
        torch.logical_and(pred != -1, real == -1),
    )

    TN = tn_mask.sum().item()
    FN = fn_mask.sum().item()
    TP = tp_mask.sum().item()
    FP = fp_mask.sum().item()

    FP_delay[tn_mask] = seq_len
    FP_delay[fn_mask] = seq_len
    FP_delay[tp_mask] = real[tp_mask]
    FP_delay[fp_mask] = pred[fp_mask]

    delay[tn_mask] = 0
    delay[fn_mask] = seq_len - real[fn_mask]
    delay[tp_mask] = pred[tp_mask] - real[tp_mask]
    delay[fp_mask] = 0

    assert (TN + TP + FN + FP) == len(real)

    return TN, FP, FN, TP, FP_delay, delay


def calculate_conf_matrix_margin(
    real: torch.Tensor, pred: torch.Tensor, margin: int
) -> Tuple[int, int, int, int]:
    """Calculate confusion matrix, detection delay and time to false alarms.

    :param real: real labels of change points
    :param pred: predicted labels (0 or 1) of change points
    :param margin: if |true_cp_idx - pred_cp_idx| <= margin, report TP
    :return: tuple of (TN, FP, FN, TP)
    """
    tn_mask_margin = torch.logical_and(real == pred, real == -1)
    fn_mask_margin = torch.logical_and(real != pred, pred == -1)

    tp_mask_margin = torch.logical_and(
        torch.logical_and(torch.abs(real - pred) <= margin, real != -1), pred != -1
    )

    fp_mask_margin = torch.logical_or(
        torch.logical_and(
            torch.logical_and(torch.abs(real - pred) > margin, real != -1), pred != -1
        ),
        torch.logical_and(pred != -1, real == -1),
    )

    TN_margin = tn_mask_margin.sum().item()
    FN_margin = fn_mask_margin.sum().item()
    TP_margin = tp_mask_margin.sum().item()
    FP_margin = fp_mask_margin.sum().item()

    assert (TN_margin + TP_margin + FN_margin + FP_margin) == len(
        real
    ), "Check TP, TN, FP, FN cases."

    return TN_margin, FP_margin, FN_margin, TP_margin


def area_under_graph(delay_list: List[float], fp_delay_list: List[float]) -> float:
    """Calculate area under Delay - FP delay curve.

    :param delay_list: list of delays
    :param fp_delay_list: list of fp delays
    :return: area under curve
    """
    return np.trapz(delay_list, fp_delay_list)


def overlap(A: set, B: set):
    """Return the overlap (i.e. Jaccard index) of two sets.

    :param A: set #1
    :param B: set #2
    return Jaccard index of the 2 sets
    """
    return len(A.intersection(B)) / len(A.union(B))


def partition_from_cps(locations: List[int], n_obs: int) -> List[set]:
    """Return a list of sets that give a partition of the set [0, T-1], as
    defined by the change point locations.

    :param locations: idxs of the change points
    :param n_obs: length of the sequence
    :return partition of the sequence (list of sets with idxs)
    """
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(n_obs):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition


def cover_single(true_partitions: List[set], pred_partitions: List[set]) -> float:
    """Compute the covering of a true segmentation by a predicted segmentation.

    :param true_partitions: partition made by true CPs
    :param true_partitions: partition made by predicted CPs
    """
    seq_len = sum(map(len, pred_partitions))
    assert seq_len == sum(map(len, true_partitions))

    cover = 0
    for t_part in true_partitions:
        cover += len(t_part) * max(
            overlap(t_part, p_part) for p_part in pred_partitions
        )
    cover /= seq_len
    return cover


def calculate_cover(
    real_change_ind: List[torch.Tensor], predicted_change_ind: List[torch.Tensor], seq_len: int
) -> List[float]:
    """Calculate covering for a given sequence.

    :param real_change_ind: indexes of true CPs
    :param predicted_change_ind: indexes of predicted CPs
    :param seq_len: length of the sequence
    :return cover
    """
    covers = []

    for real, pred in zip(real_change_ind, predicted_change_ind):
        true_partition = partition_from_cps([real.item()], seq_len)
        pred_partition = partition_from_cps([pred.item()], seq_len)
        covers.append(cover_single(true_partition, pred_partition))
    return covers


def F1_score(confusion_matrix: Tuple[int, int, int, int]) -> float:
    """Calculate F1-score.

    :param confusion_matrix: tuple with elements of the confusion matrix
    :return: f1_score
    """
    TN, FP, FN, TP = confusion_matrix
    f1_score = 2.0 * TP / (2 * TP + FN + FP)
    return f1_score


def calculate_metrics(
    true_labels: torch.Tensor, predictions: torch.Tensor, margin_list: Optional[List[int]] = None
):
    """Calculate confusion matrix, detection delay, time to false alarms, covering.

    :param true_labels: true labels (0 or 1) of change points
    :param predictions: predicted labels (0 or 1) of change points
    :return: tuple of
        - TN, FP, FN, TP
        - array of times to false alarms
        - array of detection delays
        - covering
    """
    mask_real = ~true_labels.eq(true_labels[:, 0][0])
    mask_predicted = ~predictions.eq(true_labels[:, 0][0])
    seq_len = true_labels.shape[1]

    real_change_ind = find_first_change(mask_real)
    predicted_change_ind = find_first_change(mask_predicted)

    TN, FP, FN, TP, FP_delay, delay = calculate_errors(
        real_change_ind, predicted_change_ind, seq_len
    )
    cover = calculate_cover(real_change_ind, predicted_change_ind, seq_len)

    TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict = (
        None,
        None,
        None,
        None,
    )

    # add margin metrics
    if margin_list is not None:
        TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict = {}, {}, {}, {}
        for margin in margin_list:
            TN_margin, FP_margin, FN_margin, TP_margin = calculate_conf_matrix_margin(
                real_change_ind, predicted_change_ind, margin
            )
            TN_margin_dict[margin] = TN_margin
            FP_margin_dict[margin] = FP_margin
            FN_margin_dict[margin] = FN_margin
            TP_margin_dict[margin] = TP_margin

    return (TN, FP, FN, TP, FP_delay, delay, cover), (
        TN_margin_dict,
        FP_margin_dict,
        FN_margin_dict,
        TP_margin_dict,
    )


def evaluate_metrics_on_set(
    test_out_bank: List[torch.Tensor],
    test_uncertainties_bank: List[torch.Tensor],
    test_labels_bank: List[torch.Tensor],
    threshold: float = 0.5,
    verbose: bool = True,
    device: str = "cuda",
    margin_list: Optional[List[int]] = None,
) -> Tuple[int, int, int, int, float, float]:
    """Calculate metrics for CPD.

    :param model: trained CPD model for evaluation
    :param test_loader: dataloader with test data
    :param threshold: alarm threshold (if change prob > threshold, report about a CP)
    :param verbose: if True, print the results
    :param model_type: type of the model ('seq2seq', 'kl_cpd', 'tscp', baselines)
    :param device: 'cuda' or 'cpu'
    :param q: probability for quantile-based predictions of the EnsembleCPDModel, set to 'None' is no ensemble is used
    :return: tuple of
        - TN, FP, FN, TP
        - mean time to a false alarm
        - mean detection delay
        - mean covering
    """
    FP_delays = []
    delays = []
    covers = []
    TN, FP, FN, TP = (0, 0, 0, 0)

    TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict = (
        None,
        None,
        None,
        None,
    )

    if margin_list is not None:
        TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict = (
            {},
            {},
            {},
            {},
        )
        for margin in margin_list:
            TN_margin_dict[margin] = 0
            FP_margin_dict[margin] = 0
            FN_margin_dict[margin] = 0
            TP_margin_dict[margin] = 0

    with torch.no_grad():
        for test_out, _, test_labels in zip(
            test_out_bank, test_uncertainties_bank, test_labels_bank
        ):
            cropped_outs = test_out > threshold

            (
                (tn, fp, fn, tp, FP_delay, delay, cover),
                (tn_margin_dict, fp_margin_dict, fn_margin_dict, tp_margin_dict),
            ) = calculate_metrics(test_labels, cropped_outs, margin_list)

            TN += tn
            FP += fp
            FN += fn
            TP += tp

            if margin_list is not None:
                for margin in margin_list:
                    TN_margin_dict[margin] += tn_margin_dict[margin]
                    FP_margin_dict[margin] += fp_margin_dict[margin]
                    FN_margin_dict[margin] += fn_margin_dict[margin]
                    TP_margin_dict[margin] += tp_margin_dict[margin]

            FP_delays.append(FP_delay.detach().cpu())
            delays.append(delay.detach().cpu())
            covers.extend(cover)

    mean_FP_delay = torch.cat(FP_delays).float().mean().item()
    mean_delay = torch.cat(delays).float().mean().item()
    mean_cover = np.mean(covers)

    if verbose:
        print(
            "TN: {}, FP: {}, FN: {}, TP: {}, DELAY:{}, FP_DELAY:{}, COVER: {}".format(
                TN, FP, FN, TP, mean_delay, mean_FP_delay, mean_cover
            )
        )

    del FP_delays, delays, covers
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()

    return (
        (TN, FP, FN, TP, mean_delay, mean_FP_delay, mean_cover),
        (TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict),
    )


# ------------------------------------------------------------------------------------------------------------#
#                                         Collect predictions                                                 #
# ------------------------------------------------------------------------------------------------------------#


def get_models_predictions(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    model: torch.nn.Module,
    model_type: str = "seq2seq",
    device: str = "cuda",
    q: Optional[float] = None,
    step: int = 1,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get model's prediction.

    :param inputs: input data
    :param labels: true labels
    :param model: CPD model
    :param model_type: default "seq2seq" for BCE/InDiD model, TODO
    :param device: device name
    :param q: probability for quantile-based predictions of the EnsembleCPDModel, set to 'None' is no ensemble is used
    :return: model's predictions
    """
    try:
        inputs = inputs.to(device)
    except AttributeError:
        inputs = [t.to(device) for t in inputs]

    if labels is not None:
        true_labels = labels.to(device)
    else:
        true_labels = labels

    if model_type == "tscp":
        outs = prediction_utils.get_repr_learning_output(
            model, inputs, model.window_1, model.window_2, step=step, max_pool=False
        )
        outs = prediction_utils.post_process_output(outs, alpha=alpha)
        uncertainties = None

    elif model_type == "ts2vec":
        outs = prediction_utils.get_repr_learning_output(
            model, inputs, model.window_1, model.window_2, step=step, max_pool=True
        )
        outs = prediction_utils.post_process_output(outs, alpha=alpha)
        uncertainties = None

    elif model_type == "ensemble":
        # take mean values and std (as uncertainty measure)
        outs, uncertainties = model.predict(inputs, step=step, alpha=alpha)

    elif model_type == "ensemble_all_models":
        outs = model.predict_all_models(inputs, step=step, alpha=alpha)
        uncertainties = None

    elif model_type == "ensemble_quantile":
        outs, uncertainties = model.get_quantile_predictions(
            inputs, q, step=step, alpha=alpha
        )

    elif model_type == "mmd_aggr":
        outs = model.predict(inputs, step=step, alpha=alpha)
        uncertainties = None

    elif model_type == "seq2seq":
        outs = model(inputs)
        uncertainties = None

    elif model_type == "seq2seq_cal":
        outs = model.get_predictions(inputs)
        uncertainties = None

    elif model_type == "fake_ensemble":
        outs, uncertainties = inputs[0], inputs[1]

    elif model_type == "fake_mmd":
        #outs, _ = model.fake_predict(inputs)
        outs = model.fake_predict(inputs)
        uncertainties = None

    else:
        raise ValueError(f"Wrong model type {model_type}.")

    return outs, uncertainties, true_labels


def collect_model_predictions_on_set(
    model: nn.Module,
    test_loader: DataLoader,
    verbose: bool = True,
    model_type: str = "seq2seq",
    device: str = "cuda",
    q: Optional[float] = None,
    step: int = 1,
    alpha: float = 1.0,
):
    if model is not None:
        model.eval()
        model.to(device)

    test_out_bank, test_uncertainties_bank, test_labels_bank = [], [], []

    with torch.no_grad():
        if verbose:
            print("Collectting model's outputs")

        # collect model's predictions once and reuse them
        # for test_inputs, test_labels in tqdm(test_loader):
        for test_inputs, test_labels in tqdm(test_loader, disable=not verbose):
            test_out, test_uncertainties, test_labels = get_models_predictions(
                test_inputs,
                test_labels,
                model,
                model_type=model_type,
                device=device,
                q=q,
                step=step,
                alpha=alpha,
            )

            try:
                test_out = test_out.squeeze(2)
                test_uncertainties = test_uncertainties.squeeze(2)
            except:  # noqa: E722
                pass

            # in case of different sizes, crop start of labels sequence (for TS-CP)
            crop_size = test_labels.shape[-1] - test_out.shape[-1]
            test_labels = test_labels[:, crop_size:]

            test_out_bank.append(test_out.cpu())
            test_uncertainties_bank.append(
                test_uncertainties.cpu()
                if test_uncertainties is not None
                else test_uncertainties
            )
            test_labels_bank.append(test_labels.cpu())

    del test_labels, test_out, test_uncertainties, test_inputs
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()

    return test_out_bank, test_uncertainties_bank, test_labels_bank


# ------------------------------------------------------------------------------------------------------------#
#                                              Save results                                                  #
# ------------------------------------------------------------------------------------------------------------#
def write_metrics_to_file(
    filename: str,
    metrics: tuple,
    seed: int,
    timestamp: str,
    comment: Optional[str] = None,
) -> None:
    """Write metrics to a .txt file.

    :param filename: path to the .txt file
    :param metrics: tuple of metrics (output of the 'evaluation_pipeline' function)
    :param seed: initialization seed for the model under evaluation
    :param timestamp: timestamp indicating which model was evaluated
    :param comment: any additional information about the experiment
    """
    (
        (
            best_th_f1,
            best_time_to_FA,
            best_delay,
            auc,
            best_conf_matrix,
            best_f1,
            best_cover,
            best_th_cover,
            max_cover,
        ),
        (best_th_f1_margin_dict, max_f1_margin_dict),
        _,
        _,
    ) = metrics

    with open(filename, "a") as f:
        f.writelines("Comment: {}\n".format(comment))
        f.writelines("SEED: {}\n".format(seed))
        f.writelines("Timestamp: {}\n".format(timestamp))
        f.writelines("AUC: {}\n".format(auc))
        f.writelines(
            "Time to FA {}, delay detection {} for best-F1 threshold: {}\n".format(
                round(best_time_to_FA, 4), round(best_delay, 4), round(best_th_f1, 4)
            )
        )
        f.writelines(
            "TN {}, FP {}, FN {}, TP {} for best-F1 threshold: {}\n".format(
                best_conf_matrix[0],
                best_conf_matrix[1],
                best_conf_matrix[2],
                best_conf_matrix[3],
                round(best_th_f1, 4),
            )
        )
        f.writelines(
            "Max F1 {}: for best-F1 threshold {}\n".format(
                round(best_f1, 4), round(best_th_f1, 4)
            )
        )
        f.writelines(
            "COVER {}: for best-F1 threshold {}\n".format(
                round(best_cover, 4), round(best_th_f1, 4)
            )
        )

        f.writelines(
            "Max COVER {}: for threshold {}\n".format(max_cover, best_th_cover)
        )
        if max_f1_margin_dict is not None:
            for margin, max_f1_margin in max_f1_margin_dict.items():
                f.writelines(
                    "Max F1 with margin {}: {} for threshold {}\n".format(
                        margin, max_f1_margin, best_th_f1_margin_dict[margin]
                    )
                )
        f.writelines(
            "----------------------------------------------------------------------\n"
        )


def dump_results(metrics_local: tuple, pickle_name: str) -> None:
    """Save result metrics as a .pickle file."""
    (
        best_th_f1,
        best_time_to_FA,
        best_delay,
        auc,
        best_conf_matrix,
        best_f1,
        best_cover,
        best_th_cover,
        max_cover,
    ) = metrics_local
    results = dict(
        best_th_f1=best_th_f1,
        best_time_to_FA=best_time_to_FA,
        best_delay=best_delay,
        auc=auc,
        best_conf_matrix=best_conf_matrix,
        best_f1=best_f1,
        best_cover=best_cover,
        best_th_cover=best_th_cover,
        max_cover=max_cover,
    )

    with Path(pickle_name).open("wb") as f:
        pickle.dump(results, f)
