import os
from abc import ABC

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from src.baselines import prediction_utils
from src.datasets import datasets
from src.ensembles import distances
from src.models import model_utils
from src.utils.fix_seeds import fix_seeds
from torch.utils.data import Subset

EPS = 1e-6


class EnsembleCPDModel(ABC):
    """Wrapper for general ensemble models with bootstrapping."""

    def __init__(
        self,
        args: dict,
        n_models: int,
        boot_sample_size: int = None,
        train_anomaly_num: int = None,
    ) -> None:
        """Initialize EnsembleCPDModel.

        :param args: dictionary containing core model params, learning params, loss params, etc.
        :param n_models: number of models to train
        :param boot_sample_size: size of the bootstrapped train dataset
                                 (if None, all the models are trained on the original train dataset)
        """
        super().__init__()

        self.args = args

        assert args["experiments_name"] in [
            "human_activity",
            "explosion",
            "road_accidents",
            "yahoo",
        ], "Wrong experiments name"

        self.train_dataset, self.test_dataset = datasets.CPDDatasets(
            experiments_name=args["experiments_name"],
            train_anomaly_num=train_anomaly_num,
        ).get_dataset_()

        self.n_models = n_models

        if boot_sample_size is not None:
            assert boot_sample_size <= len(
                self.train_dataset
            ), "Desired sample size is larger than the whole train dataset."
        self.boot_sample_size = boot_sample_size

        self.fitted = False
        self.initialize_models_list()

        self.calibrated = False

    def eval(self) -> None:
        """Turn all the models to 'eval' mode (for consistency with our code)."""
        for model in self.models_list:
            model.eval()

    def to(self, device: str) -> None:
        """Move all models to the device (for consistency with our code)."""
        for model in self.models_list:
            model.to(device)

    def bootstrap_datasets(self) -> None:
        """Generate new train datasets if necessary."""
        # No boostrap
        if self.boot_sample_size is None:
            self.train_datasets_list = [self.train_dataset] * self.n_models

        else:
            # for reproducibility of torch.randint()
            torch.manual_seed(42)

            self.train_datasets_list = []
            for _ in range(self.n_models):
                # sample with replacement
                idxs = torch.randint(
                    len(self.train_dataset), size=(self.boot_sample_size,)
                )
                curr_train_data = Subset(self.train_dataset, idxs)
                self.train_datasets_list.append(curr_train_data)

    def initialize_models_list(self) -> None:
        """Initialize cpd models for a particular exeriment."""
        self.bootstrap_datasets()

        self.models_list = []
        for i in range(self.n_models):
            fix_seeds(i)

            curr_model = model_utils.get_models_list(
                self.args, self.train_datasets_list[i], self.test_dataset
            )[
                -1
            ]  # list consists of 1 model as, currently, we do not work with 'combined' models
            self.models_list.append(curr_model)

    def fit(self) -> None:
        """Fit all the models on the corresponding train datasets."""
        logger = TensorBoardLogger(
            save_dir=f'logs/{self.args["experiments_name"]}',
            name=self.args["model_type"],
        )

        if not self.fitted:
            self.initialize_models_list()
            for i, cpd_model in enumerate(self.models_list):
                fix_seeds(i)

                print(f"\nFitting model number {i + 1}.")
                trainer = pl.Trainer(
                    max_epochs=self.args["learning"]["epochs"],
                    accelerator=self.args["learning"]["accelerator"],
                    devices=self.args["learning"]["devices"],
                    benchmark=True,
                    check_val_every_n_epoch=1,
                    logger=logger,
                    callbacks=EarlyStopping(**self.args["early_stopping"]),
                )
                trainer.fit(cpd_model)

            self.fitted = True

        else:
            print("Attention! Models are already fitted!")

    def predict_all_models(
        self, inputs: torch.Tensor, step: int = 1, alpha: float = 1.0
    ):
        if not self.fitted:
            print("Attention! The model is not fitted yet.")

        self.eval()

        ensemble_preds = []
        for model in self.models_list:
            inputs = inputs.to(model.device)
            if self.args["model_type"] == "seq2seq":
                if self.calibrated:
                    outs = model.get_predictions(inputs).squeeze()
                else:
                    outs = model(inputs).squeeze()
            elif self.args["model_type"] == "tscp":
                outs = prediction_utils.get_repr_learning_output(
                    model,
                    inputs,
                    model.window_1,
                    model.window_2,
                    step=step,
                    max_pool=False,
                )
                outs = prediction_utils.post_process_output(outs, alpha=alpha)

            elif self.args["model_type"] == "ts2vec":
                outs = prediction_utils.get_repr_learning_output(
                    model,
                    inputs,
                    model.window_1,
                    model.window_2,
                    step=step,
                    max_pool=True,
                )
                outs = prediction_utils.post_process_output(outs, alpha=alpha)

            else:
                raise ValueError(
                    f'Wrong or not implemented model type {self.args["model_type"]}.'
                )
            ensemble_preds.append(outs)

        # shape is (n_models, batch_size, seq_len)
        ensemble_preds = torch.stack(ensemble_preds)
        if len(ensemble_preds.shape) < 3:
            ensemble_preds = ensemble_preds.unsqueeze(0)
        self.preds = ensemble_preds

        return ensemble_preds

    def predict(
        self, inputs: torch.Tensor, step: int = 1, alpha: float = 1.0
    ) -> torch.Tensor:
        """Make a prediction.

        :param inputs: input batch of sequences

        :returns: torch.Tensor containing predictions of all the models
        """
        ensemble_preds = self.predict_all_models(inputs, step, alpha)

        _, batch_size, seq_len = ensemble_preds.shape

        preds_mean = torch.mean(ensemble_preds, axis=0).reshape(batch_size, seq_len)
        preds_std = torch.std(ensemble_preds, axis=0).reshape(batch_size, seq_len)

        return preds_mean, preds_std

    def get_quantile_predictions(
        self,
        inputs: torch.Tensor,
        q: float,
        step: int = 1,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Get the q-th quantile of the predicted CP scores distribution.

        :param inputs: input batch of sequences
        :param q: desired quantile

        :returns: torch.Tensor containing quantile predictions
        """
        ensemble_preds = self.predict_all_models(inputs, step, alpha)
        _, batch_size, seq_len = ensemble_preds.shape

        preds_quantile = torch.quantile(ensemble_preds, q, axis=0).reshape(
            batch_size, seq_len
        )
        preds_std = torch.std(ensemble_preds, axis=0).reshape(batch_size, seq_len)

        return preds_quantile, preds_std

    def save_models_list(self, path_to_folder: str) -> None:
        """Save trained models.

        :param path_to_folder: path to the folder for saving, e.g. 'saved_models/mnist'
        """

        if not self.fitted:
            print("Attention! The models are not trained.")

        loss_type = (
            self.args["loss_type"] if self.args["model_type"] == "seq2seq" else None
        )

        for i, model in enumerate(self.models_list):
            path = (
                path_to_folder
                + "/"
                + self.args["experiments_name"]
                + "_loss_type_"
                + str(loss_type)
                + "_model_type_"
                + self.args["model_type"]
                + "_sample_"
                + str(self.boot_sample_size)
                + "_model_num_"
                + str(i)
                + ".pth"
            )
            torch.save(model.state_dict(), path)

    def load_models_list(self, path_to_folder: str) -> None:
        """Load weights of the saved models from the ensemble.

        :param path_to_folder: path to the folder for saving, e.g. 'saved_models/mnist'
        """
        # check that the folder contains self.n_models files with models' weights,
        # ignore utility files
        paths_list = [
            path for path in os.listdir(path_to_folder) if not path.startswith(".")
        ]

        assert (
            len(paths_list) == self.n_models
        ), "Number of paths is not equal to the number of models."

        # initialize models list
        self.initialize_models_list()

        # load state dicts
        for model, path in zip(self.models_list, paths_list):
            try:
                model.load_state_dict(torch.load(path_to_folder + "/" + path))
            except RuntimeError:
                model.model.load_state_dict(torch.load(path_to_folder + "/" + path))

        self.fitted = True


class DistanceEnsembleCPDModel(ABC):
    def __init__(
        self,
        ens_model,
        window_size: int,
        anchor_window_type: str = "start",
        #threshold: float = 0.1,
        distance: str = "mmd",
        p: int = 1,
        kernel: str = "rbf",
    ) -> None:
        super().__init__()

        assert anchor_window_type in [
            "sliding",
            "start",
            "prev",
            "combined",
        ], "Unknown window type"
        assert distance in [
            "mmd",
            "wasserstein_1d",
            "wasserstein_nd",
        ], "Unknown distance type"

        if distance == "mmd":
            assert kernel in [
                "rbf",
                "multiscale",
            ], f"Wrong kernel type: {kernel}."

        self.ens_model = ens_model

        self.anchor_window_type = anchor_window_type
        self.distance = distance
        self.p = p
        self.window_size = window_size
        #self.threshold = threshold
        self.kernel = kernel

    def eval(self):
        self.ens_model.eval()

    def to(self, device: str) -> None:
        self.ens_model.to(device)

    def distance_detector(self, ensemble_preds: torch.Tensor):
        # anchor windows: 'start', 'prev', or 'combined'
        scores = distances.anchor_window_detector_batch(
            ensemble_preds,
            window_size=self.window_size,
            distance=self.distance,
            p=self.p,
            kernel=self.kernel,
            anchor_window_type=self.anchor_window_type,
        )
        #labels = (scores > self.threshold).to(torch.int)

        return scores #labels, scores

    def predict(
        self, inputs: torch.Tensor, step: int = 1, alpha: float = 1.0
    ) -> torch.Tensor:
        ensemble_preds = self.ens_model.predict_all_models(inputs, step, alpha)
        #preds, _ = self.distance_detector(ensemble_preds.detach())
        scores = self.distance_detector(ensemble_preds.detach())
        return scores #preds

    def fake_predict(self, ensemble_preds: torch.Tensor):
        """In case of pre-computed model outputs."""
        return self.distance_detector(ensemble_preds.transpose(0, 1))
