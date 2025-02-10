"""Methods and modules for experiments with seq2seq modeld ('indid', 'bce' and 'combided')"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.loss import loss
from torch.utils.data import DataLoader, Dataset


class CPDModel(pl.LightningModule):
    """Pytorch Lightning wrapper for change point detection models."""

    def __init__(
        self,
        loss_type: str,
        args: dict,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
    ) -> None:
        """Initialize CPD model.

        :param experiment_type: type of data used for training
        :param loss_type: type of loss function for training special CPD or common BCE loss
        :param args: dict with supplementary argumemnts
        :param model: base model
        :param train_dataset: train data
        :param test_dataset: test data
        """
        super().__init__()

        self.args = args

        self.experiments_name = args["experiments_name"]
        self.model = model

        if self.experiments_name in ["explosion", "road_accidents"]:
            self.extractor = torch.hub.load(
                "facebookresearch/pytorchvideo:main",
                "x3d_m",
                pretrained=True,
                verbose=False,
            )
            self.extractor = nn.Sequential(*list(self.extractor.blocks[:5]))

            # freeze extractor parameters
            for param in self.extractor.parameters():
                param.requires_grad = False
        else:
            self.extractor = None

        self.learning_rate = args["learning"]["lr"]
        self.batch_size = args["learning"]["batch_size"]
        self.num_workers = args["num_workers"]

        self.T = args["loss"]["T"]

        if loss_type == "indid":
            self.loss = loss.CPDLoss(len_segment=self.T)
        elif loss_type == "bce":
            self.loss = nn.BCELoss()
        else:
            raise ValueError(
                "Wrong loss_type {}. Please, choose 'indid' or 'bce' loss_type.".format(
                    loss_type
                )
            )
        self.loss_type = loss_type

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def __preprocess(self, input: torch.Tensor) -> torch.Tensor:
        """Preprocess batch before forwarding (i.e. apply extractor for video input).

        :param input: input torch.Tensor
        :return: processed input tensor to be fed into .forward method
        """
        if self.experiments_name in ["explosion", "road_accidents"]:
            input = self.extractor(input.float())
            input = input.transpose(1, 2).flatten(
                2
            )  # shape is (batch_size,  C*H*W, seq_len)

        # do nothing for non-video experiments
        return input

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward step for CPD model.

        :param inputs: batch of data
        :return: predictions
        """
        return self.model(self.__preprocess(inputs))

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Train CPD model.

        :param batch: data for training
        :param batch_idx: index of batch (special for pytorch-lightning)
        :return: loss function value
        """
        inputs, labels = batch
        pred = self.forward(inputs.float())

        train_loss = self.loss(pred.squeeze(), labels.float().squeeze())
        train_accuracy = (
            ((pred.squeeze() > 0.5).long() == labels.squeeze()).float().mean()
        )

        if self.loss_type == "indid":
            self.log("train_delay_loss", self.loss.delay_loss, prog_bar=True)
            self.log("train_fa_loss", self.loss.fa_loss, prog_bar=True)

        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", train_accuracy, prog_bar=True, on_epoch=True, on_step=False)

        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test CPD model.

        :param batch: data for validation
        :param batch_idx: index of batch (special for pytorch-lightning)
        :return: loss function value
        """
        inputs, labels = batch
        pred = self.forward(inputs.float())

        val_loss = self.loss(pred.squeeze(), labels.float().squeeze())
        val_accuracy = (
            ((pred.squeeze() > 0.5).long() == labels.squeeze()).float().mean()
        )

        if self.loss_type == "indid":
            self.log("val_delay_loss", self.loss.delay_loss, prog_bar=True)
            self.log("val_fa_loss", self.loss.fa_loss, prog_bar=True)

        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", val_accuracy, prog_bar=True, on_epoch=True, on_step=False)

        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize optimizer.

        :return: optimizer for training CPD model
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return opt

    def train_dataloader(self) -> DataLoader:
        """Initialize dataloader for training.

        :return: dataloader for training
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Initialize dataloader for validation.

        :return: dataloader for validation
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
