import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import _SpectralNorm
from torch.nn.utils.parametrize import register_parametrization
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------------------#
#                                      Loss                                             #
# --------------------------------------------------------------------------------------#


def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0.0, device=z1.device)
    # n = z1.size(1)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


def instance_contrastive_loss(z1, z2):
    B, _ = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2):
    _, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


# --------------------------------------------------------------------------------------#
#                                      Models                                           #
# --------------------------------------------------------------------------------------#


class SamePadConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        groups=1,
        spec_norm=False,
    ):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        if spec_norm:
            parametrize(SpectralMaxNorm, "weight", max_norm=1.0)(self.conv)
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        final=False,
        spec_norm=False,
    ):
        super().__init__()
        self.conv1 = SamePadConv(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            spec_norm=spec_norm,  # HERE
        )
        self.conv2 = SamePadConv(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            spec_norm=spec_norm,  # AND HERE
        )

        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final
            else None
        )

        if spec_norm and self.projector is not None:
            parametrize(SpectralMaxNorm, "weight", max_norm=1.0)(self.projector)

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, spec_norm):
        super().__init__()
        self.net = nn.Sequential(
            *[
                ConvBlock(
                    channels[i - 1] if i > 0 else in_channels,
                    channels[i],
                    kernel_size=kernel_size,
                    dilation=2**i,
                    final=(i == len(channels) - 1),
                    spec_norm=spec_norm,
                )
                for i in range(len(channels))
            ]
        )

    def forward(self, x):
        return self.net(x)


class SpectralMaxNorm(_SpectralNorm):
    def __init__(
        self,
        weight: Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
        max_norm: float = 1.0,
    ) -> None:
        super().__init__(weight, n_power_iterations, dim, eps)
        self.max_norm = max_norm

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        weight_mat = self._reshape_weight_to_matrix(weight)
        if self.training:
            self._power_method(weight_mat, self.n_power_iterations)

        u = self._u.clone(memory_format=torch.contiguous_format)
        v = self._v.clone(memory_format=torch.contiguous_format)
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        if sigma > self.max_norm:
            return self.max_norm * weight / sigma
        else:
            return weight


def parametrize(
    parametrization: type[torch.nn.Module],
    *tensor_names: str,
    silent: bool = False,
    **kwargs,
):
    """Create a function which applies the specified parametrization."""

    def _parametrize(module: torch.nn.Module):
        for tensor_name in tensor_names:
            if hasattr(module, tensor_name):
                register_parametrization(
                    module,
                    tensor_name,
                    parametrization(getattr(module, tensor_name), **kwargs),
                )
            elif not silent:
                raise ValueError(f"{tensor_name} not present in {module.__class__}")

    return _parametrize


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)

    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t : t + l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class TSEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=64,
        depth=10,
        mask_mode="binomial",
        spec_norm=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dim,
            [hidden_dim] * depth + [output_dim],
            kernel_size=3,
            spec_norm=spec_norm,
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):  # x: B x T x input_dims
        x = x.to(torch.float32)
        nan_mask = ~x.isnan().any(axis=-1)
        # x[~nan_mask] = 0
        x = x * nan_mask.unsqueeze(-1)
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = "all_true"

        if mask == "binomial":
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "continuous":
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "all_true":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == "all_false":
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == "mask_last":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        elif mask == "mask_last_20":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            for i in range(1, 20):
                mask[:, -i] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        return x


class TS2Vec(pl.LightningModule):
    """The TS2Vec model"""

    def __init__(
        self,
        args: dict,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
    ):
        """Initialize a TS2Vec model.

        Args:
        cfg (.yaml config file): must contain:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            lr (int): The learning rate.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
        after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
        after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        """

        super().__init__()
        
        # Feature extractor for video datasets
        if args["experiments_name"] in ["explosion", "road_accidents"]:
            print("Loading extractor...")
            self.extractor = torch.hub.load(
                "facebookresearch/pytorchvideo:main", "x3d_m", pretrained=True
            )
            self.extractor = nn.Sequential(*list(self.extractor.blocks[:5]))

            # freeze extractor parameters
            for param in self.extractor.parameters():
                param.requires_grad = False

        else:
            self.extractor = None

        self.lr = args["learning"]["lr"]
        self.max_train_length = args["model"]["max_train_length"]
        self.temporal_unit = args["loss"]["temporal_unit"]

        self._net = model

        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.eval_step_outputs = []

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.num_workers = args["num_workers"]
        self.batch_size = args["learning"]["batch_size"]
        self.weight_decay = args["learning"]["weight_decay"]

        self.window = args["model"]["window"]
        self.window_1 = args["model"]["window_1"]
        self.window_2 = args["model"]["window_2"]
        
    def __preprocess(self, input: torch.Tensor) -> torch.Tensor:
        """Preprocess batch before forwarding (i.e. apply extractor for video input).

        :param input: input torch.Tensor
        :return: processed input tensor to be fed into .forward method
        """
        if self.extractor:
            input = self.extractor(input.float())  # batch_size, C, seq_len, H, W
            input = input.transpose(1, 2).flatten(2) # shape is (batch_size,  C*H*W, seq_len)

        # do nothing for non-video experiments
        return input

    def forward(self, inputs) -> torch.Tensor:
        inputs = self.__preprocess(inputs)
        return self._net(inputs)

    def shared_step(self, batch, mode: str = "train") -> torch.Tensor:
        x = batch[0]
        x = self.__preprocess(x)
        x = x.to(
            torch.float32
        )  # x.shape : (batch_size, n_timestamps, n_features)
        if self.max_train_length is not None and x.size(1) > self.max_train_length:
            window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
            x = x[:, window_offset : window_offset + self.max_train_length]

        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)

        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(
            low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0)
        )
        out1 = self._net(
            take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
        )
        out1 = out1[:, -crop_l:]

        out2 = self._net(
            take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
        )
        out2 = out2[:, :crop_l]  # (batch_size, crop_l, output_dims)

        loss = hierarchical_contrastive_loss(
            out1, out2, temporal_unit=self.temporal_unit
        )
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, _):
        self.net.update_parameters(self._net)
        return self.shared_step(batch, mode="train")

    def validation_step(self, batch, _):
        return self.shared_step(batch, mode="val")

    def configure_optimizers(self):
        """
        :return: optimizer
        """
        optimizer = torch.optim.AdamW(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=100, verbose=True
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self) -> DataLoader:
        """Set train dataloader.

        :return: dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Set val dataloader.

        :return: dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )