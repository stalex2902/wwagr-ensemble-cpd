from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def _cosine_simililarity_dim1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute 1D cosine distance between 2 tensors.

    :params x, y: input tensors
    :return: cosine distance
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    v = cos(x, y)
    return v


def _cosine_simililarity_dim2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute 2D cosine distance between 2 tensors.

    :params x, y: input tensors
    :return: cosine distance
    """
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    v = cos(x.unsqueeze(1), y.unsqueeze(0))
    return v


def history_future_separation(
    data: torch.Tensor, window: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split sequences in batch on two equal slices.

    :param data: input sequences
    :param window: slice size
    :return: set of "past" subsequences and corresponded "future" subsequences
    """
    if len(data.shape) <= 4:
        history = data[:, :window]
        future = data[:, window : 2 * window]
    elif len(data.shape) == 5:
        history = data[:, :, :window]
        future = data[:, :, window : 2 * window]

    return history, future


def history_future_separation_test(
    data: torch.Tensor,
    window_1: int,
    window_2: Optional[int] = None,
    step: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare data for testing. Separate it in set of "past"-"future" slices.

    :param data: input sequence
    :param window_1: "past" subsequence size
    :param window_2: "future" subsequence size (default None), if None set equal to window_1
    :param step: step size
    :return: set of "past" subsequences and corresponded "future" subsequences
    """
    future_slices = []
    history_slices = []

    if window_2 is None:
        window_2 = window_1

    if len(data.shape) > 4:
        data = data.transpose(1, 2)

    seq_len = data.shape[1]
    for i in range(0, (seq_len - window_1 - window_2) // step + 1):
        start_ind = i * step
        end_ind = window_1 + window_2 + step * i
        slice_2w = data[:, start_ind:end_ind]
        history_slices.append(slice_2w[:, :window_1].unsqueeze(0))
        future_slices.append(slice_2w[:, window_1:].unsqueeze(0))

    future_slices = torch.cat(future_slices).transpose(0, 1)
    history_slices = torch.cat(history_slices).transpose(0, 1)

    # in case of video data
    if len(data.shape) > 4:
        history_slices = history_slices.transpose(2, 3)
        future_slices = future_slices.transpose(2, 3)

    return history_slices, future_slices


def ema_batch(outputs, alpha):
    assert alpha >= 0 and alpha <= 1, "Smoothing factor alpha should be in [0, 1]."
    seq_len = outputs.shape[1]
    ema_outputs = outputs.clone()
    for t in range(1, seq_len):
        ema_outputs[:, t] = alpha * outputs[:, t] + (1 - alpha) * ema_outputs[:, t - 1]

    return ema_outputs


def get_repr_learning_output(
    model: nn.Module,
    batch: torch.Tensor,
    window_1: int,
    window_2: Optional[int] = None,
    step: int = 1,
    max_pool: bool = False,
) -> List[torch.Tensor]:
    """Get TS-CP2 predictions scaled to [0, 1].

    :param tscp_model: pre-trained TS-CP2 model
    :param batch: input data
    :param window_1: "past" subsequence size
    :param window_2: "future" subsequence size (default None), if None set equal to window_1
    :return: predicted change probability
    """
    model.eval()

    device = model.device
    batch = batch.to(device)

    if len(batch.shape) <= 4:
        seq_len = batch.shape[1]
    else:
        seq_len = batch.shape[2]

    batch_history_slices, batch_future_slices = history_future_separation_test(
        batch, window_1, window_2, step=step
    )

    pred_out = []
    crop_size = seq_len - (window_1 + window_2 - 1)
    for history_slice, future_slice in zip(batch_history_slices, batch_future_slices):
        zeros = torch.ones(1, seq_len)

        curr_history, curr_future = map(model, [history_slice, future_slice])

        if max_pool:
            curr_history = torch.max(curr_history, axis=1)[0]
            curr_future = torch.max(curr_future, axis=1)[0]

        rep_sim = _cosine_simililarity_dim1(curr_history, curr_future).data
        rep_sim = torch.repeat_interleave(rep_sim, step, dim=0)[:crop_size]

        zeros[:, window_1 + window_2 - 1 :] = rep_sim

        pred_out.append(zeros)

    pred_out = torch.cat(pred_out).to(batch.device)
    pred_out = pred_out[:, window_1 + window_2 - 1 :]

    return pred_out


def post_process_output(outputs: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    preds = 1.0 - torch.where(outputs >= 0, outputs, torch.zeros_like(outputs))
    preds = ema_batch(preds, alpha)
    return preds
