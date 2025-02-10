import math
from typing import List, Optional

import numpy as np
import torch

from ot import wasserstein_1d
from scipy.stats import wasserstein_distance#, wasserstein_distance_nd


def MMD_batch(x, y, kernel: str = "rbf", bandwidth_range: Optional[List[float]] = None):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P,  shape = (batch_size, sample_size_1, d)
        y: second sample, distribution Q, shape = (batch_size, sample_size_2, d)
        kernel: kernel type such as "multiscale" or "rbf"
    """
    device = x.device
    bs = x.shape[0]

    xx = x @ x.transpose(1, 2)  # shape = (batch_size, sample_size_1, sample_size_1)
    yy = y @ y.transpose(1, 2)  # shape = (batch_size, sample_size_2, sample_size_2)
    zz = x @ y.transpose(1, 2)  # shape = (batch_size, , sample_size_1, sample_size_2)

    rx = (
        xx.diagonal(dim1=1, dim2=2).unsqueeze(1).expand_as(xx)
    )  # (batch_size, sample_size_1, sample_size_1)
    ry = (
        yy.diagonal(dim1=1, dim2=2).unsqueeze(1).expand_as(yy)
    )  # (batch_size, sample_size_1, sample_size_1)

    dxx = rx.transpose(1, 2) + rx - 2.0 * xx  # Used for A in (1)
    dyy = ry.transpose(1, 2) + ry - 2.0 * yy  # Used for B in (1)
    dxy = rx.transpose(1, 2) + ry - 2.0 * zz  # Used for C in (1)

    XX, YY, XY = (
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
    )

    if kernel == "multiscale":
        if bandwidth_range is None:
            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx) ** -1
            YY += a**2 * (a**2 + dyy) ** -1
            XY += a**2 * (a**2 + dxy) ** -1

    if kernel == "rbf":
        if bandwidth_range is None:
            bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return (XX + YY - 2.0 * XY).view(bs, -1).mean(1)


def wasserstein_distance_1d_batch(a, b, p):
    # a, b: (batch_size, window_size, num_models)

    batch_size = a.shape[0]

    dist_batch = []

    for i in range(batch_size):
        sample_1 = a[i].flatten()  # (window_size * num_models)
        sample_2 = b[i].flatten()

        if p == 1:
            curr_dist = wasserstein_distance(sample_1, sample_2)
        else:
            curr_dist = math.pow(wasserstein_1d(sample_1, sample_2, p=p), 1 / p)

        dist_batch.append(curr_dist)

    return np.array(dist_batch)


# def wasserstein_distance_nd_batch(a, b):
#     # a, b: (batch_size, window_size, num_models)

#     batch_size = a.shape[0]

#     dist_batch = []

#     for i in range(batch_size):
#         sample_1 = a[i]  # (window_size, num_models) == (n_samples, dim_size)
#         sample_2 = b[i]

#         curr_dist = wasserstein_distance_nd(sample_1, sample_2)

#         dist_batch.append(curr_dist)

#     return np.array(dist_batch)


def anchor_window_detector_batch(
    ensemble_preds,  # output of .predict_all_model(), shape is (n_models, batch_size, seq_len)
    window_size,
    distance="mmd",
    p=1,
    kernel="rbf",
    anchor_window_type="start",
    bandwidth_range=None,
) -> torch.Tensor:
    assert distance in [
        "mmd",
        "wasserstein_1d",
        "wasserstein_nd",
    ], "Unknown distance type"

    assert anchor_window_type in ["start", "prev", "combined"], "Unknown window type"

    # ensemble_preds = ensemble_preds.numpy()

    _, batch_size, seq_len = ensemble_preds.shape

    future_idx_range = np.arange(seq_len)[window_size:]

    # first window_size elements are zeros
    dist_scores_batch = np.zeros((batch_size, seq_len))

    for future_idx in future_idx_range:
        if anchor_window_type == "start":
            anchor_wnd = ensemble_preds[:, :, :window_size]

        elif anchor_window_type == "prev":
            anchor_window_start = max(0, future_idx - 2 * window_size)
            anchor_window_end = anchor_window_start + window_size

            anchor_wnd = ensemble_preds[:, :, anchor_window_start:anchor_window_end]

        else:
            half_wnd_size = window_size // 2
            anchor_wnd = torch.cat(
                (
                    ensemble_preds[:, :, :half_wnd_size],
                    ensemble_preds[
                        :,
                        :,
                        future_idx - 2 * half_wnd_size : future_idx - half_wnd_size,
                    ],
                ),
                dim=-1,
            )

            window_size = half_wnd_size * 2

        anchor_wnd = anchor_wnd.permute(1, 2, 0)
        future_wnd = ensemble_preds[
            :, :, future_idx - window_size : future_idx
        ].permute(1, 2, 0)

        if distance == "mmd":
            dist_batch = MMD_batch(
                anchor_wnd, future_wnd, kernel=kernel, bandwidth_range=bandwidth_range
            )
        elif distance == "wasserstein_1d":
            dist_batch = wasserstein_distance_1d_batch(anchor_wnd, future_wnd, p=p)
        else:
            #dist_batch = wasserstein_distance_nd_batch(anchor_wnd, future_wnd)
            raise ValueError(f"Unknown distance {distance}")

        dist_scores_batch[:, future_idx] = dist_batch

    return torch.Tensor(dist_scores_batch)
