from typing import Any

import matplotlib.pyplot as plt
import torch
from src.baselines.prediction_utils import get_repr_learning_output, post_process_output


def visualize_predictions(
    model: Any,
    model_type: str,
    sequences_batch: torch.Tensor,
    labels_batch: torch.Tensor,
    n_pics: int = 10,
    save_path: str = None,
    scale: int = None,
    step: int = 1,
    alpha: float = 1.0,
    device: str = "cpu",
    batch_num_prefix: int = 0,
) -> None:
    """Visualize model's predictions for a batch of test sequences.

    :param model: trained model (e.g. CPDModel or EnsembleCPDModel)
    :param sequences_batch: batch of test sequences
    :param lavels_batch: batch of corresponding labels
    :param n_pics: number of pictures to plot
    :param save: if True, save pictures to the 'pictures' folder
    """
    print("device:", device)
    model.to(device)
    sequences_batch = sequences_batch.to(device)
    labels_batch = labels_batch.cpu()

    if len(sequences_batch) < n_pics:
        print("Desired number of pictures is greater than size of the batch provided.")
        n_pics = len(sequences_batch)

    if model_type == "seq2seq":
        preds = model(sequences_batch).cpu()
        std = torch.zeros_like(preds)

    elif model_type in ["tscp", "ts2vec"]:
        preds = get_repr_learning_output(
            model,
            sequences_batch,
            window_1=model.window_1,
            window_2=model.window_2,
            step=step,
        )
        preds = post_process_output(preds, alpha=alpha)
        std = torch.zeros_like(preds)

    elif model_type == "ensemble":
        preds, std = model.predict(sequences_batch, scale=scale, step=step, alpha=alpha)
        std = std.detach().cpu().squeeze()
    else:
        raise ValueError(f"Unkown model type: {model_type}")

    preds = preds.detach().cpu().squeeze()

    # crop zero padding for TS-CP models / ensembles of TS-CP models
    crop_size = labels_batch.shape[1] - preds.shape[1]
    labels_batch = labels_batch[:, crop_size:]

    for idx in range(n_pics):
        plt.figure()
        plt.plot(preds[idx], label="Predictions")

        if model_type == "ensemble":  # plot std
            plt.fill_between(
                range(len(preds[idx])),
                preds[idx] - std[idx],
                preds[idx] + std[idx],
                alpha=0.3,
            )
        plt.plot(labels_batch[idx], label="Labels")
        plt.title("Predictions", fontsize=14)
        plt.legend(fontsize=12)
        if save_path is not None:
            plt.savefig(f"{save_path}/batch_{batch_num_prefix}_seq_{idx}.png")
        plt.show()
