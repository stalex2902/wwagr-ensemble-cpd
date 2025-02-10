import itertools
import math
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
from src.datasets.datasets import AllModelsOutputDataset
from src.ensembles.ensembles import DistanceEnsembleCPDModel
from src.metrics.metrics_utils import (
    F1_score,
    area_under_graph,
    collect_model_predictions_on_set,
    evaluate_metrics_on_set,
    write_metrics_to_file,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluation_pipeline(
    model: pl.LightningModule,
    test_dataloader: DataLoader,
    threshold_list: List[float],
    device: str = "cuda",
    verbose: bool = False,
    model_type: str = "seq2seq",
    q: float = None,
    margin_list: List[int] = None,
    step: int = 1,
    alpha: float = 1.0,
) -> Tuple[Tuple[float], dict, dict]:
    """Evaluate trained CPD model.

    :param model: trained CPD model to be evaluated
    :param test_dataloader: test data for evaluation
    :param threshold_list: listh of alarm thresholds
    :param device: 'cuda' or 'cpu'
    :param verbose: if True, print the results
    :param model_type: type of the model ('seq2seq', 'tscp', 'ts2vec', 'ensemble', etc.)
    :param q: probability for quantile-based predictions of the EnsembleCPDModel, set to 'None' is no ensemble is used
    :return: tuple of
        - threshold th_1 corresponding to the maximum F1-score
        - mean time to a False Alarm corresponding to th_1
        - mean Detection Delay corresponding to th_1
        - Area under the Detection Curve
        - number of TN, FP, FN, TP corresponding to th_1
        - value of Covering corresponding to th_1
        - threshold th_2 corresponding to the maximum Covering metric
        - maximum value of Covering
    """
    try:
        model.to(device)
        model.eval()
    except AttributeError:
        print("Cannot move model to device")

    (
        test_out_bank,
        test_uncertainties_bank,
        test_labels_bank,
    ) = collect_model_predictions_on_set(
        model=model,
        test_loader=test_dataloader,
        verbose=verbose,
        model_type=model_type,
        device=device,
        q=q,
        step=step,
        alpha=alpha,
    )

    cover_dict = {}
    f1_dict = {}

    if margin_list is not None:
        final_f1_margin_dict = {}

    delay_dict = {}
    fp_delay_dict = {}
    confusion_matrix_dict = {}

    for threshold in threshold_list:
        if margin_list is not None:
            final_f1_margin_dict[threshold] = {}

        (
            (TN, FP, FN, TP, mean_delay, mean_fp_delay, cover),
            (TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict),
        ) = evaluate_metrics_on_set(
            test_out_bank=test_out_bank,
            test_uncertainties_bank=test_uncertainties_bank,
            test_labels_bank=test_labels_bank,
            threshold=threshold,
            verbose=verbose,
            device=device,
            margin_list=margin_list,
        )

        confusion_matrix_dict[threshold] = (TN, FP, FN, TP)
        delay_dict[threshold] = mean_delay
        fp_delay_dict[threshold] = mean_fp_delay

        cover_dict[threshold] = cover
        f1_dict[threshold] = F1_score((TN, FP, FN, TP))

        if margin_list is not None:
            f1_margin_dict = {}
            for margin in margin_list:
                (TN_margin, FP_margin, FN_margin, TP_margin) = (
                    TN_margin_dict[margin],
                    FP_margin_dict[margin],
                    FN_margin_dict[margin],
                    TP_margin_dict[margin],
                )
                f1_margin_dict[margin] = F1_score(
                    (TN_margin, FP_margin, FN_margin, TP_margin)
                )
            final_f1_margin_dict[threshold] = f1_margin_dict

    # fix dict structure
    if margin_list is not None:
        final_f1_margin_dict_fixed = {}
        for margin in margin_list:
            f1_scores_for_margin_dict = {}
            for threshold in threshold_list:
                f1_scores_for_margin_dict[threshold] = final_f1_margin_dict[threshold][
                    margin
                ]
            final_f1_margin_dict_fixed[margin] = f1_scores_for_margin_dict

    auc = area_under_graph(list(delay_dict.values()), list(fp_delay_dict.values()))

    # Conf matrix and F1
    best_th_f1 = max(f1_dict, key=f1_dict.get)

    best_conf_matrix = (
        confusion_matrix_dict[best_th_f1][0],
        confusion_matrix_dict[best_th_f1][1],
        confusion_matrix_dict[best_th_f1][2],
        confusion_matrix_dict[best_th_f1][3],
    )
    best_f1 = f1_dict[best_th_f1]

    # Cover
    best_cover = cover_dict[best_th_f1]

    best_th_cover = max(cover_dict, key=cover_dict.get)
    max_cover = cover_dict[best_th_cover]

    if margin_list is not None:
        max_f1_margins_dict = {}
        max_th_f1_margins_dict = {}
        for margin in margin_list:
            curr_max_th_f1_margin = max(
                final_f1_margin_dict_fixed[margin],
                key=final_f1_margin_dict_fixed[margin].get,
            )
            max_th_f1_margins_dict[margin] = curr_max_th_f1_margin
            max_f1_margins_dict[margin] = final_f1_margin_dict_fixed[margin][
                curr_max_th_f1_margin
            ]
    else:
        max_f1_margins_dict, max_th_f1_margins_dict = None, None

    # Time to FA, detection delay
    best_time_to_FA = fp_delay_dict[best_th_f1]
    best_delay = delay_dict[best_th_f1]

    if verbose:
        print("AUC:", round(auc, 4) if auc is not None else auc)
        print(
            "Time to FA {}, delay detection {} for best-F1 threshold: {}".format(
                round(best_time_to_FA, 4), round(best_delay, 4), round(best_th_f1, 4)
            )
        )
        print(
            "TN {}, FP {}, FN {}, TP {} for best-F1 threshold: {}".format(
                best_conf_matrix[0],
                best_conf_matrix[1],
                best_conf_matrix[2],
                best_conf_matrix[3],
                round(best_th_f1, 4),
            )
        )
        print(
            "Max F1 {}: for best-F1 threshold {}".format(
                round(best_f1, 4), round(best_th_f1, 4)
            )
        )
        print(
            "COVER {}: for best-F1 threshold {}".format(
                round(best_cover, 4), round(best_th_f1, 4)
            )
        )

        print(
            "Max COVER {}: for threshold {}".format(
                round(cover_dict[max(cover_dict, key=cover_dict.get)], 4),
                round(max(cover_dict, key=cover_dict.get), 4),
            )
        )
        if margin_list is not None:
            for margin in margin_list:
                print(
                    "Max F1 with margin {}: {} for best threshold {}".format(
                        margin,
                        round(max_f1_margins_dict[margin], 4),
                        round(max_th_f1_margins_dict[margin], 4),
                    )
                )

    return (
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
        (max_th_f1_margins_dict, max_f1_margins_dict),
        delay_dict,
        fp_delay_dict,
    )


def evaluate_distance_ensemble_model(
    ens_model,
    threshold_list: List[float],
    output_dataloader: DataLoader,
    margin_list: List[int],
    window_size: int,
    anchor_window_type: str = "start",
    distance: str = "wasserstein",
    p: int = 1,
    kernel: Optional[str] = "rbf",
    device: str = "cpu",
    verbose: bool = True,
    write_metrics_filename: Optional[str] = None,
):
    #res_dict = {}
    #best_th = None
    #best_f1_global = -math.inf

    #for th in tqdm(threshold_list, disable=not verbose):
    model = DistanceEnsembleCPDModel(
        ens_model=ens_model,
        #threshold=th,
        kernel=kernel,
        window_size=window_size,
        anchor_window_type=anchor_window_type,
        distance=distance,
        p=p,
    )
    
    # (
    #         best_th_f1,
    #         best_time_to_FA,
    #         best_delay,
    #         auc,
    #         best_conf_matrix,
    #         best_f1,
    #         best_cover,
    #         best_th_cover,
    #         max_cover,
    #     ),
    #     (max_th_f1_margins_dict, max_f1_margins_dict),
    #     delay_dict,
    #     fp_delay_dict,
    # )

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
            max_cover
        ),
        (max_th_f1_margins_dict, max_f1_margins_dict), 
        delay_dict, 
        fp_delay_dict
    ) = evaluation_pipeline(
        model=model,
        test_dataloader=output_dataloader,
        #threshold_list=[0.5],
        threshold_list=threshold_list, 
        device=device,
        model_type="fake_mmd",
        #verbose=False,
        verbose=verbose,
        margin_list=margin_list,
    )

        # if write_metrics_filename is not None:
        #     write_metrics_to_file(
        #         filename=write_metrics_filename,
        #         metrics=(metrics_local, (_, max_f1_margins_dict)),
        #         seed=None,
        #         timestamp=datetime.now().strftime("%y%m%dT%H%M%S"),
        #         comment=f"kernel_{kernel}_window_size_{window_size}_th_{th}",
        #     )

        # (
        #     _,
        #     best_time_to_FA,
        #     best_delay,
        #     audc,
        #     _,
        #     best_f1,
        #     best_cover,
        #     _,
        #     max_cover,
        # ) = metrics_local
        # res_dict[th] = (
        #     audc,
        #     best_time_to_FA,
        #     best_delay,
        #     best_f1,
        #     best_cover,
        #     max_cover,
        #     max_f1_margins_dict,
        # )

        # if best_f1 > best_f1_global:
        #     best_f1_global = best_f1
        #     best_th = th

    # # if verbose:
    # #     (
    # #         _audc,
    # #         _best_time_to_FA,
    # #         _best_delay,
    # #         _best_f1,
    # #         _best_cover,
    # #         _max_cover,
    # #         _max_f1_margins_dict,
    # #     ) = res_dict[best_th]

    # #     _audc = np.round(_audc, 4)
    # #     _best_time_to_FA = np.round(_best_time_to_FA, 4)
    # #     _best_delay = np.round(_best_delay, 4)
    # #     _best_f1 = np.round(_best_f1, 4)
    # #     _best_cover = np.round(_best_cover, 4)
    # #     _max_cover = np.round(_max_cover, 4)

    # #     print(f"Results for best threshold = {best_th}")
    # #     print(
    # #         f"AUDC: {_audc}, Time to FA: {_best_time_to_FA}, DD: {_best_delay}, F1: {_best_f1}, Cover: {_best_cover}, Max Cover: {_max_cover}"
    # #     )
    # #     for margin in margin_list:
    # #         print(
    # #             f"Max F1 with margin {margin}: {np.round(_max_f1_margins_dict[margin], 4)}"
    #         )
    
    #return res_dict, best_th
    
    # res_dict
    
    #return metrics_local, (max_th_f1_margins_dict, max_f1_margins_dict), delay_dict, fp_delay_dict
    
    return (
        (auc, best_time_to_FA, best_delay, best_f1, best_cover, max_cover, max_f1_margins_dict), 
        best_th_f1
    )


def all_distances_evaluation_pipeline(
    ens_model,
    test_dataloader,
    precomputed=False,
    distance="wasserstein",
    p=1,
    device="cpu",
    verbose=True,
    window_size_list=[1, 2, 3],
    margin_list=[1, 2, 4],
    anchor_window_type_list=["start", "prev"],
    threshold_list=np.linspace(0, 1, 50),
):
    if not precomputed:
        test_out_bank, _, test_labels_bank = collect_model_predictions_on_set(
            ens_model,
            test_dataloader,
            model_type="ensemble_all_models",
            device=device,
            verbose=verbose,
        )

        out_dataset = AllModelsOutputDataset(test_out_bank, test_labels_bank)

        test_dataloader = DataLoader(
            out_dataset, batch_size=128, shuffle=False
        )  # batch size does not matter

    res_dict = {}

    for window_size, anchor_window_type in itertools.product(
        window_size_list, anchor_window_type_list
    ):
        if verbose:
            print(
                f"window_size = {window_size}, anchor_window_type = {anchor_window_type}"
            )

        res, best_th = evaluate_distance_ensemble_model(
            ens_model=ens_model,
            threshold_list=threshold_list,
            output_dataloader=test_dataloader,
            margin_list=margin_list,
            window_size=window_size,
            anchor_window_type=anchor_window_type,
            distance=distance,
            p=p,
            device="cpu",
            verbose=verbose,
        )

        #res_dict[(window_size, anchor_window_type)] = (res[best_th], best_th)
        res_dict[(window_size, anchor_window_type)] = (res, best_th)

    return res_dict


def evaluate_all_models_in_ensemble(
    ens_model,
    test_dataloader,
    threshold_number,
    device="cpu",
    model_type="seq2seq",
    scale=None,
    step=1,
    alpha=1.0,
    margin_list=None,
    verbose=True,
):
    threshold_list = np.linspace(-5, 5, threshold_number)
    threshold_list = 1 / (1 + np.exp(-threshold_list))
    threshold_list = [-0.001] + list(threshold_list) + [1.001]

    time_fa_list = []
    delay_list = []
    audc_list = []
    f1_list = []
    cover_list = []
    max_cover_list = []
    f1_m1_list = []
    f1_m2_list = []
    f1_m3_list = []

    for model in tqdm(ens_model.models_list, disable=not verbose):
        metrics, (_, max_f1_margins_dic), _, _ = evaluation_pipeline(
            model,
            test_dataloader,
            threshold_list,
            device=device,
            model_type=model_type,
            verbose=verbose,
            margin_list=margin_list,
            scale=scale,
            step=step,
            alpha=alpha,
        )

        _, time_fa, delay, audc, _, f1, cover, _, max_cover = metrics
        f1_m1, f1_m2, f1_m3 = max_f1_margins_dic.values()

        time_fa_list.append(time_fa)
        delay_list.append(delay)
        audc_list.append(audc)
        f1_list.append(f1)
        cover_list.append(cover)
        max_cover_list.append(max_cover)
        f1_m1_list.append(f1_m1)
        f1_m2_list.append(f1_m2)
        f1_m3_list.append(f1_m3)

    if verbose:
        print(f"AUC: {round(np.mean(audc_list), 4)} \pm {round(np.std(audc_list), 4)}")
        print(
            f"Time to FA: {round(np.mean(time_fa_list), 4)} \pm {round(np.std(time_fa_list), 4)}"
        )
        print(
            f"Delay detection: {round(np.mean(delay_list), 4)} \pm {round(np.std(delay_list), 4)}"
        )
        print(f"Max F1: {round(np.mean(f1_list), 4)} \pm {round(np.std(f1_list), 4)}")
        print(
            f"Cover: {round(np.mean(cover_list), 4)} \pm {round(np.std(cover_list), 4)}"
        )
        print(
            f"Max cover: {round(np.mean(max_cover_list), 4)} \pm {round(np.std(max_cover_list), 4)}"
        )
        print(
            f"Max F1 with m1: {round(np.mean(f1_m1_list), 4)} \pm {round(np.std(f1_m1_list), 4)}"
        )
        print(
            f"Max F1 with m2: {round(np.mean(f1_m2_list), 4)} \pm {round(np.std(f1_m2_list), 4)}"
        )
        print(
            f"Max F1 with m3: {round(np.mean(f1_m3_list), 4)} \pm {round(np.std(f1_m3_list), 4)}"
        )

    return (
        time_fa_list,
        delay_list,
        audc_list,
        f1_list,
        cover_list,
        max_cover_list,
        f1_m1_list,
        f1_m2_list,
        f1_m3_list,
    )
