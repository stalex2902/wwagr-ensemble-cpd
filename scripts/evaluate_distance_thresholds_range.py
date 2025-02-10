from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from src.datasets.datasets import AllModelsOutputDataset, CPDDatasets
from src.ensembles.distances import MMD_batch
from src.ensembles.ensembles import EnsembleCPDModel
from src.metrics.evaluation_pipelines import all_distances_evaluation_pipeline
from src.metrics.metrics_utils import collect_model_predictions_on_set
from src.utils.calibration import calibrate_all_models_in_ensemble
from src.utils.fix_seeds import fix_seeds
from torch.utils.data import DataLoader


def evaluate_distance_thresholds_range(
    experiments_name: str,
    model_type: str,
    loss_type: Optional[str] = None,
    n_models: int = 10,
    calibrate: bool = False,
    ensemble_num: int = 1,
    distance: str = "wasserstein_1d",
    threshold_number_list: List[int] = [1, 3, 5, 7, 9, 15, 20, 25, 30, 40, 50],
    seed: int = 42,
    verbose: bool = True,
    save_df: bool = False,
):
    if verbose:
        print("Loading datasets and models")

    if experiments_name in ["human_activity", "yahoo"]:
        path_to_config = "configs/" + experiments_name + "_" + model_type + ".yaml"
        device = "cpu"
    elif experiments_name in ["explosion", "road_accidents"]:
        assert model_type == "seq2seq", "Only seq2seq models are used for video data"
        path_to_config = "configs/" + "video" + "_" + model_type + "_" + loss_type + ".yaml"
        device = "cuda:0"
    else:
        raise ValueError(f"Unknown experiments name {experiments_name}")

    with open(path_to_config, "r") as f:
        args_config = yaml.safe_load(f.read())

    args_config["experiments_name"] = experiments_name
    args_config["model_type"] = model_type
    args_config["num_workers"] = 2

    if model_type == "seq2seq":
        if loss_type == "bce":
            args_config["loss_type"] = "bce"
        elif loss_type == "indid":
            args_config["loss_type"] = "indid"
        else:
            raise ValueError(f"Wrong loss type {loss_type}")

    if model_type == "seq2seq":
        path_to_models_folder = f"saved_models/{loss_type}/{experiments_name}/ens_{ensemble_num}"
    
    elif model_type in ["tscp", "ts2vec"]:
        path_to_models_folder = f"saved_models/{model_type}/{experiments_name}/ens_{ensemble_num}" 

    else:
        raise ValueError(f"Wrong model type {model_type}")

    fix_seeds(seed)

    train_dataset, test_dataset = CPDDatasets(experiments_name).get_dataset_()

    test_dataloader = DataLoader(
        test_dataset, batch_size=args_config["learning"]["batch_size"], shuffle=False
    )

    ens_model = EnsembleCPDModel(args_config, n_models=n_models)
    ens_model.load_models_list(path_to_models_folder)

    if calibrate:
        if verbose:
            print("Calibrating the models using Beta calibration")

        _, val_dataset = train_test_split(
            train_dataset, test_size=0.3, random_state=seed
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args_config["learning"]["batch_size"], shuffle=False
        )

        _ = calibrate_all_models_in_ensemble(
            ens_model,
            val_dataloader,
            cal_type="beta",
            verbose=verbose,
            device=device,
        )

    columns = [
        "Model name",
        "AUDC",
        "Time to FA",
        "DD",
        "Max F1",
        "Cover",
        "Max Cover",
        "Max F1, m1",
        "Max F1, m2",
        "Max F1, m3",
        "params",
    ]
    results_df = pd.DataFrame(columns=columns)

    if verbose:
        print("Collecting prediction:")

    test_out_bank, _, test_labels_bank = collect_model_predictions_on_set(
        ens_model,
        test_dataloader,
        model_type="ensemble_all_models",
        device=device,
        verbose=verbose,
    )

    out_dataset = AllModelsOutputDataset(test_out_bank, test_labels_bank)

    test_dataloader = DataLoader(out_dataset, batch_size=128, shuffle=False)

    if verbose:
        print("Evaluating dustance-based approach:")

    for threshold_number in threshold_number_list:
        if distance.startswith("wasserstein"):
            if threshold_number == 1:
                threshold_list_dist = [0.5]
            elif threshold_number == 3:
                threshold_list_dist = [0.25, 0.50, 0.75]
            elif threshold_number == 5:
                threshold_list_dist = [0.17, 0.33, 0.50, 0.67, 0.84]
            elif threshold_number == 7:
                threshold_list_dist = [0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875]
            elif threshold_number >= 9:
                threshold_list_dist = np.linspace(-0.01, 1.01, threshold_number + 2)[
                    1:-1
                ]  # avoid 0.0 and 1.0 thresholds
            else:
                raise ValueError(f"Unsupported th number {threshold_number}")

        elif distance == "mmd":
            outs_batch, _ = next(iter(test_dataloader))
            outs_batch = outs_batch.permute(0, 2, 1)  # (bs, seq_len, ens_size)
            ws = 2
            history_sample, future_sample = (
                outs_batch[:, :ws, :],
                outs_batch[:, -ws:, :],
            )
            mmd_sample = MMD_batch(history_sample, future_sample)
            min_value = torch.quantile(mmd_sample, 0.05).item()
            max_value = torch.quantile(mmd_sample, 0.95).item()

            threshold_list_dist = np.linspace(
                min_value, max_value, threshold_number + 2
            )[1:-1]

        res_dist = all_distances_evaluation_pipeline(
            ens_model,
            test_dataloader,
            precomputed=True,
            distance=distance,
            p=2, # !!!!!!!!
            device=device,
            verbose=verbose,
            window_size_list=args_config["distance"]["window_size_list"],
            margin_list=args_config["evaluation"]["margin_list"],
            anchor_window_type_list=args_config["distance"]["anchor_window_type_list"],
            threshold_list=threshold_list_dist,
        )

        # extract metrics for distances
        best_f1_start, best_f1_prev = -np.inf, -np.inf
        best_res_start, best_res_prev = None, None
        best_ws_start, best_ws_prev = None, None
        best_th_start, best_th_prev = None, None

        for (ws, anchor_type), (res, best_th) in res_dist.items():
            f1 = res[3]

            if anchor_type == "start":
                if f1 > best_f1_start:
                    best_f1_start = f1
                    best_res_start = res
                    best_ws_start = ws
                    best_th_start = best_th

            if anchor_type == "prev":
                if f1 > best_f1_prev:
                    best_f1_prev = f1
                    best_res_prev = res
                    best_ws_prev = ws
                    best_th_prev = best_th

        (
            auc_start,
            time_to_FA_start,
            delay_start,
            f1_start,
            cover_start,
            max_cover_start,
            f1_margins_start,
        ) = best_res_start
        f1_m1_start, f1_m2_start, f1_m3_start = f1_margins_start.values()

        (
            auc_prev,
            time_to_FA_prev,
            delay_prev,
            f1_prev,
            cover_prev,
            max_cover_prev,
            f1_margins_prev,
        ) = best_res_prev
        f1_m1_prev, f1_m2_prev, f1_m3_prev = f1_margins_prev.values()

        results_df = results_df.append(
            {
                "Model name": f"Ens num {ensemble_num}, cal = {calibrate}, {distance} start, th number {threshold_number}",
                "AUDC": auc_start,
                "Time to FA": time_to_FA_start,
                "DD": delay_start,
                "Max F1": f1_start,
                "Cover": cover_start,
                "Max Cover": max_cover_start,
                "Max F1, m1": f1_m1_start,
                "Max F1, m2": f1_m2_start,
                "Max F1, m3": f1_m3_start,
                "params": f"ws = {best_ws_start}, th = {best_th_start}",
            },
            ignore_index=True,
        )

        results_df = results_df.append(
            {
                "Model name": f"Ens num {ensemble_num}, cal = {calibrate}, {distance} prev, th num {threshold_number}",
                "AUDC": auc_prev,
                "Time to FA": time_to_FA_prev,
                "DD": delay_prev,
                "Max F1": f1_prev,
                "Cover": cover_prev,
                "Max Cover": max_cover_prev,
                "Max F1, m1": f1_m1_prev,
                "Max F1, m2": f1_m2_prev,
                "Max F1, m3": f1_m3_prev,
                "params": f"ws = {best_ws_prev}, th = {best_th_prev}",
            },
            ignore_index=True,
        )

    if save_df:
        if verbose:
            print("Saving the results")

        if model_type == "seq2seq":
            model_name = model_type + "_" + loss_type
        else:
            model_name = model_type

        save_path = f"results/{experiments_name}/{model_name}_calibrated_{calibrate}_ens_num_{ensemble_num}_{experiments_name}_{distance}_th_num_range.csv"
        results_df.to_csv(save_path)

    return results_df


def evaluate_distance_thresholds_range_all_ensembles(
    experiments_name: str,
    model_type: str,
    loss_type: str = None,
    n_models: int = 10,
    calibrate: bool = False,
    distance: str = "wasserstein_1d",
    threshold_number_list: List[int] = [1, 3, 5, 7, 9, 15, 20, 25, 30, 40, 50],
    seed: int = 42,
    verbose: bool = True,
    save_df: bool = False,
):
    res_df_list = []
    for ensemble_num in [1, 2, 3]:
        curr_res_df = evaluate_distance_thresholds_range(
            experiments_name,
            model_type,
            loss_type,
            n_models,
            calibrate,
            ensemble_num,
            distance,
            threshold_number_list,
            seed,
            verbose,
        )
        res_df_list.append(curr_res_df)

    results_df = pd.concat(res_df_list)

    if save_df:
        if verbose:
            print("Saving the results")

        if model_type == "seq2seq":
            model_name = model_type + "_" + loss_type
        else:
            model_name = model_type

        save_path = f"results/{experiments_name}/{model_name}_calibrated_{calibrate}_all_ensembles_{experiments_name}_{distance}_p2_th_range.csv"
        results_df.to_csv(save_path)
