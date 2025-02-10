"""Default script for a threshold number range evaluation."""

import argparse
import warnings

from scripts.evaluate_distance_thresholds_range import evaluate_distance_thresholds_range_all_ensembles

warnings.filterwarnings("ignore")


def get_parser():
    """Parse command line arguments for run.py"""

    parser = argparse.ArgumentParser(description="Evaluate ensemble")
    parser.add_argument(
        "--experiments_name",
        type=str,
        required=True,
        help="name of sdataset",
        choices=[
            "human_activity",
            "yahoo",
            "explosion",
            "road_accidents",
        ],
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Model type",
        choices=["seq2seq", "tscp", "ts2vec"],
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        help="Loss type for seq2seq model",
        choices=["indid", "bce"],
    )

    parser.add_argument(
        "--n_models", type=int, default=10, help="Number of models in ensemble"
    )
    parser.add_argument(
        "-cal",
        "--calibrate",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="If True, calibrate the models using Beta calibration",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="wasserstein_1d",
        help="Distance type",
        choices=["wasserstein_1d", "wasserstein_nd", "mmd"],
    )
    parser.add_argument(
        "-tn_list",
        "--threshold_number_list",
        nargs="*",
        type=int,
        default=[1, 3, 5, 7, 9, 15, 20, 25, 30, 40, 50, 300],
        help="threshold number list",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # boolean
    parser.add_argument(
        "--verbose",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="If true, print the metrics to the console.",
    )
    parser.add_argument(
        "--save_df",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="If true, save the dataframe with results.",
    )

    return parser


def main(args) -> None:
    (
        experiments_name,
        model_type,
        loss_type,
        n_models,
        calibrate,
        distance,
        threshold_number_list,
        seed,
        verbose,
        save_df,
    ) = args.values()
    _ = evaluate_distance_thresholds_range_all_ensembles(
        experiments_name=experiments_name,
        model_type=model_type,
        loss_type=loss_type,
        n_models=n_models,
        calibrate=calibrate,
        distance=distance,
        threshold_number_list=threshold_number_list,
        seed=seed,
        verbose=verbose,
        save_df=save_df,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = dict(vars(parser.parse_args()))

    main(args)
