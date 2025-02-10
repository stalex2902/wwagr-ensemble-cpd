"""Default script for a complete ensemble evaluation."""

import argparse
import warnings

from scripts.train_single_models import train_single_models

warnings.filterwarnings("ignore")

def get_parser():
    """Parse command line arguments for main.py"""

    parser = argparse.ArgumentParser(description="Evaluate ensemble")
    parser.add_argument(
        "--experiments_name",
        type=str,
        required=True,
        help="name of sdataset",
        choices=[
            "human_activity",
            "explosion",
            "road_accidents",
            "yahoo",
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
        "--ens_num_list",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="Number of ensembles to be trained",
    )

    parser.add_argument(
        "--n_models", type=int, default=10, help="Number of models in ensemble"
    )

    # boolean
    parser.add_argument(
        "--verbose",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="If true, print the metrics to the console.",
    )
    
    parser.add_argument(
        "--evaluate",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="If true, print the metrics to the console.",
    )
        
    parser.add_argument(
        "--save_models",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="If true, save models' weights",
    )
    
    parser.add_argument(
        "--boot_sample_size",
        type=int,
        default=None,
        help="If not None, bootrap train subsample of this size",
    )
    return parser


def main(args) -> None:
    (
        experiments_name,
        model_type,
        loss_type,
        ens_num_list,
        n_models,
        verbose,
        evaluate,
        save_models,
        boot_sample_size,
    ) = args.values()
    _ = train_single_models(
        experiments_name=experiments_name,
        model_type=model_type,
        loss_type=loss_type,
        ens_num_list=ens_num_list,
        n_models=n_models,
        verbose=verbose,
        evaluate=evaluate,
        save_models=save_models,
        boot_sample_size=boot_sample_size,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = dict(vars(parser.parse_args()))

    main(args)
