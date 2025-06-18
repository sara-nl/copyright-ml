import argparse
from pathlib import Path

from xgboost_function import CopyrightXGBoost


def main(args: argparse.Namespace):
    xgboost = CopyrightXGBoost(args.random_state)
    xgboost.train_xgboost(Path(args.data), Path(args.out), args.train, args.use_class_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/traindata.csv",
        help="input csv file that contains the data",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out",
        help="name of the folder that output will be saved to",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="percentage of data used for training (as opposed to evaluation)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="random state for reproducible results",
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="apply class weights to handle class imbalance",
    )
    
    args = parser.parse_args()
    
    # create output directory if it does not exist
    Path(args.out).mkdir(parents=True, exist_ok=True)

    main(args)
