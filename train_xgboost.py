import argparse
from pathlib import Path

from xgboost_function import CopyrightXGBoost


def main(args: argparse.Namespace):
    xgboost = CopyrightXGBoost()
    xgboost.train_xgboost(Path(args.data), Path(args.out), args.train)


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
        default="/out",
        help="name of the folder that output will be saved to",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.8,
        help="percentage of data used for training (as opposed to evaluation)",
    )
    args = parser.parse_args()
    main(args)
