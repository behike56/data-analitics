import pandas as pd
import numpy as np
import random


def read_3_data() -> list[pd.DataFrame]:
    train_df: pd.DataFrame = pd.read_csv("./data/train.csv")
    test_df: pd.DataFrame = pd.read_csv("./data/test.csv")
    submission: pd.DataFrame = pd.read_csv("./data/gender_submission.csv")
    return train_df, test_df, submission


def main():
    train_df, test_df, submission = read_3_data()
    print(train_df.head())
    print(test_df.head())
    print(submission.head())
