import torch
from model import *
import pandas as pd
from tokenizer import y_numerics
train_set = "data/train.csv"

test_set = "data/test.csv"
if __name__ == "__main__":
    df = pd.read_csv(train_set)
    # df = df[df[y_numerics] < df[y_numerics].quantile(0.95)]
    y = df[y_numerics]
    m = Model()
    m.build(df)
    best_loss = m.train(df, y, 200)
    
    test_df = pd.read_csv(test_set)
    print(f"best loss: {best_loss}")
    best = torch.load(f"saved_models/dense_{best_loss}.pickle")
    y_pred = best(test_df)
    test_df[y_numerics] = y_pred
    answer = test_df[["Id", y_numerics]]
    answer.to_csv("submission.csv", index=False)
    