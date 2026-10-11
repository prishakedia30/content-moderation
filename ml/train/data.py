"""Jigsaw Toxic Comment data -> one binary label + a fixed 80/10/10 split.
The split is deterministic (sorted ids, seed 42), so training on Kaggle and evaluating
on the server see exactly the same test set."""
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 42


def load(path):
    """path: Jigsaw train.csv (or train.csv.zip as Kaggle ships it)."""
    df = pd.read_csv(path)
    df["label"] = (df[LABELS].sum(axis=1) > 0).astype(int)   # toxic = any of the six labels
    df = df.rename(columns={"comment_text": "text"})[["id", "text", "label"]]
    return df.sort_values("id").reset_index(drop=True)


def split(df):
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def describe(train, val, test):
    for name, d in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:5s} {len(d):7d} rows, {d['label'].mean():.1%} toxic")
