"""Data loading, validation and splitting."""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def load_raw(path=C.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    expected = set(C.NUMERIC + C.CATEGORICAL + C.PROTECTED + [C.SOURCE_TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not df[C.SOURCE_TARGET].isin([0, 1]).all():
        raise ValueError("Target must be binary 0/1")
    if df[C.NUMERIC].isna().any().any():
        raise ValueError("Unexpected nulls in numeric columns")


def make_xy(df: pd.DataFrame):
    X = df[C.NUMERIC + C.CATEGORICAL + C.PROTECTED].copy()  # protected cols ride along for audits
    y = (1 - df[C.SOURCE_TARGET]).astype(int).rename(C.TARGET)  # 1 = default
    return X, y


def split(X, y):
    return train_test_split(X, y, test_size=C.TEST_SIZE, stratify=y,
                            random_state=C.RANDOM_STATE)


def data_fingerprint(path=C.DATA_PATH) -> str:
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
