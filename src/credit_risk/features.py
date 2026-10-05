"""Feature engineering + preprocessing pipeline (fit inside CV, no leakage)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless domain features derived from loan amount, tenure and age."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["monthly_burden"] = X["amount"] / X["duration"].clip(lower=1)
        X["log_amount"] = np.log1p(X["amount"])
        X["burden_to_age"] = X["monthly_burden"] / X["age"].clip(lower=18)
        X["is_young"] = (X["age"] < 25).astype(int)
        X["long_loan"] = (X["duration"] > 36).astype(int)
        return X


class SafeNames(BaseEstimator, TransformerMixin):
    """XGBoost rejects '[', ']' and '<' in feature names; make them readable words instead."""

    @staticmethod
    def clean(n: str) -> str:
        for a, b in ((">=", "at least"), ("<=", "up to"), ("<", "below"), (">", "above"),
                     ("[", "("), ("]", ")")):
            n = n.replace(a, b)
        return n

    def fit(self, X, y=None):
        self.is_fitted_ = True   # lets sklearn's check_is_fitted pass on the enclosing Pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X.columns = [self.clean(c) for c in X.columns]
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array([self.clean(c) for c in input_features])


def build_preprocessor(scale_numeric: bool = False) -> Pipeline:
    num = C.NUMERIC + C.ENGINEERED
    num_tf = StandardScaler() if scale_numeric else "passthrough"
    ct = ColumnTransformer(
        [("num", num_tf, num),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), C.CATEGORICAL)],
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return Pipeline([("fe", FeatureEngineer()), ("ct", ct), ("safe", SafeNames())])
