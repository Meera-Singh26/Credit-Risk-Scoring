"""Experiment tracking. MLflow when available, silent no-op otherwise."""
import contextlib
import logging

from . import config as C

log = logging.getLogger("tracking")
try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


@contextlib.contextmanager
def run(name: str):
    if mlflow is None:
        yield None
        return
    C.MLRUNS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{C.MLRUNS / 'mlflow.db'}")
    mlflow.set_experiment("credit-risk-scoring")
    with mlflow.start_run(run_name=name) as r:
        yield r


def log_params(d: dict):
    if mlflow and mlflow.active_run():
        mlflow.log_params({k: str(v)[:250] for k, v in d.items()})


def log_metrics(d: dict):
    if mlflow and mlflow.active_run():
        mlflow.log_metrics({k: float(v) for k, v in d.items() if isinstance(v, (int, float))})


def log_artifacts(path):
    if mlflow and mlflow.active_run():
        mlflow.log_artifacts(str(path))
