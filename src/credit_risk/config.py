"""Central configuration: paths, schema, business constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "german_credit.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
MODEL_PATH = MODEL_DIR / "credit_model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
METRICS_PATH = REPORT_DIR / "metrics.json"
FAIRNESS_PATH = REPORT_DIR / "fairness.json"
DRIFT_PATH = REPORT_DIR / "drift_report.json"
REFERENCE_PATH = MODEL_DIR / "reference_profile.json"
REGISTRY_DIR = MODEL_DIR / "registry"
LOG_DIR = ROOT / "logs"
PRED_LOG = LOG_DIR / "predictions.jsonl"
MLRUNS = ROOT / "mlruns"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_REPEATS = 3
BOOTSTRAPS = 1000

# Source column "credit_risk": 1 = good, 0 = bad. We model P(default) so bad = 1.
SOURCE_TARGET = "credit_risk"
TARGET = "default"

NUMERIC = ["duration", "amount", "installment_rate", "present_residence",
           "age", "number_credits", "people_liable"]
# Legally sensitive attributes: kept in the data for FAIRNESS AUDITING only, never fed to the model.
PROTECTED = ["personal_status_sex", "foreign_worker"]
CATEGORICAL = ["status", "credit_history", "purpose", "savings", "employment_duration",
               "other_debtors", "property", "other_installment_plans", "housing", "job",
               "telephone"]
ENGINEERED = ["monthly_burden", "log_amount", "burden_to_age", "is_young", "long_loan"]

# Original dataset cost matrix: approving a bad applicant costs 5x rejecting a good one.
COST_FN = 5.0   # approve someone who defaults
COST_FP = 1.0   # reject someone who would have repaid

# Scorecard scaling (industry style): 600 points = 30:1 good:bad odds, 20 points to double odds.
BASE_SCORE, BASE_ODDS, PDO = 600, 30, 20

# Monitoring thresholds (Population Stability Index): <0.1 stable, 0.1-0.25 watch, >0.25 drift
PSI_WARN, PSI_ALERT = 0.10, 0.25
MIN_MONITOR_ROWS = 30
