import pytest
from fastapi.testclient import TestClient

from credit_risk import config as C

pytestmark = pytest.mark.skipif(not C.MODEL_PATH.exists(), reason="train the model first")
from api.main import app  # noqa: E402

GOOD = {"duration": 12, "amount": 1500, "installment_rate": 2, "present_residence": 3, "age": 45,
        "number_credits": 1, "people_liable": 1, "status": "no checking account",
        "credit_history": "existing credits paid back duly till now", "purpose": "radio/television",
        "savings": "... >= 1000 DM", "employment_duration": "... >= 7 years",
        "personal_status_sex": "male : single", "other_debtors": "none", "property": "real estate",
        "other_installment_plans": "none", "housing": "own", "job": "skilled employee/official",
        "telephone": "yes", "foreign_worker": "no"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_score_shape(client):
    r = client.post("/score", json=GOOD)
    assert r.status_code == 200
    b = r.json()
    assert 0 <= b["default_probability"] <= 1 and 300 <= b["credit_score"] <= 850
    assert b["decision"] in {"APPROVE", "REJECT"} and len(b["reasons"]) == 5


def test_risky_profile_scores_lower(client):
    risky = {**GOOD, "duration": 60, "amount": 15000, "age": 21, "status": "... < 100 DM",
             "savings": "... < 100 DM", "employment_duration": "unemployed"}
    assert (client.post("/score", json=risky).json()["default_probability"]
            > client.post("/score", json=GOOD).json()["default_probability"])


def test_validation_errors(client):
    assert client.post("/score", json={**GOOD, "age": 5}).status_code == 422
    assert client.post("/score", json={**GOOD, "housing": "castle"}).status_code == 422
    assert client.post("/score", json={"age": 30}).status_code == 422


def test_batch(client):
    r = client.post("/score/batch", json={"applications": [GOOD, GOOD]})
    assert r.status_code == 200 and len(r.json()) == 2


def test_auth_enforced_when_key_set(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    assert client.post("/score", json=GOOD).status_code == 401
    assert client.post("/score", json=GOOD, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/score", json=GOOD, headers={"X-API-Key": "secret123"}).status_code == 200
    assert client.get("/health").status_code == 200   # health stays open for probes


def test_request_id_and_metrics(client):
    r = client.post("/score", json=GOOD, headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123" and r.json()["model_version"]
    m = client.get("/metrics").text
    assert "credit_api_requests_total" in m and "credit_api_decisions_total" in m


def test_predictions_are_logged(client, monkeypatch, tmp_path):
    monkeypatch.setattr(C, "LOG_DIR", tmp_path)
    monkeypatch.setattr(C, "PRED_LOG", tmp_path / "p.jsonl")
    client.post("/score", json=GOOD)
    lines = (tmp_path / "p.jsonl").read_text().splitlines()
    assert len(lines) == 1 and "credit_score" in lines[0]


def test_protected_fields_optional_and_ignored(client):
    slim = {k: v for k, v in GOOD.items() if k not in ("personal_status_sex", "foreign_worker")}
    a = client.post("/score", json=slim).json()["default_probability"]
    b = client.post("/score", json={**GOOD, "personal_status_sex": "female : divorced/separated/married",
                                    "foreign_worker": "yes"}).json()["default_probability"]
    assert a == b   # protected attributes never influence the score
