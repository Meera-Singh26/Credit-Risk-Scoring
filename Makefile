export PYTHONPATH := src:.
install:; pip install -r requirements-dev.txt
eda:; python -m credit_risk.eda
train:; python -m credit_risk.train
lint:; ruff check .
test:; python -m pytest -q
api:; uvicorn api.main:app --reload
app:; streamlit run app/streamlit_app.py
mlflow:; mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
simulate:; python scripts/simulate_traffic.py --reset
simulate-drift:; python scripts/simulate_traffic.py --reset --drift
all: lint eda train test
