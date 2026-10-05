"""Credit Risk Scoring API.  Run: uvicorn api.main:app --reload

Production features: API-key auth (env API_KEY), request IDs, structured JSON logs,
Prometheus-style /metrics, prediction logging for drift monitoring, model versioning.
"""
import json
import logging
import os
import secrets
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import APIKeyHeader

from credit_risk import __version__, monitor
from credit_risk.predict import CreditScorer, ModelNotTrained

from .schemas import Application, BatchRequest, ScoreResponse


class JsonFormatter(logging.Formatter):
    def format(self, r):
        d = {"ts": self.formatTime(r), "level": r.levelname, "msg": r.getMessage()}
        d.update(getattr(r, "extra_fields", {}))
        return json.dumps(d)


_h = logging.StreamHandler()
_h.setFormatter(JsonFormatter())
log = logging.getLogger("api")
log.handlers, log.level = [_h], logging.INFO

state: dict = {}
metrics = {"requests": Counter(), "decisions": Counter(), "latency_sum": 0.0, "latency_n": 0}
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def auth(key: str | None = Security(api_key_header)):
    expected = os.getenv("API_KEY")
    if expected and not (key and secrets.compare_digest(key, expected)):
        raise HTTPException(401, "Invalid or missing X-API-Key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["scorer"] = CreditScorer()
        log.info("model loaded", extra={"extra_fields": state["scorer"].info()})
    except ModelNotTrained as e:
        log.error(str(e))
    yield
    state.clear()


app = FastAPI(title="Credit Risk Scoring API", version=__version__, lifespan=lifespan,
              description="Calibrated probability of default, 300-850 score, decision and SHAP reason codes.")


def scorer() -> CreditScorer:
    if "scorer" not in state:
        raise HTTPException(503, "Model not loaded. Train it first: python -m credit_risk.train")
    return state["scorer"]


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    request.state.rid = rid
    t = time.perf_counter()
    resp = await call_next(request)
    ms = (time.perf_counter() - t) * 1000
    resp.headers["X-Request-ID"], resp.headers["X-Process-Time-ms"] = rid, f"{ms:.1f}"
    metrics["requests"][(request.url.path, resp.status_code)] += 1
    metrics["latency_sum"] += ms
    metrics["latency_n"] += 1
    log.info("request", extra={"extra_fields": {"rid": rid, "path": request.url.path,
                                                "status": resp.status_code, "ms": round(ms, 1)}})
    return resp


@app.exception_handler(ValueError)
async def value_error(_, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health")
def health():
    ok = "scorer" in state
    return {"status": "ok" if ok else "model_not_loaded", "api_version": __version__,
            "model": state["scorer"].info() if ok else None}


@app.get("/schema", dependencies=[Depends(auth)])
def schema():
    s = scorer()
    return {"categories": s.categories, "numeric_ranges": s.numeric_ranges, "threshold": s.threshold}


@app.post("/score", response_model=ScoreResponse, dependencies=[Depends(auth)])
def score(app_in: Application, request: Request, bg: BackgroundTasks):
    rec = app_in.model_dump()
    res = scorer().score(rec)
    metrics["decisions"][res["decision"]] += 1
    bg.add_task(monitor.log_prediction, rec, res, request.state.rid)
    return res


@app.post("/score/batch", response_model=list[ScoreResponse], dependencies=[Depends(auth)])
def score_batch(batch: BatchRequest, request: Request, bg: BackgroundTasks):
    out = []
    for a in batch.applications:
        rec = a.model_dump()
        res = scorer().score(rec)
        metrics["decisions"][res["decision"]] += 1
        bg.add_task(monitor.log_prediction, rec, res, request.state.rid)
        out.append(res)
    return out


@app.get("/monitoring/drift", dependencies=[Depends(auth)])
def drift():
    scorer()
    return monitor.drift_report()


@app.get("/metrics", response_class=PlainTextResponse)
def prom():
    lines = ["# TYPE credit_api_requests_total counter"]
    lines += [f'credit_api_requests_total{{path="{p}",status="{s}"}} {n}'
              for (p, s), n in metrics["requests"].items()]
    lines += ["# TYPE credit_api_decisions_total counter"]
    lines += [f'credit_api_decisions_total{{decision="{d}"}} {n}' for d, n in metrics["decisions"].items()]
    avg = metrics["latency_sum"] / metrics["latency_n"] if metrics["latency_n"] else 0
    lines += ["# TYPE credit_api_latency_ms_avg gauge", f"credit_api_latency_ms_avg {avg:.2f}"]
    return "\n".join(lines) + "\n"
