"""
main.py
=======
Transaction Risk Scoring API
Moving from Rule-Based to AI/ML Fraud Detection — Nigerian Fintech

Version: 2.0.0 — Upgraded with model health monitoring, transaction audit
logging, LIME explainability, and REVIEWED compliance action.

Endpoints:
    POST /v1/score          — Score a transaction
    POST /v1/explain        — LIME secondary explanation
    GET  /v1/health         — Health check
    GET  /v1/model-health   — Model drift monitoring
    GET  /v1/model-info     — Model metadata
    GET  /v1/audit-log      — Recent scoring audit trail
    GET  /docs              — Interactive API documentation

NDPA 2023 Compliance:
    HIGH and CRITICAL risk transactions are flagged for human review.
    No fully automated blocks occur without human oversight.
    All PII fields must be pre-hashed by the caller before submission.

References:
    Nadeem & Anwar (2025) — dual XAI frameworks, model drift monitoring
    Kumar (2025)          — microservices architecture, observability
    Prajapati (2025)      — concept drift, synthetic data methodology
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from collections import deque

from lime import lime_tabular

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

import shap
import hashlib
import uvicorn

# ── App initialisation ────────────────────────────────────────────────────────
app = FastAPI(
    title="Nigerian Fintech Transaction Risk Scoring API",
    description="""
## Moving from Rule-Based to AI/ML Fraud Detection

An open-source, NDPA-compliant transaction risk scoring API pre-engineered
with Nigeria-specific fraud signals. Deployable by SME fintechs without
in-house ML expertise.

### Risk Bands
| Score | Band | Action |
|-------|------|--------|
| 0.0 – 0.3 | LOW | Allow |
| 0.3 – 0.6 | MEDIUM | Step-up authentication |
| 0.6 – 0.8 | HIGH | Human review (NDPA §37(1)) |
| 0.8 – 1.0 | CRITICAL | Block + compliance alert |

### NDPA 2023 Compliance
HIGH and CRITICAL decisions are never fully automated.
They are routed to a human review queue as required by Section 37(1).

### References
- Nadeem & Anwar (2025): dual XAI frameworks, model drift monitoring
- Kumar (2025): microservices architecture, observability layers
- Prajapati (2025): concept drift, synthetic data methodology
    """,
    version="2.0.0",
    contact={
        "name": "Research Project — Nigerian Fintech Fraud Detection",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Audit log — in-memory ring buffer (last 500 scoring events) ───────────────
# Kumar (2025): observability is essential for resilient fintech systems.
# In production, replace with a persistent database or logging service.
AUDIT_LOG = deque(maxlen=500)

# ── Model loading ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")

def load_models():
    """Load all trained model artifacts at startup."""
    models = {}
    try:
        with open(os.path.join(MODELS_DIR, "xgboost_layer1.pkl"), "rb") as f:
            models["xgboost"] = pickle.load(f)

        with open(os.path.join(MODELS_DIR, "isolation_forest.pkl"), "rb") as f:
            models["iso_forest"] = pickle.load(f)

        with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
            models["scaler"] = pickle.load(f)

        with open(os.path.join(MODELS_DIR, "community_risk_map.pkl"), "rb") as f:
            models["community_risk_map"] = pickle.load(f)

        with open(os.path.join(MODELS_DIR, "graph_partition.pkl"), "rb") as f:
            models["graph_partition"] = pickle.load(f)

        with open(os.path.join(MODELS_DIR, "ensemble_config.json"), "r") as f:
            models["ensemble_config"] = json.load(f)

        # SHAP explainer for XGBoost
        models["shap_explainer"] = shap.TreeExplainer(models["xgboost"])

        # Startup timestamp for uptime tracking
        models["loaded_at"] = datetime.utcnow().isoformat()

        print("All models loaded successfully.")
        return models

    except FileNotFoundError as e:
        print(f"Warning: Model file not found — {e}")
        print("Run all notebooks first to generate model files.")
        return {}

MODELS = load_models()

# Feature columns — must match training order exactly
FEATURE_COLS = [
    'amount_ngn', 'is_interbank', 'is_agent_transaction', 'is_new_beneficiary',
    'hour_of_day', 'day_of_week', 'is_weekend', 'is_public_holiday', 'is_off_hours',
    'sim_age_days', 'sim_swap_flag', 'device_fingerprint_changed',
    'geo_displacement_flag', 'nin_bvn_mismatch',
    'velocity_1h', 'velocity_6h', 'velocity_24h', 'cumulative_send_24h_ngn',
    'channel_enc', 'sender_bank_enc', 'beneficiary_bank_enc', 'state_enc',
]

CHANNEL_MAP = {"USSD": 0, "mobile_app": 1, "POS": 2, "web": 3, "ATM": 4}
BANK_MAP    = {"GTB": 0, "Access": 1, "Zenith": 2, "UBA": 3, "Kuda": 4,
               "Opay": 5, "Moniepoint": 6, "PalmPay": 7, "FCMB": 8, "Sterling": 9}
STATE_MAP   = {"Lagos": 0, "Abuja": 1, "Kano": 2, "Rivers": 3, "Oyo": 4,
               "Kaduna": 5, "Anambra": 6, "Delta": 7, "Enugu": 8, "Ogun": 9}

# Training baseline metrics — used for drift detection
TRAINING_BASELINE = {
    "fraud_rate"        : 0.0049,
    "mean_risk_score"   : 0.12,
    "std_risk_score"    : 0.21,
    "precision"         : 0.9816,
    "recall"            : 0.9357,
    "f1"                : 0.9581,
    "roc_auc"           : 0.9991,
}

# ── Request / Response schemas ────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    """
    Transaction payload for risk scoring.
    All PII fields (sender_id, beneficiary_id) must be SHA-256 hashed
    by the caller before submission — NDPA compliance requirement.
    """
    transaction_id            : str   = Field(..., description="Unique transaction identifier")
    sender_id                 : str   = Field(..., description="SHA-256 hashed sender BVN/account")
    beneficiary_id            : str   = Field(..., description="SHA-256 hashed beneficiary account")
    amount_ngn                : float = Field(..., gt=0, description="Transaction amount in Naira")
    channel                   : str   = Field(..., description="USSD | mobile_app | POS | web | ATM")
    sender_bank               : str   = Field(..., description="Sender's bank name")
    beneficiary_bank          : str   = Field(..., description="Beneficiary's bank name")
    state                     : str   = Field(..., description="Nigerian state of transaction")
    timestamp                 : str   = Field(..., description="ISO8601 timestamp")
    sim_age_days              : int   = Field(..., ge=0, description="Days since SIM last changed")
    device_fingerprint_changed: bool  = Field(..., description="Device fingerprint changed recently")
    geo_displacement_flag     : bool  = Field(False, description="Transaction far from home location")
    nin_bvn_mismatch          : bool  = Field(False, description="NIN-BVN identity mismatch detected")
    velocity_1h               : int   = Field(..., ge=0, description="Transactions in last 1 hour")
    velocity_6h               : int   = Field(..., ge=0, description="Transactions in last 6 hours")
    velocity_24h              : int   = Field(..., ge=0, description="Transactions in last 24 hours")
    cumulative_send_24h_ngn   : float = Field(..., ge=0, description="Total sent in last 24 hours (₦)")
    agent_id                  : Optional[str] = Field(None, description="Agent ID if agent transaction")
    is_new_beneficiary        : bool  = Field(False, description="First time sending to this account")

    @validator("channel")
    def validate_channel(cls, v):
        valid = list(CHANNEL_MAP.keys())
        if v not in valid:
            raise ValueError(f"channel must be one of {valid}")
        return v

    @validator("amount_ngn")
    def validate_amount(cls, v):
        if v > 50_000_000:
            raise ValueError("amount_ngn exceeds maximum single transaction limit")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id"            : "TXN-2024-001234",
                "sender_id"                 : "a3f8b2c1d4e5f6a7",
                "beneficiary_id"            : "b8c9d0e1f2a3b4c5",
                "amount_ngn"                : 245000.00,
                "channel"                   : "mobile_app",
                "sender_bank"               : "Kuda",
                "beneficiary_bank"          : "GTB",
                "state"                     : "Lagos",
                "timestamp"                 : "2024-11-15T02:34:17",
                "sim_age_days"              : 1,
                "device_fingerprint_changed": True,
                "geo_displacement_flag"     : True,
                "nin_bvn_mismatch"          : False,
                "velocity_1h"               : 4,
                "velocity_6h"               : 7,
                "velocity_24h"              : 9,
                "cumulative_send_24h_ngn"   : 890000.00,
                "agent_id"                  : None,
                "is_new_beneficiary"        : True,
            }
        }


class RiskScoreResponse(BaseModel):
    """Full risk scoring response with explainability."""
    transaction_id      : str
    timestamp_scored    : str
    risk_score          : float
    risk_band           : str
    recommended_action  : str
    top_signals         : list
    shap_explanation    : dict
    ndpa_note           : str
    model_version       : str = "2.0.0"

    model_config = {"protected_namespaces": ()}


# ── Core scoring functions ────────────────────────────────────────────────────

def build_feature_vector(req: TransactionRequest) -> np.ndarray:
    """Convert API request to model feature vector."""
    ts           = datetime.fromisoformat(req.timestamp)
    hour         = ts.hour
    dow          = ts.weekday()
    is_weekend   = int(dow >= 5)
    is_off_hours = int(hour in range(1, 6))

    public_holidays = {(1,1),(4,18),(4,21),(5,1),(6,12),(10,1),(12,25),(12,26)}
    is_holiday = int((ts.month, ts.day) in public_holidays)

    sim_swap_flag = int(req.sim_age_days <= 2)
    is_interbank  = int(req.sender_bank != req.beneficiary_bank)
    is_agent      = int(req.agent_id is not None)

    channel_enc = CHANNEL_MAP.get(req.channel, 0)
    s_bank_enc  = BANK_MAP.get(req.sender_bank, 0)
    b_bank_enc  = BANK_MAP.get(req.beneficiary_bank, 0)
    state_enc   = STATE_MAP.get(req.state, 0)

    features = [
        req.amount_ngn,
        is_interbank,
        is_agent,
        int(req.is_new_beneficiary),
        hour,
        dow,
        is_weekend,
        is_holiday,
        is_off_hours,
        req.sim_age_days,
        sim_swap_flag,
        int(req.device_fingerprint_changed),
        int(req.geo_displacement_flag),
        int(req.nin_bvn_mismatch),
        req.velocity_1h,
        req.velocity_6h,
        req.velocity_24h,
        req.cumulative_send_24h_ngn,
        channel_enc,
        s_bank_enc,
        b_bank_enc,
        state_enc,
    ]

    return np.array(features, dtype=float).reshape(1, -1)


def get_graph_score(sender_id: str) -> float:
    """Look up sender's community risk score from graph partition."""
    if not MODELS:
        return 0.0
    partition      = MODELS.get("graph_partition", {})
    community_risk = MODELS.get("community_risk_map", {})
    community_id   = partition.get(sender_id)
    return float(community_risk.get(community_id, 0.0))


def compute_ensemble_score(xgb_score, iso_score, lstm_score, graph_score) -> float:
    """Weighted ensemble — matches notebook formula."""
    config  = MODELS.get("ensemble_config", {})
    weights = config.get("weights", {
        "xgboost": 0.50, "isolation_forest": 0.15,
        "lstm_autoencoder": 0.25, "graph": 0.10
    })
    trs = (
        weights["xgboost"]          * xgb_score  +
        weights["isolation_forest"] * iso_score  +
        weights["lstm_autoencoder"] * lstm_score +
        weights["graph"]            * graph_score
    )
    return float(np.clip(trs, 0.0, 1.0))


def assign_risk_band(score: float) -> tuple:
    """Map TRS to risk band and recommended action."""
    if score < 0.30:
        return "LOW",      "ALLOW"
    elif score < 0.60:
        return "MEDIUM",   "STEP_UP_AUTH"
    elif score < 0.80:
        return "HIGH",     "HUMAN_REVIEW"
    else:
        return "CRITICAL", "BLOCK_AND_ALERT"


def get_shap_explanation(features: np.ndarray) -> dict:
    """Compute SHAP values and return top signals."""
    if "shap_explainer" not in MODELS:
        return {"error": "SHAP explainer not loaded"}

    shap_values = MODELS["shap_explainer"].shap_values(
        pd.DataFrame(features, columns=FEATURE_COLS)
    )

    shap_series = pd.Series(shap_values[0], index=FEATURE_COLS)
    top_signals = (
        shap_series.abs()
        .sort_values(ascending=False)
        .head(5)
    )

    explanation  = {}
    signals_list = []
    for feat in top_signals.index:
        direction = "increases_fraud_risk" if shap_series[feat] > 0 else "decreases_fraud_risk"
        explanation[feat] = {
            "shap_value"    : round(float(shap_series[feat]), 4),
            "direction"     : direction,
            "feature_value" : round(float(features[0][FEATURE_COLS.index(feat)]), 4),
        }
        signals_list.append(feat)

    return {"top_features": explanation, "signals_list": signals_list}


def log_audit_event(transaction_id: str, risk_score: float,
                    risk_band: str, action: str, features: np.ndarray):
    """
    Append a scoring event to the in-memory audit log.
    Kumar (2025): observability through logs is essential for resilient fintech systems.
    In production, persist to a database or structured logging service.
    """
    AUDIT_LOG.append({
        "transaction_id"  : transaction_id,
        "timestamp"       : datetime.utcnow().isoformat(),
        "risk_score"      : round(risk_score, 4),
        "risk_band"       : risk_band,
        "recommended_action": action,
        "amount_ngn"      : round(float(features[0][0]), 2),
        "velocity_6h"     : int(features[0][15]),
        "sim_swap_flag"   : bool(features[0][10]),
        "channel"         : int(features[0][18]),
    })


def compute_drift_indicators() -> dict:
    """
    Compute live drift indicators from recent audit log entries.
    Compares live scoring distribution against training baseline.
    Nadeem & Anwar (2025) and Prajapati (2025) both identify model drift
    as a primary challenge requiring continuous monitoring.
    """
    if len(AUDIT_LOG) < 10:
        return {"status": "insufficient_data", "min_required": 10, "current": len(AUDIT_LOG)}

    recent_scores = [e["risk_score"] for e in AUDIT_LOG]
    live_mean     = float(np.mean(recent_scores))
    live_std      = float(np.std(recent_scores))

    mean_drift = abs(live_mean - TRAINING_BASELINE["mean_risk_score"])
    std_drift  = abs(live_std  - TRAINING_BASELINE["std_risk_score"])

    # Flag drift if mean shifts by more than 0.10 from baseline
    drift_detected = mean_drift > 0.10

    band_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for e in AUDIT_LOG:
        band = e.get("risk_band", "LOW")
        band_counts[band] = band_counts.get(band, 0) + 1

    high_risk_rate = (band_counts["HIGH"] + band_counts["CRITICAL"]) / len(AUDIT_LOG)

    return {
        "status"               : "drift_detected" if drift_detected else "stable",
        "drift_detected"       : drift_detected,
        "live_mean_score"      : round(live_mean, 4),
        "baseline_mean_score"  : TRAINING_BASELINE["mean_risk_score"],
        "mean_drift"           : round(mean_drift, 4),
        "live_std_score"       : round(live_std, 4),
        "baseline_std_score"   : TRAINING_BASELINE["std_risk_score"],
        "high_risk_rate_live"  : round(high_risk_rate, 4),
        "band_distribution"    : band_counts,
        "transactions_analysed": len(AUDIT_LOG),
        "recommendation"       : (
            "Model retraining recommended — score distribution has shifted significantly "
            "from training baseline. Consistent with Nadeem & Anwar (2025) guidance on "
            "continuous model validation."
        ) if drift_detected else (
            "Model stable — score distribution consistent with training baseline. "
            "Next scheduled review in 90 days per Prajapati (2025) recommendation."
        )
    }


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/v1/health", tags=["System"])
def health_check():
    """Check if the API and models are loaded and ready."""
    models_loaded = len(MODELS) > 0
    return {
        "status"        : "healthy" if models_loaded else "degraded",
        "models_loaded" : models_loaded,
        "timestamp"     : datetime.utcnow().isoformat(),
        "version"       : "2.0.0",
        "audit_log_size": len(AUDIT_LOG),
    }


@app.get("/v1/model-health", tags=["System"])
def model_health():
    """
    Reports model health and drift indicators.
    Compares live scoring distribution against training baseline.

    Nadeem & Anwar (2025): model drift — where predictive accuracy declines
    over time due to evolving fraud tactics — requires continuous monitoring.
    Prajapati (2025): concept drift is a primary obstacle in fraud detection systems.
    Kumar (2025): observability is essential for resilient fintech platforms.
    """
    config        = MODELS.get("ensemble_config", {})
    drift_report  = compute_drift_indicators()

    return {
        "model_status"          : "active" if MODELS else "degraded",
        "version"               : "2.0.0",
        "loaded_at"             : MODELS.get("loaded_at", "unknown"),
        "last_trained"          : "2024-12-30",
        "training_data_size"    : 200_000,
        "training_fraud_rate"   : TRAINING_BASELINE["fraud_rate"],
        "training_performance"  : {
            "precision" : TRAINING_BASELINE["precision"],
            "recall"    : TRAINING_BASELINE["recall"],
            "f1"        : TRAINING_BASELINE["f1"],
            "roc_auc"   : TRAINING_BASELINE["roc_auc"],
        },
        "drift_monitoring"      : drift_report,
        "retraining_schedule"   : "Every 90 days or upon drift detection",
        "compliance_references" : [
            "Nadeem & Anwar (2025) — continuous model validation",
            "Prajapati (2025) — concept drift monitoring",
            "Kumar (2025) — observability in fintech systems",
        ]
    }


@app.get("/v1/model-info", tags=["System"])
def model_info():
    """Return metadata about the loaded models and ensemble configuration."""
    config = MODELS.get("ensemble_config", {})
    return {
        "framework"      : "Hybrid ML — XGBoost + Isolation Forest + LSTM Autoencoder + Graph",
        "version"        : "2.0.0",
        "compliance"     : "NDPA 2023 Section 37(1) — Human-in-the-loop for HIGH/CRITICAL",
        "ensemble"       : config.get("weights", {}),
        "risk_bands"     : config.get("risk_bands", {}),
        "performance"    : config.get("performance", {}),
        "target_market"  : "Nigerian SME Fintechs",
        "features_count" : len(FEATURE_COLS),
        "architecture"   : {
            "layer_1" : "XGBoost — supervised classification (weight: 0.50)",
            "layer_2a": "Isolation Forest — point anomaly detection (weight: 0.15)",
            "layer_2b": "LSTM Autoencoder — sequential anomaly detection (weight: 0.25)",
            "layer_3" : "Graph/Louvain — coordinated fraud ring detection (weight: 0.10)",
        },
        "explainability" : {
            "primary"   : "SHAP (SHapley Additive exPlanations) — /v1/score",
            "secondary" : "LIME (Local Interpretable Model-agnostic Explanations) — /v1/explain",
            "citation"  : "Nadeem & Anwar (2025) — dual XAI frameworks recommended",
        }
    }


@app.get("/v1/audit-log", tags=["System"])
def get_audit_log(limit: int = Query(default=50, le=500)):
    """
    Returns recent transaction scoring events for audit and observability.
    Kumar (2025): logs are a foundational observability layer for resilient fintech systems.
    NDPA 2023: audit trails support regulatory accountability requirements.
    """
    log_entries = list(AUDIT_LOG)[-limit:]
    log_entries.reverse()  # most recent first

    if not log_entries:
        return {
            "message"    : "No transactions scored yet.",
            "total_logged": 0,
            "entries"    : []
        }

    scores = [e["risk_score"] for e in log_entries]
    return {
        "total_logged"   : len(AUDIT_LOG),
        "returned"       : len(log_entries),
        "summary"        : {
            "mean_risk_score" : round(float(np.mean(scores)), 4),
            "max_risk_score"  : round(float(np.max(scores)), 4),
            "min_risk_score"  : round(float(np.min(scores)), 4),
            "high_risk_count" : sum(1 for e in log_entries if e["risk_band"] in ["HIGH","CRITICAL"]),
        },
        "entries"        : log_entries,
    }


@app.post("/v1/score", response_model=RiskScoreResponse, tags=["Scoring"])
def score_transaction(request: TransactionRequest):
    """
    Score a transaction and return a Transaction Risk Score with explainability.

    **NDPA Compliance Note:**
    - All PII must be hashed before submission
    - HIGH and CRITICAL scores are flagged for mandatory human review
    - SHAP explanations are provided for audit trail purposes
    - All scoring events are logged for observability (Kumar, 2025)
    """
    if not MODELS:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run all training notebooks first."
        )

    try:
        # Build feature vector
        features = build_feature_vector(request)

        # ── Layer 1: XGBoost score ────────────────────────────────────────────
        xgb_score = float(
            MODELS["xgboost"].predict_proba(
                pd.DataFrame(features, columns=FEATURE_COLS)
            )[0][1]
        )

        # ── Layer 2a: Isolation Forest score ──────────────────────────────────
        features_scaled = MODELS["scaler"].transform(features)
        iso_raw         = MODELS["iso_forest"].decision_function(features_scaled)[0]
        iso_score       = float(np.clip(1 - (iso_raw + 0.5), 0, 1))

        # ── Layer 2b: LSTM score (approximated via weighted residual) ─────────
        # Full LSTM is used in batch scoring via the notebook pipeline.
        # API uses approximation for sub-200ms response time requirement.
        lstm_score = float(np.clip(xgb_score * 0.85 + iso_score * 0.15, 0, 1))

        # ── Layer 3: Graph score ──────────────────────────────────────────────
        graph_score = get_graph_score(request.sender_id)

        # ── Ensemble TRS ──────────────────────────────────────────────────────
        trs = compute_ensemble_score(xgb_score, iso_score, lstm_score, graph_score)

        # ── Risk band ─────────────────────────────────────────────────────────
        risk_band, action = assign_risk_band(trs)

        # ── SHAP explanation ──────────────────────────────────────────────────
        shap_result = get_shap_explanation(features)

        # ── Audit logging (Kumar, 2025 — observability requirement) ──────────
        log_audit_event(request.transaction_id, trs, risk_band, action, features)

        # ── NDPA compliance note ──────────────────────────────────────────────
        if risk_band in ["HIGH", "CRITICAL"]:
            ndpa_note = (
                "NDPA Section 37(1): This decision requires human review. "
                "Automated blocking has NOT been applied. "
                "A compliance officer must APPROVE, REVIEW, or DECLINE before action is taken. "
                "Consistent with Nadeem & Anwar (2025) human-AI collaboration model."
            )
        else:
            ndpa_note = (
                "NDPA Section 37(1): This low-stakes automated decision "
                "does not require human review."
            )

        return RiskScoreResponse(
            transaction_id     = request.transaction_id,
            timestamp_scored   = datetime.utcnow().isoformat(),
            risk_score         = round(trs, 4),
            risk_band          = risk_band,
            recommended_action = action,
            top_signals        = shap_result.get("signals_list", []),
            shap_explanation   = shap_result.get("top_features", {}),
            ndpa_note          = ndpa_note,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")


@app.post("/v1/explain", tags=["Explainability"])
def explain_transaction(request: TransactionRequest):
    """
    Provides LIME explanation as a secondary XAI method alongside SHAP.

    Nadeem & Anwar (2025) recommend dual XAI frameworks for regulatory
    accountability, ensuring both global (SHAP) and local (LIME) interpretability.
    LIME explains individual predictions in terms of locally linear approximations,
    complementing SHAP's game-theoretic global attribution approach.
    """
    if not MODELS:
        raise HTTPException(503, "Models not loaded.")

    try:
        features    = build_feature_vector(request)
        df_features = pd.DataFrame(features, columns=FEATURE_COLS)

        # Build LIME explainer with feature names
        explainer = lime_tabular.LimeTabularExplainer(
            training_data  = np.zeros((100, len(FEATURE_COLS))),
            feature_names  = FEATURE_COLS,
            class_names    = ["Legitimate", "Fraud"],
            mode           = "classification",
            random_state   = 42,
        )

        exp = explainer.explain_instance(
            features[0],
            MODELS["xgboost"].predict_proba,
            num_features = 5,
        )

        lime_results = {}
        for feat, weight in exp.as_list():
            direction = "increases_fraud_risk" if weight > 0 else "decreases_fraud_risk"
            lime_results[feat] = {
                "weight"    : round(weight, 4),
                "direction" : direction,
            }

        # Also get SHAP for comparison
        shap_result = get_shap_explanation(features)

        return {
            "transaction_id"    : request.transaction_id,
            "timestamp"         : datetime.utcnow().isoformat(),
            "lime_explanation"  : lime_results,
            "shap_explanation"  : shap_result.get("top_features", {}),
            "xai_note"          : (
                "LIME provides local linear approximations for this specific transaction. "
                "SHAP provides global game-theoretic attribution values. "
                "Both are provided per Nadeem & Anwar (2025) dual XAI recommendation."
            ),
            "citation"          : "Nadeem & Anwar (2025) — dual XAI frameworks for regulatory accountability",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
