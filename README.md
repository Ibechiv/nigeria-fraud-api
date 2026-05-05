# Nigeria Fraud Detection API

An open-source, NDPA-compliant transaction risk scoring API pre-engineered with Nigeria-specific fraud signals. Built for SME fintechs without requiring in-house ML expertise.

## Features

- **Multi-layer Ensemble Model**: XGBoost + Isolation Forest + LSTM Autoencoder + Graph-based scoring
- **Nigeria-Specific Signals**: SIM swap detection, NIN-BVN verification, geo-displacement, velocity patterns
- **NDPA 2023 Compliance**: Automatic flagging of HIGH/CRITICAL transactions for human review
- **SHAP Explainability**: Understand why each transaction is flagged with explainable AI
- **Ready for Production**: Docker-ready, scalable FastAPI backend

## Risk Bands

| Score | Band | Action |
|-------|------|--------|
| 0.0 – 0.3 | LOW | Allow |
| 0.3 – 0.6 | MEDIUM | Step-up authentication |
| 0.6 – 0.8 | HIGH | Human review (NDPA §37(1)) |
| 0.8 – 1.0 | CRITICAL | Block + compliance alert |

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/Ibechiv/nigeria-fraud-api.git
cd nigeria-fraud-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn api.main:app --reload --port 8000
```

Visit http://localhost:8000/docs for interactive API documentation.

### Deploy to Render

1. **Push to GitHub** (already done ✓)
   
2. **Create a new Web Service on Render**:
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository (`nigeria-fraud-api`)
   - Choose the `main` branch

3. **Configure the service**:
   - **Name**: `nigeria-fraud-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free (or Starter for production)

4. **Set Environment Variables** (if needed):
   - `PYTHONUNBUFFERED=1` (already in render.yaml)

5. **Deploy**: Click "Create Web Service"

> The `Procfile` and `render.yaml` are already configured for you!

### API Endpoints

#### Health Check
```bash
GET /v1/health
```

#### Model Information
```bash
GET /v1/model-info
```

#### Score a Transaction
```bash
POST /v1/score
Content-Type: application/json

{
  "transaction_id": "TXN-2024-001234",
  "sender_id": "a3f8b2c1d4e5f6a7",
  "beneficiary_id": "b8c9d0e1f2a3b4c5",
  "amount_ngn": 245000.00,
  "channel": "mobile_app",
  "sender_bank": "Kuda",
  "beneficiary_bank": "GTB",
  "state": "Lagos",
  "timestamp": "2024-11-15T02:34:17",
  "sim_age_days": 1,
  "device_fingerprint_changed": true,
  "geo_displacement_flag": true,
  "nin_bvn_mismatch": false,
  "velocity_1h": 4,
  "velocity_6h": 7,
  "velocity_24h": 9,
  "cumulative_send_24h_ngn": 890000.00,
  "agent_id": null,
  "is_new_beneficiary": true
}
```

**Response:**
```json
{
  "transaction_id": "TXN-2024-001234",
  "timestamp_scored": "2024-11-15T02:35:42.123456",
  "risk_score": 0.72,
  "risk_band": "HIGH",
  "recommended_action": "HUMAN_REVIEW",
  "top_signals": ["velocity_24h", "device_fingerprint_changed", "geo_displacement_flag"],
  "shap_explanation": {
    "velocity_24h": {
      "shap_value": 0.31,
      "direction": "increases_fraud_risk",
      "feature_value": 9.0
    }
  },
  "ndpa_note": "NDPA Section 37(1): This decision requires human review...",
  "model_version": "1.0.0"
}
```

## NDPA 2023 Compliance

This API implements automatic human-in-the-loop safeguards required by Section 37(1) of Nigeria's Data Protection Regulation:

- ✓ HIGH and CRITICAL transactions are **never** fully automated
- ✓ All decisions are routed to a compliance review queue
- ✓ SHAP explanations provide audit trails for regulatory review
- ✓ Caller must pre-hash all PII before submission

## Architecture

- **Layer 1**: XGBoost gradient boosting classifier (50% weight)
- **Layer 2a**: Isolation Forest anomaly detection (15% weight)
- **Layer 2b**: LSTM Autoencoder reconstruction error (25% weight)
- **Layer 3**: Graph-based community risk scoring (10% weight)

Final score is a weighted ensemble of all four signals.

## Model Training

See the Jupyter notebooks for training details:
- `eda.ipynb` - Exploratory data analysis
- `model_layer1_xgboost.ipynb` - XGBoost training
- `model_layer2_anomaly.ipynb` - Isolation Forest + LSTM training
- `model_layer3_graph.ipynb` - Graph-based community risk
- `model_ensemble.ipynb` - Ensemble combination

## Troubleshooting Render Deployment

### Issue: "Models not loaded"
- **Cause**: Model files in `models/` directory weren't found
- **Fix**: Ensure all `.pkl`, `.keras`, and `.json` files are committed to git

### Issue: Port error
- **Fix**: Already handled! The app now respects the `PORT` environment variable

### Issue: Import errors
- **Fix**: All dependencies are in `requirements.txt`. Run `pip install -r requirements.txt` locally first to verify

## Testing

```bash
# Test imports
python test_imports.py

# Test API locally
python test_api.py
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## License

Research project for Nigerian fintech fraud detection

## Contact

For questions or issues, open a GitHub issue on the repository.
