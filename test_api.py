"""
test_api.py
===========
Tests the Transaction Risk Scoring API endpoints.
Run this after starting the API with: uvicorn api.main:app --reload

Usage:
    python test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

# ── Test 1: Health Check ──────────────────────────────────────────────────────
print_section("TEST 1 — Health Check")
r = requests.get(f"{BASE_URL}/v1/health")
print(f"Status Code : {r.status_code}")
print(json.dumps(r.json(), indent=2))

# ── Test 2: Model Info ────────────────────────────────────────────────────────
print_section("TEST 2 — Model Info")
r = requests.get(f"{BASE_URL}/v1/model-info")
print(f"Status Code : {r.status_code}")
print(json.dumps(r.json(), indent=2))

# ── Test 3: Low-Risk Transaction ──────────────────────────────────────────────
print_section("TEST 3 — Low-Risk Transaction (Expected: LOW)")
low_risk_txn = {
    "transaction_id"            : "TXN-TEST-LOW-001",
    "sender_id"                 : "a3f8b2c1d4e5f6a7",
    "beneficiary_id"            : "b8c9d0e1f2a3b4c5",
    "amount_ngn"                : 5000.00,
    "channel"                   : "mobile_app",
    "sender_bank"               : "Kuda",
    "beneficiary_bank"          : "Kuda",
    "state"                     : "Lagos",
    "timestamp"                 : "2024-11-15T14:30:00",
    "sim_age_days"              : 365,
    "device_fingerprint_changed": False,
    "geo_displacement_flag"     : False,
    "nin_bvn_mismatch"          : False,
    "velocity_1h"               : 0,
    "velocity_6h"               : 1,
    "velocity_24h"              : 2,
    "cumulative_send_24h_ngn"   : 8000.00,
    "is_new_beneficiary"        : False,
}

r = requests.post(f"{BASE_URL}/v1/score", json=low_risk_txn)
print(f"Status Code : {r.status_code}")
result = r.json()
print(f"Risk Score  : {result['risk_score']}")
print(f"Risk Band   : {result['risk_band']}")
print(f"Action      : {result['recommended_action']}")
print(f"Top Signals : {result['top_signals']}")

# ── Test 4: High-Risk SIM-Swap Transaction ────────────────────────────────────
print_section("TEST 4 — SIM-Swap Fraud Attempt (Expected: HIGH or CRITICAL)")
sim_swap_txn = {
    "transaction_id"            : "TXN-TEST-SIMSWAP-001",
    "sender_id"                 : "f9e8d7c6b5a4f3e2",
    "beneficiary_id"            : "d4c3b2a1e0f9d8c7",
    "amount_ngn"                : 850000.00,
    "channel"                   : "USSD",
    "sender_bank"               : "GTB",
    "beneficiary_bank"          : "Opay",
    "state"                     : "Kano",
    "timestamp"                 : "2024-11-15T02:17:44",
    "sim_age_days"              : 1,
    "device_fingerprint_changed": True,
    "geo_displacement_flag"     : True,
    "nin_bvn_mismatch"          : True,
    "velocity_1h"               : 5,
    "velocity_6h"               : 8,
    "velocity_24h"              : 11,
    "cumulative_send_24h_ngn"   : 2400000.00,
    "is_new_beneficiary"        : True,
}

r = requests.post(f"{BASE_URL}/v1/score", json=sim_swap_txn)
print(f"Status Code : {r.status_code}")
result = r.json()
print(f"Risk Score  : {result['risk_score']}")
print(f"Risk Band   : {result['risk_band']}")
print(f"Action      : {result['recommended_action']}")
print(f"Top Signals : {result['top_signals']}")
print(f"NDPA Note   : {result['ndpa_note']}")
print()
print("SHAP Explanation (top features):")
for feat, details in result['shap_explanation'].items():
    direction = "↑ FRAUD" if details['direction'] == 'increases_fraud_risk' else "↓ LEGIT"
    print(f"  {feat:<35} SHAP={details['shap_value']:+.4f}  {direction}")

# ── Test 5: Low-and-Slow Splitting ────────────────────────────────────────────
print_section("TEST 5 — Low-and-Slow Splitting (Expected: MEDIUM or HIGH)")
splitting_txn = {
    "transaction_id"            : "TXN-TEST-SPLIT-001",
    "sender_id"                 : "c1b2a3f4e5d6c7b8",
    "beneficiary_id"            : "e5f6a7b8c9d0e1f2",
    "amount_ngn"                : 47500.00,
    "channel"                   : "mobile_app",
    "sender_bank"               : "Access",
    "beneficiary_bank"          : "PalmPay",
    "state"                     : "Lagos",
    "timestamp"                 : "2024-11-15T16:45:00",
    "sim_age_days"              : 180,
    "device_fingerprint_changed": False,
    "geo_displacement_flag"     : False,
    "nin_bvn_mismatch"          : False,
    "velocity_1h"               : 4,
    "velocity_6h"               : 14,
    "velocity_24h"              : 31,
    "cumulative_send_24h_ngn"   : 1425000.00,
    "is_new_beneficiary"        : True,
}

r = requests.post(f"{BASE_URL}/v1/score", json=splitting_txn)
print(f"Status Code : {r.status_code}")
result = r.json()
print(f"Risk Score  : {result['risk_score']}")
print(f"Risk Band   : {result['risk_band']}")
print(f"Action      : {result['recommended_action']}")
print(f"Top Signals : {result['top_signals']}")

print_section("ALL TESTS COMPLETE")
print("If all tests returned status 200 and sensible risk bands,")
print("your API is working correctly.")
print()
print("Open http://localhost:8000/docs for interactive API documentation.")
