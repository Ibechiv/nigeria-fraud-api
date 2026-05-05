"""
synthetic_generator.py
======================
Generates 200,000 synthetic Nigerian fintech transactions
with Nigeria-specific fraud signals for ML research.

Usage:
    python synthetic_generator.py

Output:
    data/transactions_raw.csv       — full dataset
    data/transactions_train.csv     — 70% training split
    data/transactions_val.csv       — 15% validation split
    data/transactions_test.csv      — 15% test split (chronologically latest)

Research Note:
    Fraud rate (~0.5%) and feature distributions are calibrated to approximate
    NIBSS 2024 aggregate statistics. These are research approximations and must
    be documented as such in any thesis or publication.

Author: [Your Name]
Project: Moving from Rule-Based to AI/ML Fraud Detection — Nigerian Fintech
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import hashlib
import random
import warnings
warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Configuration ─────────────────────────────────────────────────────────────
N_TRANSACTIONS   = 200_000
FRAUD_RATE       = 0.005          # 0.5% — approximates NIBSS 2024 aggregate ratio
START_DATE       = datetime(2024, 1, 1)
END_DATE         = datetime(2024, 12, 31)
OUTPUT_DIR       = "data"

# Nigeria-specific constants
CHANNELS         = ["USSD", "mobile_app", "POS", "web", "ATM"]
CHANNEL_WEIGHTS  = [0.38, 0.34, 0.14, 0.08, 0.06]   # USSD/mobile dominant in Nigeria

BANKS            = ["GTB", "Access", "Zenith", "UBA", "Kuda",
                    "Opay", "Moniepoint", "PalmPay", "FCMB", "Sterling"]

STATES           = ["Lagos", "Abuja", "Kano", "Rivers", "Oyo",
                    "Kaduna", "Anambra", "Delta", "Enugu", "Ogun"]
# Lagos/Abuja/Kano carry disproportionate fraud volume per NIBSS data
STATE_WEIGHTS    = [0.30, 0.18, 0.12, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.04]

N_USERS          = 15_000
N_AGENTS         = 500
N_BENEFICIARIES  = 20_000


# ── Helper functions ──────────────────────────────────────────────────────────

def hash_id(value: str) -> str:
    """Pseudonymise IDs — simulates NDPA-compliant hashing of BVN/account numbers."""
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def random_timestamp(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def transaction_amount(is_fraud: bool, channel: str) -> float:
    """
    Generate transaction amount in Naira.
    Fraud patterns:
      - SIM-swap fraud: large single transfers (₦50k–₦2M)
      - Low-and-slow splitting: many small amounts (₦5k–₦50k)
      - BEC: large corporate amounts (₦500k–₦10M)
    Legitimate: mostly small daily transactions.
    """
    if not is_fraud:
        # Legitimate: log-normal centred around ₦15,000
        amount = np.random.lognormal(mean=9.6, sigma=1.2)
        return round(min(max(amount, 100), 5_000_000), 2)
    else:
        fraud_type = np.random.choice(
            ["sim_swap", "splitting", "bec", "agent"],
            p=[0.30, 0.35, 0.15, 0.20]
        )
        if fraud_type == "sim_swap":
            return round(np.random.uniform(50_000, 2_000_000), 2)
        elif fraud_type == "splitting":
            return round(np.random.uniform(5_000, 49_999), 2)   # below typical alert threshold
        elif fraud_type == "bec":
            return round(np.random.uniform(500_000, 10_000_000), 2)
        else:  # agent fraud
            return round(np.random.uniform(10_000, 300_000), 2)


def sim_age_days(is_fraud: bool) -> int:
    """
    Days since SIM card was last changed.
    Fraud signal: SIM-swap fraudsters act within 72h of swap.
    """
    if is_fraud and np.random.random() < 0.45:
        return np.random.randint(0, 3)       # 0–72 hours — strong SIM-swap signal
    return np.random.randint(3, 1825)        # 3 days to 5 years for legitimate users


def transaction_velocity_window(is_fraud: bool, fraud_type_hint: str) -> dict:
    """
    Compute transaction counts in rolling time windows.
    Low-and-slow splitting: elevated 24h count, moderate 1h count.
    SIM-swap: burst in 1h window.
    """
    if not is_fraud:
        v1h  = np.random.choice([0, 1, 2, 3], p=[0.60, 0.25, 0.10, 0.05])
        v6h  = v1h + np.random.randint(0, 4)
        v24h = v6h + np.random.randint(0, 8)
    elif fraud_type_hint == "splitting":
        v1h  = np.random.randint(2, 6)
        v6h  = np.random.randint(8, 20)
        v24h = np.random.randint(20, 48)
    elif fraud_type_hint == "sim_swap":
        v1h  = np.random.randint(3, 10)
        v6h  = np.random.randint(5, 12)
        v24h = np.random.randint(6, 15)
    else:
        v1h  = np.random.randint(1, 5)
        v6h  = np.random.randint(2, 8)
        v24h = np.random.randint(3, 12)

    return {"velocity_1h": v1h, "velocity_6h": v6h, "velocity_24h": v24h}


# ── Main generation function ──────────────────────────────────────────────────

def generate_dataset(n: int = N_TRANSACTIONS) -> pd.DataFrame:

    print(f"Generating {n:,} transactions...")
    print(f"  Target fraud rate : {FRAUD_RATE*100:.1f}%")
    print(f"  Date range        : {START_DATE.date()} → {END_DATE.date()}")
    print(f"  Random seed       : {SEED}\n")

    # Pre-generate user/agent/beneficiary ID pools
    user_ids        = [hash_id(f"USER_{i}") for i in range(N_USERS)]
    agent_ids       = [hash_id(f"AGENT_{i}") for i in range(N_AGENTS)]
    beneficiary_ids = [hash_id(f"BENE_{i}") for i in range(N_BENEFICIARIES)]

    records = []

    for i in range(n):

        # ── Fraud label ───────────────────────────────────────────────────────
        is_fraud = np.random.random() < FRAUD_RATE

        # Assign fraud sub-type (used to shape correlated features)
        if is_fraud:
            fraud_subtype = np.random.choice(
                ["sim_swap", "splitting", "bec", "agent_fraud"],
                p=[0.30, 0.35, 0.15, 0.20]
            )
        else:
            fraud_subtype = "none"

        # ── Core transaction fields ───────────────────────────────────────────
        timestamp    = random_timestamp(START_DATE, END_DATE)
        channel      = np.random.choice(CHANNELS, p=CHANNEL_WEIGHTS)
        sender_id    = np.random.choice(user_ids)
        amount       = transaction_amount(is_fraud, channel)
        state        = np.random.choice(STATES, p=STATE_WEIGHTS)

        # Agent channel: ~20% of USSD and POS transactions go through agents
        is_agent_txn = (channel in ["USSD", "POS"]) and (np.random.random() < 0.20)
        agent_id     = np.random.choice(agent_ids) if is_agent_txn else None

        # Beneficiary — fraud more likely to use new/unseen beneficiaries
        if is_fraud and np.random.random() < 0.70:
            # New beneficiary — not in user's normal set
            beneficiary_id      = np.random.choice(beneficiary_ids[-5000:])
            is_new_beneficiary  = True
        else:
            beneficiary_id      = np.random.choice(beneficiary_ids[:15000])
            is_new_beneficiary  = np.random.random() < 0.05

        # ── Nigeria-specific signals ──────────────────────────────────────────
        sim_days     = sim_age_days(is_fraud)
        sim_swap_flag = sim_days <= 2   # SIM changed within 48h

        # Device fingerprint change — correlated with SIM-swap fraud
        if fraud_subtype == "sim_swap":
            device_changed = np.random.random() < 0.85
        else:
            device_changed = np.random.random() < 0.03

        # USSD metadata — only for USSD channel
        if channel == "USSD":
            ussd_session_duration_s = (
                np.random.randint(5, 45) if is_fraud     # rushed fraudulent sessions
                else np.random.randint(30, 180)          # normal user browsing
            )
            ussd_retry_count = (
                np.random.randint(0, 2) if not is_fraud
                else np.random.randint(0, 1)             # fraudsters know the menus
            )
        else:
            ussd_session_duration_s = None
            ussd_retry_count        = None

        # Geographic displacement — transaction far from user's home state
        geo_displacement_flag = (
            np.random.random() < 0.55 if is_fraud
            else np.random.random() < 0.04
        )

        # NIN-BVN mismatch — identity inconsistency signal
        nin_bvn_mismatch = (
            np.random.random() < 0.40 if is_fraud
            else np.random.random() < 0.01
        )

        # ── Velocity features ─────────────────────────────────────────────────
        vel = transaction_velocity_window(is_fraud, fraud_subtype)

        # ── Time-based features ───────────────────────────────────────────────
        hour_of_day     = timestamp.hour
        day_of_week     = timestamp.weekday()   # 0=Monday
        is_weekend      = day_of_week >= 5
        # Nigerian public holidays (approximate — major ones)
        public_holidays = {
            (1,1), (4,18), (4,21), (5,1), (6,12),
            (10,1), (12,25), (12,26)
        }
        is_public_holiday = (timestamp.month, timestamp.day) in public_holidays

        # Off-hours flag — fraud more common 1am–5am
        is_off_hours = hour_of_day in range(1, 6)

        # ── Cumulative amount features ────────────────────────────────────────
        cumulative_24h_send = amount * vel["velocity_24h"] * np.random.uniform(0.6, 1.4)

        # ── Bank / network ────────────────────────────────────────────────────
        sender_bank      = np.random.choice(BANKS)
        beneficiary_bank = np.random.choice(BANKS)
        is_interbank     = sender_bank != beneficiary_bank

        # ── Assemble record ───────────────────────────────────────────────────
        record = {
            # Identifiers
            "transaction_id"          : hash_id(f"TXN_{i}_{timestamp}"),
            "sender_id"               : sender_id,
            "beneficiary_id"          : beneficiary_id,
            "agent_id"                : agent_id,

            # Core transaction
            "timestamp"               : timestamp,
            "amount_ngn"              : amount,
            "channel"                 : channel,
            "sender_bank"             : sender_bank,
            "beneficiary_bank"        : beneficiary_bank,
            "state"                   : state,

            # Flags
            "is_interbank"            : is_interbank,
            "is_agent_transaction"    : is_agent_txn,
            "is_new_beneficiary"      : is_new_beneficiary,
            "is_weekend"              : is_weekend,
            "is_public_holiday"       : is_public_holiday,
            "is_off_hours"            : is_off_hours,

            # Nigeria-specific signals
            "sim_age_days"            : sim_days,
            "sim_swap_flag"           : sim_swap_flag,
            "device_fingerprint_changed": device_changed,
            "geo_displacement_flag"   : geo_displacement_flag,
            "nin_bvn_mismatch"        : nin_bvn_mismatch,

            # USSD metadata
            "ussd_session_duration_s" : ussd_session_duration_s,
            "ussd_retry_count"        : ussd_retry_count,

            # Velocity features
            "velocity_1h"             : vel["velocity_1h"],
            "velocity_6h"             : vel["velocity_6h"],
            "velocity_24h"            : vel["velocity_24h"],
            "cumulative_send_24h_ngn" : round(cumulative_24h_send, 2),

            # Time features
            "hour_of_day"             : hour_of_day,
            "day_of_week"             : day_of_week,

            # Target
            "fraud_subtype"           : fraud_subtype,
            "is_fraud"                : int(is_fraud),
        }

        records.append(record)

        if (i + 1) % 50_000 == 0:
            print(f"  {i+1:,} / {n:,} transactions generated...")

    df = pd.DataFrame(records)

    # Sort chronologically — CRITICAL for valid train/test splitting
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def split_dataset(df: pd.DataFrame):
    """
    Temporal train/val/test split — 70/15/15.
    Test set is chronologically LATEST to prevent data leakage.
    This is mandatory for time-series fraud data.
    """
    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    return train, val, test


def print_summary(df, train, val, test):
    print("\n" + "="*55)
    print("DATASET SUMMARY")
    print("="*55)
    print(f"Total transactions : {len(df):,}")
    print(f"Fraud transactions : {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Date range         : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print()
    print(f"Train set  : {len(train):,} rows | fraud: {train['is_fraud'].sum():,} ({train['is_fraud'].mean()*100:.2f}%)")
    print(f"Val set    : {len(val):,} rows   | fraud: {val['is_fraud'].sum():,} ({val['is_fraud'].mean()*100:.2f}%)")
    print(f"Test set   : {len(test):,} rows  | fraud: {test['is_fraud'].sum():,} ({test['is_fraud'].mean()*100:.2f}%)")
    print()
    print("Fraud subtype distribution:")
    fraud_only = df[df['is_fraud'] == 1]
    print(fraud_only['fraud_subtype'].value_counts().to_string())
    print()
    print("Channel distribution:")
    print(df['channel'].value_counts().to_string())
    print("="*55)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate
    df = generate_dataset(N_TRANSACTIONS)

    # Split
    train, val, test = split_dataset(df)

    # Save
    print("\nSaving datasets...")
    df.to_csv(f"{OUTPUT_DIR}/transactions_raw.csv", index=False)
    train.to_csv(f"{OUTPUT_DIR}/transactions_train.csv", index=False)
    val.to_csv(f"{OUTPUT_DIR}/transactions_val.csv", index=False)
    test.to_csv(f"{OUTPUT_DIR}/transactions_test.csv", index=False)

    print_summary(df, train, val, test)

    print(f"\nFiles saved to ./{OUTPUT_DIR}/")
    print("Next step: open eda.ipynb and run your exploratory data analysis.")
