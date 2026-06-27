"""SmartRecon - Affiliate revenue reconciliation platform.

Hybrid agentic design: a deterministic Pandas matching engine handles exact
financial math, and a Gemini-powered Forensic Auditor Agent reasons over a
compact summary of the flagged discrepancies. Single-file Streamlit app,
ready for Streamlit Community Cloud.

requirements.txt:
  streamlit
  pandas
  google-genai

Secrets (optional - falls back to a sidebar input):
  GEMINI_API_KEY
"""

import os
import json
from datetime import datetime

import pandas as pd
import streamlit as st

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_IMPORT_OK = True
except ImportError:
    GENAI_IMPORT_OK = False


st.set_page_config(
    page_title="SmartRecon",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="auto",
)

# Minimal styling only: native Streamlit spacing, theme colors, and fonts
# do the rest. No custom backgrounds, borders, or card treatments.
st.markdown(
    """
    <style>
        .stMainBlockContainer { max-width: 880px; }
        div[data-testid="stMetric"] {
            background: transparent;
            padding: 0;
        }
        div[data-testid="stMetricLabel"] { opacity: 0.65; }
    </style>
    """,
    unsafe_allow_html=True,
)

for key, default in {
    "audit_report_md": None,
    "dispute_email_md": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("SmartRecon")
st.caption("Affiliate revenue reconciliation platform.")
st.write("")


# ----------------------------------------------------------------------------
# Gemini API key resolution: secrets -> environment -> sidebar input.
# ----------------------------------------------------------------------------
def resolve_gemini_api_key() -> str | None:
    try:
        if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key

    if "manual_gemini_key" not in st.session_state:
        st.session_state.manual_gemini_key = ""

    with st.sidebar:
        st.caption("AI Agent")
        st.session_state.manual_gemini_key = st.text_input(
            "Gemini API key",
            value=st.session_state.manual_gemini_key,
            type="password",
            placeholder="AIza...",
        )
        st.caption("Used for this session only. Not stored.")

    return st.session_state.manual_gemini_key or None


GEMINI_API_KEY = resolve_gemini_api_key()
AI_AGENT_READY = bool(GEMINI_API_KEY) and GENAI_IMPORT_OK
GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY) if AI_AGENT_READY else None
GEMINI_MODEL_NAME = "gemini-3.5-flash"


# ----------------------------------------------------------------------------
# Data Ingestion Workspace
# ----------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("Data Ingestion Workspace")
    upload_col1, upload_col2 = st.columns(2)

    with upload_col1:
        ledger_file = st.file_uploader("Upload Internal Ledger", type=["csv"], key="ledger_upload")

    with upload_col2:
        amazon_file = st.file_uploader("Upload Partner Statement", type=["csv"], key="amazon_upload")

files_ready = ledger_file is not None and amazon_file is not None

if not files_ready:
    st.write("")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.info("Upload both files above to start the reconciliation.")
    st.stop()


LEDGER_REQUIRED_COLS = ["transaction_id", "click_timestamp", "product_id", "expected_commission_usd"]
AMAZON_REQUIRED_COLS = ["amazon_order_id", "payout_date", "sku", "actual_payout_usd", "status"]


def load_and_validate_csv(uploaded_file, required_cols: list[str], label: str) -> pd.DataFrame | None:
    """Read an uploaded CSV and verify it contains the required columns."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read '{label}': {e}")
        return None

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"'{label}' is missing column(s): {', '.join(missing)}")
        return None

    return df


ledger_df = load_and_validate_csv(ledger_file, LEDGER_REQUIRED_COLS, "Internal Ledger")
amazon_df = load_and_validate_csv(amazon_file, AMAZON_REQUIRED_COLS, "Partner Statement")

if ledger_df is None or amazon_df is None:
    st.warning("Fix the file(s) above and re-upload to continue.")
    st.stop()


# ----------------------------------------------------------------------------
# Deterministic reconciliation engine (pure Pandas, no AI).
#
# Matching key: ledger.transaction_id <-> amazon.amazon_order_id
#   CLEAN_MATCH        expected ≈ actual (within a 1-cent tolerance)
#   UNDERPAID          actual is less than expected
#   OVERPAID           actual is more than expected
#   MISSING_PAYOUT     ledger row has no matching partner payout
#   UNEXPECTED_PAYOUT  partner row has no matching ledger transaction
#
# Leakage = sum of (expected - actual), floored at 0 per row, for any
# row the publisher is owed money on but didn't receive in full.
# ----------------------------------------------------------------------------
TOLERANCE_USD = 0.01


def run_reconciliation(ledger: pd.DataFrame, amazon: pd.DataFrame) -> dict:
    """Run the deterministic match and return a structured results dict."""
    ledger = ledger.copy()
    amazon = amazon.copy()

    ledger["transaction_id"] = ledger["transaction_id"].astype(str).str.strip()
    amazon["amazon_order_id"] = amazon["amazon_order_id"].astype(str).str.strip()

    merged = pd.merge(
        ledger,
        amazon,
        how="outer",
        left_on="transaction_id",
        right_on="amazon_order_id",
        indicator=True,
    )

    def classify(row) -> str:
        if row["_merge"] == "left_only":
            return "MISSING_PAYOUT"
        if row["_merge"] == "right_only":
            return "UNEXPECTED_PAYOUT"

        expected = row.get("expected_commission_usd")
        actual = row.get("actual_payout_usd")
        if pd.isna(expected) or pd.isna(actual):
            return "DATA_GAP"

        diff = expected - actual
        if abs(diff) <= TOLERANCE_USD:
            return "CLEAN_MATCH"
        elif diff > TOLERANCE_USD:
            return "UNDERPAID"
        else:
            return "OVERPAID"

    merged["match_status"] = merged.apply(classify, axis=1)

    def compute_leakage(row) -> float:
        if row["match_status"] in ("UNDERPAID", "MISSING_PAYOUT", "DATA_GAP"):
            expected = row.get("expected_commission_usd")
            actual = row.get("actual_payout_usd")
            expected = 0.0 if pd.isna(expected) else float(expected)
            actual = 0.0 if pd.isna(actual) else float(actual)
            return max(expected - actual, 0.0)
        return 0.0

    merged["leakage_usd"] = merged.apply(compute_leakage, axis=1)

    clean_matches = merged[merged["match_status"] == "CLEAN_MATCH"]
    anomalies = merged[merged["match_status"] != "CLEAN_MATCH"]

    return {
        "merged": merged,
        "clean_matches": clean_matches,
        "anomalies": anomalies,
        "total_rows": len(merged),
        "clean_count": len(clean_matches),
        "anomaly_count": len(anomalies),
        "total_leakage": float(merged["leakage_usd"].sum()),
    }


recon = run_reconciliation(ledger_df, amazon_df)


# ----------------------------------------------------------------------------
# Executive Summary
# ----------------------------------------------------------------------------
st.write("")
st.subheader("Executive Summary")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.metric("Total Transactions", f"{recon['total_rows']:,}")

with kpi_col2:
    st.metric("Clean Matches", f"{recon['clean_count']:,}")

with kpi_col3:
    st.metric("Anomalies Flagged", f"{recon['anomaly_count']:,}")

with kpi_col4:
    st.metric("Estimated Leakage", f"${recon['total_leakage']:,.2f}")

st.write("")

# Human-readable labels for the technical match_status values, used only
# for display - the underlying values still drive the AI agent payload.
CATEGORY_LABELS = {
    "UNDERPAID": "Underpaid",
    "OVERPAID": "Overpaid",
    "MISSING_PAYOUT": "Missing Payout",
    "UNEXPECTED_PAYOUT": "Unexpected Payout",
    "DATA_GAP": "Data Gap",
}


def build_display_table(anomalies: pd.DataFrame) -> pd.DataFrame:
    """Reduce the anomaly rows to the essential columns for a financial reviewer."""
    df = anomalies.copy()
    df["Transaction ID"] = df["transaction_id"].combine_first(df["amazon_order_id"])
    df["Category"] = df["match_status"].map(CATEGORY_LABELS).fillna(df["match_status"])
    df["Expected Payout"] = df["expected_commission_usd"].map(
        lambda v: f"${v:,.2f}" if pd.notna(v) else "—"
    )
    df["Actual Payout"] = df["actual_payout_usd"].map(
        lambda v: f"${v:,.2f}" if pd.notna(v) else "—"
    )
    df["Variance"] = df["leakage_usd"].map(lambda v: f"-${v:,.2f}" if v else "$0.00")
    return df[["Transaction ID", "Category", "Expected Payout", "Actual Payout", "Variance"]]


if recon["anomaly_count"] > 0:
    with st.expander("View flagged transactions"):
        st.dataframe(
            build_display_table(recon["anomalies"]),
            width="stretch",
            hide_index=True,
        )
else:
    st.success("No anomalies detected. All transactions reconcile cleanly.")

st.divider()


# ----------------------------------------------------------------------------
# Anomaly summarization for the AI agent. Raw rows are never sent to the
# model - Python pre-aggregates counts, totals, and day-of-week patterns
# into a compact JSON payload that the agent reasons over.
# ----------------------------------------------------------------------------
def build_anomaly_summary(recon: dict, max_samples_per_type: int = 5) -> dict:
    """Aggregate anomalies into a compact, LLM-friendly structured summary."""
    anomalies = recon["anomalies"].copy()

    if anomalies.empty:
        return {"has_anomalies": False}

    date_col = None
    for candidate in ["click_timestamp", "payout_date"]:
        if candidate in anomalies.columns:
            date_col = candidate
            break

    if date_col:
        anomalies["_parsed_date"] = pd.to_datetime(anomalies[date_col], errors="coerce")
        anomalies["_day_of_week"] = anomalies["_parsed_date"].dt.day_name()
        anomalies["_date_label"] = anomalies["_parsed_date"].dt.strftime("%A the %-d")
    else:
        anomalies["_day_of_week"] = None
        anomalies["_date_label"] = None

    by_type = []
    for status, group in anomalies.groupby("match_status"):
        day_counts = (
            group["_day_of_week"].dropna().value_counts().to_dict() if date_col else {}
        )
        date_label_counts = (
            group["_date_label"].dropna().value_counts().head(3).to_dict() if date_col else {}
        )

        sample_cols = [
            c for c in [
                "transaction_id", "amazon_order_id", "product_id", "sku",
                "expected_commission_usd", "actual_payout_usd", "status", "leakage_usd",
            ] if c in group.columns
        ]
        samples = group[sample_cols].head(max_samples_per_type).fillna("N/A").to_dict(orient="records")

        by_type.append({
            "anomaly_type": status,
            "count": int(len(group)),
            "total_leakage_usd": round(float(group["leakage_usd"].sum()), 2),
            "day_of_week_distribution": day_counts,
            "top_specific_dates": date_label_counts,
            "sample_rows": samples,
        })

    status_field_dist = (
        anomalies["status"].dropna().value_counts().to_dict()
        if "status" in anomalies.columns else {}
    )

    return {
        "has_anomalies": True,
        "total_anomalies": int(len(anomalies)),
        "total_leakage_usd": round(float(recon["total_leakage"]), 2),
        "anomaly_breakdown": by_type,
        "amazon_status_field_distribution": status_field_dist,
    }


anomaly_summary = build_anomaly_summary(recon)


FORENSIC_AUDITOR_SYSTEM_PROMPT = """You are a Senior Forensic Financial Auditor Agent \
specializing in affiliate marketing revenue reconciliation for retail content publishers.

You will be given a structured JSON summary of payment discrepancies between an internal \
click-tracking ledger and a partner payout statement. This summary has already been \
aggregated by a deterministic Pandas engine - you do NOT need to re-verify the math, \
only reason about WHY these patterns exist and WHAT the publisher should do.

Your job is to think like an experienced auditor who has seen hundreds of affiliate \
statements: spot root causes, surface systemic patterns (timing clusters, day-of-week \
effects, specific dates, SKU concentration), and produce actionable next steps.

Respond ONLY in Markdown using EXACTLY these three top-level headings, in this order:

## Root-Cause Classification
Categorize the discrepancies into these buckets (use the ones that apply, skip ones that don't):
- **Returns/Cancellations** (customer returned the product or cancelled the order)
- **Policy Changes** (commission rate changes, program T&C updates, category exclusions)
- **Tracking Drops** (cookie/click attribution failures, missing clicks, lost referrals)
- **Other/Unclear** (anything that doesn't cleanly fit above)
For each bucket that applies, briefly explain the evidence from the data that supports it.

## Systemic Patterns Detected
Call out concrete, specific patterns - cite exact numbers, percentages, dates, days of the \
week, or SKUs/products wherever the data supports it (e.g. "70% of tracking drops occurred \
on Tuesday the 12th"). If the data doesn't support a strong pattern, say so honestly rather \
than inventing one.

## Recommended Action Items
Give a short, prioritized, numbered list of concrete next steps the publisher's finance \
team should take (e.g. what to dispute, what to monitor, what to fix in their tracking setup).

Be concise, precise, and professional. Use real numbers from the data provided. Never \
fabricate transaction IDs or details that are not present in the input JSON. If the input \
indicates there are no anomalies, state clearly that the books are clean.
"""


def run_forensic_audit_agent(summary: dict) -> str:
    """Send the pre-aggregated anomaly summary (not raw rows) to Gemini for analysis."""
    if not AI_AGENT_READY:
        return ""

    try:
        user_payload = (
            "Here is the structured discrepancy summary from this month's affiliate "
            "reconciliation run. Analyze it per your instructions:\n\n"
            f"```json\n{json.dumps(summary, indent=2, default=str)}\n```"
        )
        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=user_payload,
            config=genai_types.GenerateContentConfig(
                system_instruction=FORENSIC_AUDITOR_SYSTEM_PROMPT,
                temperature=0.3,
            ),
        )
        return response.text
    except Exception as e:
        return f"**AI Agent error:** could not complete analysis ({e})"


# ----------------------------------------------------------------------------
# Forensic Audit Agent
# ----------------------------------------------------------------------------
st.subheader("Forensic Audit Agent")

if not anomaly_summary.get("has_anomalies"):
    st.success("No anomalies were flagged this cycle. Your books reconcile cleanly.")
elif not AI_AGENT_READY:
    st.info("Add a Gemini API key in the sidebar to enable root-cause analysis.")
else:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        run_audit = st.button("Run Audit Agent", type="primary", width="stretch")

    if run_audit or st.session_state.audit_report_md is None:
        with st.spinner("Reviewing flagged anomalies..."):
            st.session_state.audit_report_md = run_forensic_audit_agent(anomaly_summary)

    if st.session_state.audit_report_md:
        st.write("")
        st.markdown(st.session_state.audit_report_md)

st.divider()


# ----------------------------------------------------------------------------
# Dispute email drafting
# ----------------------------------------------------------------------------
DISPUTE_EMAIL_SYSTEM_PROMPT = """You are an AI assistant helping a retail content publisher \
draft a professional dispute email to Amazon Associates / Amazon Affiliate Support.

You will receive: (1) a structured JSON summary of payment discrepancies, and (2) optionally \
a prior forensic audit report. Use BOTH to write a clear, factual, professional dispute email.

Requirements for the email:
- Professional business tone, no exaggeration or accusatory language.
- A clear subject line.
- Open by stating the purpose: a discrepancy found during month-end reconciliation.
- Reference SPECIFIC numbers: total anomaly count, total estimated leakage in USD, and the \
breakdown by anomaly type (e.g. "X transactions were underpaid, totaling $Y").
- Mention the dominant root-cause pattern(s) if the audit report identifies one clearly.
- Politely request: a review of the listed transaction IDs, an explanation for the discrepancy, \
and remediation (corrected payout) where appropriate.
- Close with a professional sign-off placeholder.
- Do NOT invent transaction IDs, dates, or numbers that are not present in the provided data.
- Output ONLY the email (subject line + body) in Markdown. No extra commentary before or after.
"""


def generate_dispute_email(summary: dict, audit_report: str | None) -> str:
    """Use the LLM to draft a structured dispute email grounded in the anomaly summary."""
    if not AI_AGENT_READY:
        return ""

    try:
        context_parts = [
            "Discrepancy summary JSON:",
            f"```json\n{json.dumps(summary, indent=2, default=str)}\n```",
        ]
        if audit_report:
            context_parts.append("Prior forensic audit findings (for context on root causes):")
            context_parts.append(audit_report)

        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents="\n\n".join(context_parts),
            config=genai_types.GenerateContentConfig(
                system_instruction=DISPUTE_EMAIL_SYSTEM_PROMPT,
                temperature=0.4,
            ),
        )
        return response.text
    except Exception as e:
        return f"**AI Agent error:** could not draft the dispute email ({e})"


st.subheader("Next Steps")

if not anomaly_summary.get("has_anomalies"):
    st.caption("No discrepancies to dispute this cycle.")
elif not AI_AGENT_READY:
    st.caption("Connect the AI Audit Agent above to enable dispute email drafting.")
else:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        stage_email = st.button("Stage Dispute Email", type="primary", width="stretch")

    if stage_email:
        with st.spinner("Drafting dispute email..."):
            st.session_state.dispute_email_md = generate_dispute_email(
                anomaly_summary, st.session_state.audit_report_md
            )

    if st.session_state.dispute_email_md:
        st.write("")
        st.markdown(st.session_state.dispute_email_md)
        st.write("")
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.download_button(
                "Download Draft (.md)",
                data=st.session_state.dispute_email_md,
                file_name=f"dispute_draft_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                width="stretch",
            )


st.divider()
st.caption(
    "AI Agent insights are probabilistic. Review and validate all discrepancies "
    "before submitting external financial disputes."
)
