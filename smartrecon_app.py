"""
SmartRecon: AI-Agentic Affiliate Audit Dashboard
==================================================
A hybrid agentic financial reconciliation tool for retail content publishers.

Architecture:
  1. DETERMINISTIC TOOL LAYER  -> Pandas-based exact matching engine (fast, exact math)
  2. COGNITIVE AGENT LAYER     -> Gemini-powered Forensic Financial Auditor Agent
                                   (pattern recognition + root-cause reasoning)
  3. ACTION LAYER              -> Agentic dispute-email drafting tool

Deploy: Streamlit Community Cloud. Single file. No external DB. No persistence.

Required secrets (optional - app falls back to sidebar input):
  GEMINI_API_KEY

requirements.txt should contain:
  streamlit
  pandas
  google-genai

Note: this app uses Google's current `google-genai` SDK (the older
`google-generativeai` package was deprecated by Google and is no longer
maintained). Usage below follows the modern `genai.Client(...)` pattern.
"""

import os
import io
import json
from datetime import datetime

import pandas as pd
import streamlit as st

# google-genai is the current, actively-maintained Google SDK for the Gemini
# API (the older `google-generativeai` package was deprecated by Google in
# 2025 and is no longer receiving updates). Imported defensively so the app
# doesn't crash if the package is briefly unavailable - usage is guarded too.
try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_IMPORT_OK = True
except ImportError:
    GENAI_IMPORT_OK = False


# ============================================================================
# PAGE CONFIG  (must be the first Streamlit call)
# ============================================================================
st.set_page_config(
    page_title="SmartRecon | AI-Agentic Affiliate Audit",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="auto",
)


# ============================================================================
# LIGHTWEIGHT CUSTOM STYLING
# Native Streamlit layout (columns/containers) handles the responsiveness;
# this CSS only polishes spacing/typography and never fights the grid system.
# ============================================================================
st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        .smartrecon-subtitle {
            color: #5b6470;
            font-size: 1.05rem;
            margin-top: -0.6rem;
            margin-bottom: 1.4rem;
        }
        .section-divider {
            border-top: 1px solid rgba(120,120,120,0.25);
            margin: 1.6rem 0 1.4rem 0;
        }
        .guardrail-box {
            background-color: rgba(255, 193, 7, 0.12);
            border: 1px solid rgba(255, 193, 7, 0.45);
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            font-size: 0.92rem;
            margin-top: 2rem;
        }
        .agent-card {
            background-color: rgba(99, 102, 241, 0.06);
            border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# SESSION STATE INITIALIZATION
# Keeps results stable across reruns (e.g., when the dispute-email button
# is clicked) without recomputation of the deterministic match.
# ============================================================================
defaults = {
    "audit_report_md": None,
    "dispute_email_md": None,
    "recon_results": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ============================================================================
# HEADER / BRANDING
# ============================================================================
st.title("🧮 SmartRecon: AI-Agentic Affiliate Audit Dashboard")
st.markdown(
    '<p class="smartrecon-subtitle">Automating financial reconciliation by combining '
    "deterministic matching engines with cognitive AI Audit Agents.</p>",
    unsafe_allow_html=True,
)


# ============================================================================
# GEMINI API KEY RESOLUTION
# Order of precedence: st.secrets -> environment variable -> sidebar input.
# Nothing is logged or persisted to disk; the key only lives in session_state
# for the duration of the browser session.
# ============================================================================
def resolve_gemini_api_key() -> str | None:
    """Resolve the Gemini API key from secrets, env vars, or sidebar input."""
    # 1) Streamlit secrets (preferred for Community Cloud deployments)
    try:
        if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        # st.secrets raises if no secrets.toml exists at all - safe to ignore
        pass

    # 2) Environment variable fallback
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key

    # 3) Manual sidebar input fallback (session-only, not persisted)
    if "manual_gemini_key" not in st.session_state:
        st.session_state.manual_gemini_key = ""

    with st.sidebar:
        st.subheader("🔑 AI Agent Configuration")
        st.caption(
            "No Gemini API key was found in secrets or environment variables. "
            "Enter one below to enable the AI Audit Agent layer."
        )
        st.session_state.manual_gemini_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.manual_gemini_key,
            type="password",
            placeholder="AIza...",
            help="Get a key at https://aistudio.google.com/app/apikey",
        )
        st.caption(
            "🔒 Your key is used only for this session and is never stored or logged."
        )

    return st.session_state.manual_gemini_key or None


GEMINI_API_KEY = resolve_gemini_api_key()
AI_AGENT_READY = bool(GEMINI_API_KEY) and GENAI_IMPORT_OK

# The current SDK uses an explicit client object (no global `configure()` call).
GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY) if AI_AGENT_READY else None
GEMINI_MODEL_NAME = "gemini-3.5-flash"  # current-generation fast Gemini model


# ============================================================================
# SECTION 1: DATA INGESTION
# Two side-by-side uploaders via st.columns. On narrow/mobile viewports,
# Streamlit's column system automatically stacks these vertically.
# ============================================================================
st.header("1️⃣ Ingestion")
st.caption("Upload both files to run the deterministic + AI-agentic audit.")

upload_col1, upload_col2 = st.columns(2)

with upload_col1:
    st.markdown("**Upload Internal Ledger (CSV)**")
    st.caption("Expected columns: `transaction_id`, `click_timestamp`, `product_id`, `expected_commission_usd`")
    ledger_file = st.file_uploader(
        "Upload Internal Ledger (CSV)",
        type=["csv"],
        key="ledger_upload",
        label_visibility="collapsed",
    )

with upload_col2:
    st.markdown("**Upload Amazon Statement (CSV)**")
    st.caption("Expected columns: `amazon_order_id`, `payout_date`, `sku`, `actual_payout_usd`, `status`")
    amazon_file = st.file_uploader(
        "Upload Amazon Statement (CSV)",
        type=["csv"],
        key="amazon_upload",
        label_visibility="collapsed",
    )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Gate: do not run ANY calculation logic until both files are present.
files_ready = ledger_file is not None and amazon_file is not None

if not files_ready:
    st.info(
        "📂 Please upload both the Internal Ledger and Amazon Payout Statement "
        "to begin the AI-driven audit."
    )
    st.stop()  # Halts execution here - nothing below runs without both files.


# ============================================================================
# CSV LOADING + SCHEMA VALIDATION
# Defensive parsing so a malformed upload fails gracefully with a clear
# message rather than crashing the app mid-render.
# ============================================================================
LEDGER_REQUIRED_COLS = ["transaction_id", "click_timestamp", "product_id", "expected_commission_usd"]
AMAZON_REQUIRED_COLS = ["amazon_order_id", "payout_date", "sku", "actual_payout_usd", "status"]


def load_and_validate_csv(uploaded_file, required_cols: list[str], label: str) -> pd.DataFrame | None:
    """Read an uploaded CSV and verify it contains the required columns."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Could not parse '{label}' as CSV: {e}")
        return None

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(
            f"❌ '{label}' is missing required column(s): {', '.join(missing)}. "
            f"Expected: {', '.join(required_cols)}"
        )
        return None

    return df


ledger_df = load_and_validate_csv(ledger_file, LEDGER_REQUIRED_COLS, "Internal Ledger")
amazon_df = load_and_validate_csv(amazon_file, AMAZON_REQUIRED_COLS, "Amazon Statement")

if ledger_df is None or amazon_df is None:
    st.warning("Fix the file(s) above and re-upload to continue.")
    st.stop()


# ============================================================================
# SECTION 2 (PART A): DETERMINISTIC RECONCILIATION ENGINE
# Pure Pandas - no AI involved. This is the "fast, exact math" tool layer.
#
# Matching key:  ledger.transaction_id  <->  amazon.amazon_order_id
#
# Classification logic for matched rows:
#   - CLEAN MATCH:    abs(expected_commission_usd - actual_payout_usd) <= tolerance
#   - UNDERPAID:      actual_payout_usd < expected_commission_usd (beyond tolerance)
#   - OVERPAID:       actual_payout_usd > expected_commission_usd (beyond tolerance)
#   - MISSING_PAYOUT: ledger row has no matching Amazon order at all
#   - UNEXPECTED_PAYOUT: Amazon row has no matching ledger transaction at all
#
# Revenue leakage = sum of (expected - actual) for underpaid/missing rows,
# floored at 0 per-row so overpayments don't net against leakage.
# ============================================================================
TOLERANCE_USD = 0.01  # treat differences under 1 cent as floating point noise


def run_reconciliation(ledger: pd.DataFrame, amazon: pd.DataFrame) -> dict:
    """Run the deterministic match and return a structured results dict."""
    ledger = ledger.copy()
    amazon = amazon.copy()

    # Normalize join keys to string + stripped, so type/whitespace mismatches
    # (e.g. "1024" vs 1024.0, or trailing spaces) don't create false anomalies.
    ledger["transaction_id"] = ledger["transaction_id"].astype(str).str.strip()
    amazon["amazon_order_id"] = amazon["amazon_order_id"].astype(str).str.strip()

    # Outer join surfaces BOTH underpayment-type mismatches AND orphan rows
    # on either side (rows present in one file but absent in the other).
    merged = pd.merge(
        ledger,
        amazon,
        how="outer",
        left_on="transaction_id",
        right_on="amazon_order_id",
        indicator=True,
    )

    def classify(row) -> str:
        # Row only exists in the ledger -> Amazon never paid it at all.
        if row["_merge"] == "left_only":
            return "MISSING_PAYOUT"
        # Row only exists in the Amazon statement -> no internal click record.
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

    # Per-row leakage: only count money the publisher is OWED but didn't get.
    # Overpayments are not netted off (real leakage isn't reduced by them).
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

    total_rows = len(merged)
    total_leakage = float(merged["leakage_usd"].sum())

    return {
        "merged": merged,
        "clean_matches": clean_matches,
        "anomalies": anomalies,
        "total_rows": total_rows,
        "clean_count": len(clean_matches),
        "anomaly_count": len(anomalies),
        "total_leakage": total_leakage,
    }


recon = run_reconciliation(ledger_df, amazon_df)
st.session_state.recon_results = recon


# ============================================================================
# SECTION 2: EXECUTIVE SUMMARY KPIs
# st.columns(4) wraps into a 2x2 or 1x4 grid automatically on narrow screens.
# ============================================================================
st.header("2️⃣ Executive Summary")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.metric("Total Processed Rows", f"{recon['total_rows']:,}")

with kpi_col2:
    clean_pct = (recon["clean_count"] / recon["total_rows"] * 100) if recon["total_rows"] else 0
    st.metric("Clean Matches (Deterministic)", f"{recon['clean_count']:,}", f"{clean_pct:.1f}% of total")

with kpi_col3:
    anomaly_pct = (recon["anomaly_count"] / recon["total_rows"] * 100) if recon["total_rows"] else 0
    st.metric(
        "Anomalies Flagged (Sent to AI Agent)",
        f"{recon['anomaly_count']:,}",
        f"{anomaly_pct:.1f}% of total",
        delta_color="inverse",
    )

with kpi_col4:
    st.metric(
        "Estimated Revenue Leakage",
        f"${recon['total_leakage']:,.2f}",
        help="Sum of unpaid or underpaid differences across all flagged anomalies.",
    )

# Quick visual breakdown of anomaly types beneath the KPI row
if recon["anomaly_count"] > 0:
    with st.expander("📋 View anomaly breakdown table"):
        breakdown = (
            recon["anomalies"]["match_status"]
            .value_counts()
            .rename_axis("Anomaly Type")
            .reset_index(name="Count")
        )
        st.dataframe(breakdown, width='stretch', hide_index=True)
        st.dataframe(
            recon["anomalies"][
                [c for c in [
                    "transaction_id", "amazon_order_id", "product_id", "sku",
                    "expected_commission_usd", "actual_payout_usd", "status",
                    "match_status", "leakage_usd",
                ] if c in recon["anomalies"].columns]
            ],
            width='stretch',
            hide_index=True,
        )
else:
    st.success("🎉 No anomalies detected. All transactions reconcile cleanly.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ============================================================================
# SECTION 3 (PART A): ANOMALY SUMMARIZATION FOR THE AI AGENT
# CRITICAL: we never send 1,000+ raw rows to the LLM. Instead, Python
# pre-aggregates the anomalies into a compact structured JSON summary
# (counts, totals, day-of-week patterns, sample rows) that the agent
# reasons over. This keeps token usage low and focuses the model on
# patterns rather than row-by-row arithmetic it doesn't need to redo.
# ============================================================================
def build_anomaly_summary(recon: dict, max_samples_per_type: int = 5) -> dict:
    """Aggregate anomalies into a compact, LLM-friendly structured summary."""
    anomalies = recon["anomalies"].copy()

    if anomalies.empty:
        return {"has_anomalies": False}

    # Try to parse a date column (whichever side has it) for day-of-week patterns.
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
            group["_day_of_week"].dropna().value_counts().to_dict()
            if date_col else {}
        )
        date_label_counts = (
            group["_date_label"].dropna().value_counts().head(3).to_dict()
            if date_col else {}
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

    # Pull Amazon "status" field distribution if present (e.g. Returned/Cancelled)
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


# ============================================================================
# SECTION 3 (PART B): GEMINI FORENSIC AUDITOR AGENT
# ============================================================================
FORENSIC_AUDITOR_SYSTEM_PROMPT = """You are a Senior Forensic Financial Auditor Agent \
specializing in affiliate marketing revenue reconciliation for retail content publishers.

You will be given a structured JSON summary of payment discrepancies between an internal \
click-tracking ledger and an Amazon Associates payout statement. This summary has already \
been aggregated by a deterministic Pandas engine - you do NOT need to re-verify the math, \
only reason about WHY these patterns exist and WHAT the publisher should do.

Your job is to think like an experienced auditor who has seen hundreds of affiliate \
statements: spot root causes, surface systemic patterns (timing clusters, day-of-week \
effects, specific dates, SKU concentration), and produce actionable next steps.

Respond ONLY in Markdown using EXACTLY these three top-level headings, in this order:

## 🕵️‍♂️ Root-Cause Classification
Categorize the discrepancies into these buckets (use the ones that apply, skip ones that don't):
- **Returns/Cancellations** (customer returned the product or cancelled the order)
- **Policy Changes** (commission rate changes, program T&C updates, category exclusions)
- **Tracking Drops** (cookie/click attribution failures, missing clicks, lost referrals)
- **Other/Unclear** (anything that doesn't cleanly fit above)
For each bucket that applies, briefly explain the evidence from the data that supports it.

## 📈 Systemic Patterns Detected
Call out concrete, specific patterns - cite exact numbers, percentages, dates, days of the \
week, or SKUs/products wherever the data supports it (e.g. "70% of tracking drops occurred \
on Tuesday the 12th"). If the data doesn't support a strong pattern, say so honestly rather \
than inventing one.

## 🛠️ Recommended Action Items
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
        return f"⚠️ **AI Agent Error:** Could not complete analysis. Details: `{e}`"


# ============================================================================
# SECTION 3 (PART C): AI AGENT DEEP DIVE - UI
# ============================================================================
st.header("3️⃣ AI Agent Deep Dive")
st.caption(
    "The Forensic Auditor Agent analyzes a *summarized* version of the flagged "
    "anomalies (never raw row-by-row data) to reason about root causes and patterns."
)

if not anomaly_summary.get("has_anomalies"):
    st.success(
        "✅ No anomalies were flagged this cycle, so there is nothing for the AI "
        "Audit Agent to investigate. Your books reconcile cleanly."
    )
elif not AI_AGENT_READY:
    st.info(
        "🔌 The AI Audit Agent is not yet connected. Add a `GEMINI_API_KEY` to your "
        "Streamlit secrets, set it as an environment variable, or enter it in the "
        "sidebar to enable root-cause analysis."
    )
else:
    run_audit = st.button("🤖 Run Forensic AI Audit Agent", type="primary", width='stretch')

    if run_audit or st.session_state.audit_report_md is None:
        with st.spinner("🕵️‍♂️ The Forensic Auditor Agent is reviewing flagged anomalies..."):
            st.session_state.audit_report_md = run_forensic_audit_agent(anomaly_summary)

    if st.session_state.audit_report_md:
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        st.markdown(st.session_state.audit_report_md)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ============================================================================
# SECTION 4: AGENTIC NEXT STEPS - DISPUTE EMAIL DRAFTING ACTION LAYER
# This is a second, distinct agentic action: rather than just analyzing,
# the agent now PRODUCES a deliverable artifact (an email draft) grounded
# in the same structured anomaly summary used for the audit above.
# ============================================================================
st.header("4️⃣ Agentic Next Steps")
st.caption(
    "Generate a ready-to-send dispute email, pre-populated with the specific "
    "systemic errors identified by the AI Audit Agent above."
)

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

        user_payload = "\n\n".join(context_parts)
        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=user_payload,
            config=genai_types.GenerateContentConfig(
                system_instruction=DISPUTE_EMAIL_SYSTEM_PROMPT,
                temperature=0.4,
            ),
        )
        return response.text
    except Exception as e:
        return f"⚠️ **AI Agent Error:** Could not draft the dispute email. Details: `{e}`"


if not anomaly_summary.get("has_anomalies"):
    st.info("No discrepancies to dispute this cycle - nothing to draft.")
elif not AI_AGENT_READY:
    st.info("Connect the AI Audit Agent above to enable automated dispute email drafting.")
else:
    stage_email = st.button(
        "📝 Stage Dispute Email Templates",
        type="secondary",
        width='stretch',
    )

    if stage_email:
        with st.spinner("✍️ Drafting dispute email from the structured anomaly breakdown..."):
            st.session_state.dispute_email_md = generate_dispute_email(
                anomaly_summary, st.session_state.audit_report_md
            )

    if st.session_state.dispute_email_md:
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        st.markdown(st.session_state.dispute_email_md)
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download Email Draft (.md)",
            data=st.session_state.dispute_email_md,
            file_name=f"amazon_dispute_draft_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            width='stretch',
        )


# ============================================================================
# PM GUARDRAIL DISCLAIMER (FOOTER)
# Always visible at the very bottom of the single-page layout.
# ============================================================================
st.markdown(
    '<div class="guardrail-box">⚠️ <strong>Notice:</strong> AI Agent insights are '
    "probabilistic. Review and validate all discrepancies before submitting external "
    "financial disputes.</div>",
    unsafe_allow_html=True,
)
