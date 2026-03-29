import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data-Driven Insight Generation",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────
# THEME STATE
# ─────────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
def apply_theme(theme):
    bg   = "#111827" if theme == "dark" else "#f8f9fa"
    card = "#1f2937" if theme == "dark" else "#ffffff"
    text = "#f9fafb" if theme == "dark" else "#111827"
    div  = "#374151" if theme == "dark" else "#e5e7eb"
    exp  = "#1e293b" if theme == "dark" else "#f1f5f9"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    /* Scoped Inter font to avoid breaking Material Icons */
    .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, button {{
        font-family: 'Inter', sans-serif !important;
    }}

    .stApp {{ background-color: {bg}; color: {text}; }}

    [data-testid="stSidebar"] {{
        background-color: {card};
        border-right: 1px solid {div};
    }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%) !important;
        border-radius: 14px;
        padding: 18px 22px !important;
    }}
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {{
        color: #ffffff !important;
    }}
    [data-testid="stMetricValue"] {{
        font-weight: 800;
        font-size: 1.5rem !important;
    }}

    .stButton > button {{
        background: linear-gradient(90deg, #7c3aed, #ec4899) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }}

    .streamlit-expanderHeader {{
        background-color: {exp} !important;
        border-radius: 10px !important;
        font-weight: 600;
    }}
    </style>
    """, unsafe_allow_html=True)

apply_theme(st.session_state.theme)
PLOTLY_TPL = "plotly_dark" if st.session_state.theme == "dark" else "plotly_white"
GRAD       = ["#7c3aed", "#ec4899"]

# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🔄 Loading & cleaning data…")
def load_and_clean_data(file_bytes=None):
    try:
        if file_bytes is not None:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")
        else:
            df = pd.read_csv("superstore.csv", encoding="latin1")
    except Exception:
        return pd.DataFrame()

    df.columns = df.columns.str.strip()
    before = len(df)
    df = df.drop_duplicates()
    df.attrs["dups_removed"] = before - len(df)

    DATE_CANDIDATES = ["Order Date", "Ship Date", "Date", "date", "Timestamp"]
    date_col = next((c for c in DATE_CANDIDATES if c in df.columns), None)

    if date_col is None:
        for col in df.select_dtypes("object").columns:
            sample = df[col].dropna().head(20)
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() > len(sample) * 0.7:
                date_col = col
                break

    df.attrs["date_col"] = date_col or "Not found"

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df["YearMonth"]    = df[date_col].dt.to_period("M").astype(str)
        df["Year"]         = df[date_col].dt.year
        df["Month"]        = df[date_col].dt.month
        df["Ordinal_Date"] = df[date_col].apply(lambda x: x.toordinal())

    return df

# ─────────────────────────────────────────────────────────────────
# APP LOGIC
# ─────────────────────────────────────────────────────────────────
INTERNAL_COLS = {"YearMonth", "Year", "Month", "Ordinal_Date"}

with st.sidebar:
    st.markdown("### 🛠️ Control Panel")
    uploaded = st.file_uploader("Upload Dataset (CSV)", type="csv")
    df = load_and_clean_data(uploaded.read() if uploaded else None)

    if not df.empty:
        st.divider()
        view = st.radio("📂 Navigate Modules", ["Overview (Analytics)", "Sales Trends", "Correlation & EDA", "ML Predictions", "System Design"])
        
        num_cols = [c for c in df.select_dtypes("number").columns if c not in INTERNAL_COLS]
        cat_cols = [c for c in df.select_dtypes("object").columns if c not in INTERNAL_COLS]

        value_col = st.selectbox("Value Column", num_cols) if num_cols else None
        group_col = st.selectbox("Group-by Column", cat_cols) if cat_cols else None
        cat2_col  = st.selectbox("Sub-Category", [c for c in cat_cols if c != group_col]) if len(cat_cols) > 1 else None
        profit_col = st.selectbox("Profit Column (optional)", ["(None)"] + num_cols)
        profit_col = None if profit_col == "(None)" else profit_col

# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown('<div style="background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%); text-align: center; padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;"><h1 style="color:white; margin:0;">📊 Data Insight Generator</h1></div>', unsafe_allow_html=True)

if df.empty:
    st.warning("Please upload a CSV file to begin.")
    st.stop()

# ─────────────────────────────────────────────────────────────────
# MODULES (Example: Overview)
# ─────────────────────────────────────────────────────────────────
if view == "Overview (Analytics)":
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${df[value_col].sum():,.2f}")
    c2.metric("Total Records", f"{len(df):,}")
    c3.metric("Missing Values", f"{df.isnull().sum().sum():,}")

    st.divider()
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.pie(df, names=group_col, values=value_col, title=f"Revenue by {group_col}", template=PLOTLY_TPL)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        if profit_col:
            fig2 = px.box(df, x=group_col, y=profit_col, title="Profit Distribution", template=PLOTLY_TPL)
            st.plotly_chart(fig2, use_container_width=True)

elif view == "ML Predictions":
    st.subheader("🤖 Revenue Forecasting")
    if "YearMonth" in df.columns:
        ml_df = df.groupby("YearMonth")[value_col].sum().reset_index()
        ml_df["Month_Index"] = np.arange(len(ml_df))
        
        X = ml_df[["Month_Index"]]
        y = ml_df[value_col]
        
        model = LinearRegression().fit(X, y)
        ml_df["Pred"] = model.predict(X)
        
        fig_ml = px.line(ml_df, x="YearMonth", y=[value_col, "Pred"], title="Trend Forecast", template=PLOTLY_TPL)
        st.plotly_chart(fig_ml, use_container_width=True)
    else:
        st.error("Date column required for forecasting.")

elif view == "System Design":
    st.info("System architecture and bug fix logs are displayed here.")
    st.json({"Team": "4 Members", "Tech": "Streamlit, Plotly, Scikit-Learn"})

# Footer
st.divider()
st.caption("Built with ❤️ by the Data Team")
