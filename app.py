import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from datetime import datetime
import statsmodels.api as sm
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
# ROOT FIX: Never apply font-family with !important on the * selector.
# That overrides Streamlit's Material Icons font, causing icon names
# like "arrow_drop_down" to render as literal text in expanders.
# Scope Poppins ONLY to text-bearing HTML elements.
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

    /* Poppins on text nodes only — leave icon fonts untouched */
    html, body, p, h1, h2, h3, h4, h5, h6,
    label, button, .stMarkdown {{
    font-family: 'Inter', sans-serif !important;
    }}
    .material-icons {{
    font-family: 'Material Icons' !important;
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

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(90deg, #7c3aed, #ec4899) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }}
    .stButton > button:hover {{ opacity: 0.88; }}

    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: {exp} !important;
        border-radius: 10px !important;
        font-weight: 600;
    }}

    hr {{ border-color: {div} !important; }}
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
    """
    Accepts raw bytes so Streamlit can hash them reliably.
    Falls back to superstore.csv when nothing is uploaded.
    Returns a cleaned DataFrame with auto-derived date/sales columns.
    """
    try:
        if file_bytes is not None:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")
        else:
            df = pd.read_csv("superstore.csv", encoding="latin1")
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

    df.columns = df.columns.str.strip()

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    df.attrs["dups_removed"] = before - len(df)

    # Auto-detect date column
    DATE_CANDIDATES = [
        "Order Date", "Ship Date", "InvoiceDate", "invoice_date",
        "Date", "date", "Purchase Date", "Transaction Date",
        "order_date", "created_at", "timestamp", "OrderDate",
    ]
    date_col = next((c for c in DATE_CANDIDATES if c in df.columns), None)

    # Fallback: sniff object columns
    if date_col is None:
        for col in df.select_dtypes("object").columns:
            sample = df[col].dropna().head(20)
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            if parsed.notna().sum() > len(sample) * 0.7:
                date_col = col
                break

    df.attrs["date_col"] = date_col or "Not found"

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
        df = df.dropna(subset=[date_col])
        df["YearMonth"]    = df[date_col].dt.to_period("M").astype(str)
        df["Year"]         = df[date_col].dt.year.astype(int)
        df["Month"]        = df[date_col].dt.month.astype(int)
        df["Ordinal_Date"] = df[date_col].map(datetime.toordinal)

    # Auto-derive Sales = Qty × Price if Sales column is missing
    if "Sales" not in df.columns:
        qty   = next((c for c in df.columns if c.lower() in ["quantity", "qty"]), None)
        price = next((c for c in df.columns if c.lower() in ["unitprice", "price", "unit_price", "unit price"]), None)
        if qty and price:
            df["Sales"] = (
                pd.to_numeric(df[qty],   errors="coerce") *
                pd.to_numeric(df[price], errors="coerce")
            )

    return df


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
INTERNAL_COLS = {"YearMonth", "Year", "Month", "Ordinal_Date"}

with st.sidebar:
    st.markdown("### 🛠️ Control Panel")

    uploaded = st.file_uploader("Upload Dataset (CSV)", type="csv")
    df = load_and_clean_data(uploaded.read() if uploaded else None)

    st.divider()

    st.markdown("**🎨 Theme**")
    t1, t2 = st.columns(2)
    if t1.button("☀️ Light"):
        st.session_state.theme = "light"
        st.rerun()
    if t2.button("🌙 Dark"):
        st.session_state.theme = "dark"
        st.rerun()

    st.divider()

    view = st.radio("📂 Navigate Modules", [
        "Overview (Analytics)",
        "Sales Trends",
        "Correlation & EDA",
        "ML Predictions",
        "System Design",
    ])

    # Column selectors
    st.divider()
    st.markdown("**⚙️ Column Mapping**")

    value_col  = None
    profit_col = None
    group_col  = None
    cat2_col   = None

    if not df.empty:
        num_cols = [c for c in df.select_dtypes("number").columns if c not in INTERNAL_COLS]
        cat_cols = [c for c in df.select_dtypes("object").columns if c not in INTERNAL_COLS]

        SALES_HINTS  = {"sales","revenue","amount","total","income","price","unitprice","value","gmv","turnover"}
        PROFIT_HINTS = {"profit","net income","net_income","margin","earnings","gain"}
        GROUP_HINTS  = {"region","country","category","segment","product","department","brand","type","city","state"}
        CAT2_HINTS   = {"sub-category","sub_category","subcategory","category","type","product","department","group"}

        if num_cols:
            default_val = next((c for c in num_cols if c.lower() in SALES_HINTS), num_cols[0])
            value_col = st.selectbox("Value / Revenue Column", num_cols,
                                     index=num_cols.index(default_val))

        profit_opts  = ["(None)"] + num_cols
        default_prof = next((c for c in num_cols if c.lower() in PROFIT_HINTS), "(None)")
        profit_sel   = st.selectbox("Profit Column (optional)", profit_opts,
                                    index=profit_opts.index(default_prof))
        profit_col   = None if profit_sel == "(None)" else profit_sel

        if cat_cols:
            default_grp = next((c for c in cat_cols if c.lower() in GROUP_HINTS), cat_cols[0])
            group_col   = st.selectbox("Group-by Column", cat_cols,
                                       index=cat_cols.index(default_grp))

        remaining = [c for c in cat_cols if c != group_col]
        if remaining:
            default_c2 = next((c for c in remaining if c.lower() in CAT2_HINTS), remaining[0])
            cat2_col   = st.selectbox("Category Column (for charts)", remaining,
                                      index=remaining.index(default_c2))

    st.divider()
    if not df.empty:
        st.download_button(
            "⬇️ Download Cleaned Data",
            df.to_csv(index=False).encode("utf-8"),
            "cleaned_data.csv",
            "text/csv",
        )


# ─────────────────────────────────────────────────────────────────
# HEADER BANNER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%);
    text-align: center; padding: 2rem; border-radius: 20px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(124,58,237,0.3);
">
  <h1 style="color:white; margin:0; font-size:1.85rem; font-weight:800;">
    📊 Data-Driven Insight Generation using Python
  </h1>
  <p style="color:white; opacity:0.85; margin-top:0.4rem;">
    Data Engineering &nbsp;•&nbsp; Analytics &nbsp;•&nbsp; Machine Learning
  </p>
</div>
""", unsafe_allow_html=True)

# Guards
if df.empty:
    st.warning("⚠️ No data loaded. Upload a CSV or place 'superstore.csv' in your project folder.")
    st.stop()

if value_col is None:
    st.error("❌ No numeric columns found in this dataset. Please upload a valid CSV.")
    st.stop()


# ─────────────────────────────────────────────────────────────────
# MODULE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────
if view == "Overview (Analytics)":
    st.subheader("📈 Key Performance Indicators")

    total_val    = df[value_col].sum()
    total_profit = df[profit_col].sum() if profit_col else None
    margin       = (total_profit / total_val * 100) if (total_profit is not None and total_val) else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue",  f"${total_val:,.2f}")
    c2.metric("Net Profit",     f"${total_profit:,.2f}" if total_profit is not None else "N/A")
    c3.metric("Total Records",  f"{len(df):,}")
    c4.metric("Profit Margin",  f"{margin:.1f}%"        if margin      is not None else "N/A")

    st.divider()

    # Data Quality Report (no expander for the outer section — avoids icon glitch risk)
    st.markdown("#### 🔍 Data Quality Report")
    nulls     = int(df.isnull().sum().sum())
    dups      = df.attrs.get("dups_removed", 0)
    date_info = df.attrs.get("date_col", "Not found")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rows",               f"{len(df):,}")
    q2.metric("Columns",            f"{df.shape[1]}")
    q3.metric("Missing Values",     f"{nulls:,}")
    q4.metric("Duplicates Removed", f"{dups:,}")

    st.write(f"**Date column detected:** `{date_info}`")
    display_num = [c for c in df.select_dtypes("number").columns if c not in INTERNAL_COLS]
    st.write(f"**Numeric columns:** {', '.join(display_num)}")

    null_series = df.isnull().sum()
    null_series = null_series[null_series > 0]
    if not null_series.empty:
        null_df = null_series.reset_index()
        null_df.columns = ["Column", "Null Count"]
        fig_null = px.bar(
            null_df, x="Column", y="Null Count",
            title="Missing Values per Column",
            template=PLOTLY_TPL, color="Null Count",
            color_continuous_scale=["#f59e0b", "#ef4444"],
        )
        st.plotly_chart(fig_null, width='stretch')
    else:
        st.success("✅ No missing values found in the dataset!")

    with st.expander("📋 Raw Data Preview (first 50 rows)"):
        st.dataframe(df.head(50), width='stretch')

    st.divider()

    # Charts
    col_a, col_b = st.columns(2)
    with col_a:
        if group_col and cat2_col and group_col != cat2_col:
            fig = px.sunburst(df, path=[group_col, cat2_col], values=value_col,
                              title=f"{value_col} — {group_col} → {cat2_col}",
                              template=PLOTLY_TPL)
        elif group_col:
            grp = df.groupby(group_col)[value_col].sum().reset_index()
            fig = px.pie(grp, names=group_col, values=value_col,
                         title=f"{value_col} by {group_col}", template=PLOTLY_TPL)
        else:
            fig = None
        if fig:
            st.plotly_chart(fig, width='stretch')

    with col_b:
        if profit_col and cat2_col:
            fig2 = px.box(df, x=cat2_col, y=profit_col, color=cat2_col,
                          title=f"Profit Distribution by {cat2_col}", template=PLOTLY_TPL)
        elif group_col:
            bar = df.groupby(group_col)[value_col].mean().reset_index()
            fig2 = px.bar(bar, x=group_col, y=value_col,
                          title=f"Average {value_col} by {group_col}",
                          template=PLOTLY_TPL, color=group_col)
        else:
            fig2 = None
        if fig2:
            st.plotly_chart(fig2, width='stretch')

    if group_col:
        st.markdown(f"#### 🏆 Top 10 by {value_col}")
        top10 = (df.groupby(group_col)[value_col]
                   .sum().sort_values(ascending=False)
                   .head(10).reset_index())
        top10[value_col] = top10[value_col].map(lambda x: f"${x:,.2f}")
        st.dataframe(top10, width='stretch')


# ─────────────────────────────────────────────────────────────────
# MODULE 2 — SALES TRENDS
# ─────────────────────────────────────────────────────────────────
elif view == "Sales Trends":
    st.subheader("📅 Temporal Analysis")

    if "YearMonth" not in df.columns:
        st.warning("⚠️ No date column detected. Temporal analysis is unavailable for this dataset.")
        st.stop()

    trend = df.groupby("YearMonth")[value_col].sum().reset_index()
    fig_t = px.line(trend, x="YearMonth", y=value_col, markers=True,
                    title=f"Monthly {value_col} Trend", template=PLOTLY_TPL)
    fig_t.update_traces(line_color="#7c3aed", marker_color="#ec4899")
    st.plotly_chart(fig_t, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        yoy = df.groupby("Year")[value_col].sum().reset_index()
        fig_y = px.bar(yoy, x="Year", y=value_col,
                       title=f"Year-over-Year {value_col}",
                       template=PLOTLY_TPL, color=value_col,
                       color_continuous_scale=GRAD)
        st.plotly_chart(fig_y, width='stretch')

    with col2:
        MONTHS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                  7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        sea = df.groupby("Month")[value_col].mean().reset_index()
        sea["Month_Name"] = sea["Month"].map(MONTHS)
        fig_s = px.bar(sea, x="Month_Name", y=value_col,
                       title=f"Average {value_col} by Month (Seasonality)",
                       template=PLOTLY_TPL, color=value_col,
                       color_continuous_scale=GRAD)
        st.plotly_chart(fig_s, width='stretch')

    if group_col:
        grp_trend = df.groupby(["YearMonth", group_col])[value_col].sum().reset_index()
        fig_g = px.line(grp_trend, x="YearMonth", y=value_col, color=group_col,
                        title=f"Monthly {value_col} by {group_col}",
                        template=PLOTLY_TPL, markers=True)
        st.plotly_chart(fig_g, width='stretch')


# ─────────────────────────────────────────────────────────────────
# MODULE 3 — CORRELATION & EDA
# ─────────────────────────────────────────────────────────────────
elif view == "Correlation & EDA":
    st.subheader("🔬 Exploratory Data Analysis")

    num_df = df.select_dtypes("number").drop(
        columns=[c for c in INTERNAL_COLS if c in df.columns], errors="ignore"
    )

    if num_df.shape[1] >= 2:
        fig_c = px.imshow(
            num_df.corr(), text_auto=".2f",
            title="Feature Correlation Heatmap",
            template=PLOTLY_TPL,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto",
        )
        st.plotly_chart(fig_c, width='stretch')
    else:
        st.warning("Need at least 2 numeric columns for a correlation heatmap.")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        hist_col = st.selectbox("Distribution of", num_df.columns.tolist())
        fig_h = px.histogram(df, x=hist_col, nbins=40,
                             title=f"Distribution of {hist_col}",
                             template=PLOTLY_TPL,
                             color_discrete_sequence=["#7c3aed"],
                             marginal="box")
        st.plotly_chart(fig_h, width='stretch')

    with col2:
        nc  = num_df.columns.tolist()
        x_c = st.selectbox("Scatter X", nc, index=0)
        y_c = st.selectbox("Scatter Y", nc, index=min(1, len(nc)-1))
        fig_sc = px.scatter(df, x=x_c, y=y_c,
                            color=group_col if group_col else None,
                            title=f"{x_c} vs {y_c}",
                            template=PLOTLY_TPL,
                            trendline="ols", opacity=0.6)
        st.plotly_chart(fig_sc, width='stretch')

    st.markdown("#### 📊 Descriptive Statistics")
    st.dataframe(num_df.describe().T.style.background_gradient(cmap="Purples"),
                 width='stretch')


# ─────────────────────────────────────────────────────────────────
# MODULE 4 — ML PREDICTIONS
# ─────────────────────────────────────────────────────────────────
elif view == "ML Predictions":
    st.subheader("🤖 Monthly Revenue Forecasting")

    if "YearMonth" not in df.columns:
        st.warning("⚠️ No date column detected — cannot build a time-series forecast.")
        st.stop()

    ml_df = df.groupby("YearMonth")[value_col].sum().reset_index()
    ml_df["Month_Index"] = range(len(ml_df))

    if len(ml_df) < 8:
        st.warning(f"⚠️ Only {len(ml_df)} months of data. Need at least 8 for a meaningful split.")
        st.stop()

    split = int(len(ml_df) * 0.8)
    train = ml_df.iloc[:split]
    test  = ml_df.iloc[split:]

    model      = LinearRegression().fit(train[["Month_Index"]], train[value_col])
    train_pred = model.predict(train[["Month_Index"]])
    test_pred  = model.predict(test[["Month_Index"]])

    r2_tr  = r2_score(train[value_col], train_pred)
    r2_te  = r2_score(test[value_col],  test_pred)
    mae    = mean_absolute_error(test[value_col], test_pred)

    N         = 6
    fut_idx   = np.arange(len(ml_df), len(ml_df) + N).reshape(-1, 1)
    fut_pred  = model.predict(fut_idx)
    last_dt   = pd.to_datetime(ml_df["YearMonth"].iloc[-1])
    fut_dates = [(last_dt + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(1, N+1)]

    m1, m2, m3 = st.columns(3)
    m1.metric("Train R²",    f"{r2_tr:.4f}")
    m2.metric("Hold-out R²", f"{r2_te:.4f}")
    m3.metric("Test MAE",    f"${mae:,.2f}")

    st.info(
        "ℹ️ **Hold-out R²** is evaluated on the last 20% of months (unseen by the model). "
        "For datasets with strong seasonality, ARIMA or Facebook Prophet will perform better."
    )

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=train["YearMonth"], y=train[value_col],
                                name="Training Data", mode="lines+markers",
                                line=dict(color="#7c3aed", width=2)))
    fig_fc.add_trace(go.Scatter(x=test["YearMonth"], y=test[value_col],
                                name="Actual (Test)", mode="lines+markers",
                                line=dict(color="#10b981", width=2)))
    fig_fc.add_trace(go.Scatter(x=test["YearMonth"], y=test_pred,
                                name="Predicted (Test)", mode="lines+markers",
                                line=dict(color="#f59e0b", dash="dot", width=2)))
    fig_fc.add_trace(go.Scatter(x=fut_dates, y=fut_pred,
                                name="6-Month Forecast", mode="lines+markers",
                                line=dict(color="#ef4444", dash="dash", width=2)))
    fig_fc.update_layout(
        title=f"6-Month {value_col} Forecast (Train / Test Evaluation)",
        xaxis_title="Month", yaxis_title=value_col,
        template=PLOTLY_TPL,
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig_fc, width='stretch')

    st.markdown("#### 📋 Forecast Values")
    st.dataframe(
        pd.DataFrame({"Month": fut_dates,
                      f"Forecasted {value_col}": [f"${v:,.2f}" for v in fut_pred]}),
        width='stretch',
    )

    residuals = test[value_col].values - test_pred
    fig_r = px.bar(
        x=test["YearMonth"], y=residuals,
        title="Residuals on Test Set (Actual − Predicted)",
        labels={"x": "Month", "y": "Residual ($)"},
        template=PLOTLY_TPL, color=residuals,
        color_continuous_scale=["#ef4444", "#ffffff", "#10b981"],
    )
    st.plotly_chart(fig_r, width='stretch')


# ─────────────────────────────────────────────────────────────────
# MODULE 5 — SYSTEM DESIGN
# ─────────────────────────────────────────────────────────────────
elif view == "System Design":
    st.subheader("🏗️ System Architecture")
    st.markdown("""

Architecture Flow

CSV Upload / Local File
      │
      ▼
load_and_clean_data()
├─ Strip column whitespace
├─ Drop duplicates
├─ Auto-detect date column (12+ candidate names + sniff fallback)
├─ Parse dates → YearMonth, Year, Month, Ordinal_Date
└─ Auto-derive Sales = Qty × Price (if Sales column absent)
      │
      ▼
Sidebar Column Mapper
(auto-selects Value / Profit / Group columns, user can override)
      │
      ├──▶ Overview        → KPIs + Data Quality + Sunburst/Pie + Box/Bar + Top-10
      ├──▶ Sales Trends    → Monthly Line + YoY Bar + Seasonality + Group Breakdown
      ├──▶ Correlation EDA → Heatmap + Histogram + Scatter + Descriptive Stats
      └──▶ ML Predictions  → Train/Test Forecast + Residual Chart + Forecast Table

    """)


# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;opacity:0.4;font-size:0.8rem;'>"
    "Data-Driven Insight Generation &nbsp;•&nbsp; Built with Streamlit & Plotly</p>",
    unsafe_allow_html=True,
)
