import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Data-Driven Insight Generation using Python",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────
# THEME STATE
# ─────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
def apply_theme(theme):
    bg_main   = "#f8f9fa" if theme == "light" else "#111827"
    card_bg   = "#ffffff" if theme == "light" else "#1f2937"
    text      = "#111827" if theme == "light" else "#f9fafb"
    sub_text  = "#6b7280" if theme == "light" else "#9ca3af"

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;800&display=swap');

        * {{ font-family: 'Poppins', sans-serif !important; }}

        .stApp {{
            background-color: {bg_main};
            color: {text};
        }}

        /* Metric cards */
        [data-testid="stMetric"] {{
            background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%) !important;
            border-radius: 14px;
            padding: 18px 22px;
        }}
        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"],
        [data-testid="stMetricDelta"] {{
            color: white !important;
        }}
        [data-testid="stMetricValue"] {{ font-weight: 800; font-size: 1.6rem; }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {card_bg};
            border-right: 1px solid {"#e5e7eb" if theme == "light" else "#374151"};
        }}

        /* Buttons */
        .stButton > button {{
            background: linear-gradient(90deg, #7c3aed, #ec4899);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 8px 18px;
        }}
        .stButton > button:hover {{
            opacity: 0.9;
            transform: translateY(-1px);
        }}

        /* Expander */
        .streamlit-expanderHeader {{
            background-color: {card_bg};
            border-radius: 10px;
        }}

        /* Subheaders */
        h2, h3 {{ color: {text}; }}

        /* Data quality badge */
        .quality-good  {{ color: #10b981; font-weight: 700; }}
        .quality-warn  {{ color: #f59e0b; font-weight: 700; }}
        .quality-bad   {{ color: #ef4444; font-weight: 700; }}

        /* Divider */
        hr {{ border-color: {"#e5e7eb" if theme == "light" else "#374151"}; }}
    </style>
    """, unsafe_allow_html=True)

apply_theme(st.session_state.theme)
plotly_template = "plotly_dark" if st.session_state.theme == "dark" else "plotly_white"

# ─────────────────────────────────────────────
# HEADER BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%);
    text-align: center;
    padding: 2.5rem 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(124,58,237,0.25);
">
  <h1 style="color:white; margin:0; font-size:2rem; font-weight:800;">
    📊 Data-Driven Insight Generation using Python
  </h1>
  <p style="color:white; opacity:0.9; margin-top:0.5rem; font-size:1rem;">
    Data Engineering &nbsp;•&nbsp; Analytics &nbsp;•&nbsp; Machine Learning
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING  (FIX: bytes-based caching)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="🔄 Loading and cleaning data…")
def load_and_clean_data(file_bytes=None, filename=""):
    """
    Accepts raw bytes so Streamlit can hash them reliably.
    Falls back to the local superstore.csv if nothing is uploaded.
    """
    try:
        if file_bytes is not None:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")
        else:
            df = pd.read_csv("superstore.csv", encoding="latin1")
    except FileNotFoundError:
        st.error("⚠️ No file uploaded and 'superstore.csv' not found in the project folder.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

    # ── Standardize column names ──────────────────────────
    df.columns = df.columns.str.strip()

    # ── Remove exact duplicates ───────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before - len(df)

    # ── Parse date columns if they exist ─────────────────
    for col in ["Order Date", "Ship Date", "InvoiceDate", "Date", "date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ── Build derived date columns from the first date col found ──
    date_col = next(
        (c for c in ["Order Date", "InvoiceDate", "Date", "date", "Ship Date"] if c in df.columns),
        None,
    )
    if date_col:
        df = df.dropna(subset=[date_col])
        df["_DateCol"]    = df[date_col]
        df["YearMonth"]   = df[date_col].dt.to_period("M").astype(str)
        df["Year"]        = df[date_col].dt.year
        df["Month"]       = df[date_col].dt.month
        df["Ordinal_Date"] = df[date_col].map(datetime.toordinal)

    # Stash metadata for the data-quality panel
    df.attrs["duplicates_removed"] = duplicates_removed
    df.attrs["date_col"]           = date_col or "None"

    return df


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛠️ Control Panel")

    # ── File upload ───────────────────────────
    uploaded = st.file_uploader("Upload Dataset (CSV)", type="csv")

    # ── Load data ────────────────────────────
    if uploaded:
        file_bytes = uploaded.read()
        df = load_and_clean_data(file_bytes, filename=uploaded.name)
    else:
        df = load_and_clean_data()

    st.divider()

    # ── Theme toggle ─────────────────────────
    st.markdown("**🎨 Theme**")
    col1, col2 = st.columns(2)
    if col1.button("☀️ Light"):
        st.session_state.theme = "light"
        st.rerun()
    if col2.button("🌙 Dark"):
        st.session_state.theme = "dark"
        st.rerun()

    st.divider()

    # ── Navigation ───────────────────────────
    view = st.radio("📂 Navigate Modules", [
        "Overview (Analytics)",
        "Sales Trends",
        "Correlation & EDA",
        "ML Predictions",
        "System Design",
    ])

    st.divider()

    # ── Dynamic column selectors ─────────────
    st.markdown("**⚙️ Column Mapping**")
    if not df.empty:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols     = df.select_dtypes(include="object").columns.tolist()

        # Value / target column
        default_value = next(
            (c for c in ["Sales", "Revenue", "Amount", "Total", "UnitPrice"] if c in numeric_cols),
            numeric_cols[0] if numeric_cols else None,
        )
        value_col = st.selectbox(
            "Value / Revenue Column",
            numeric_cols,
            index=numeric_cols.index(default_value) if default_value else 0,
        )

        # Profit column (optional)
        profit_options = ["(None)"] + numeric_cols
        default_profit = next(
            (c for c in ["Profit", "profit", "Net Income"] if c in numeric_cols),
            "(None)",
        )
        profit_col_sel = st.selectbox(
            "Profit Column (optional)",
            profit_options,
            index=profit_options.index(default_profit),
        )
        profit_col = None if profit_col_sel == "(None)" else profit_col_sel

        # Group-by column
        default_group = next(
            (c for c in ["Region", "Country", "Category", "Segment", "Product"] if c in cat_cols),
            cat_cols[0] if cat_cols else None,
        )
        group_col = st.selectbox(
            "Group-by Column",
            cat_cols,
            index=cat_cols.index(default_group) if default_group else 0,
        ) if cat_cols else None

        # Second category column (for sunburst / box plots)
        default_cat2 = next(
            (c for c in ["Category", "Sub-Category", "Department", "Type"] if c in cat_cols),
            cat_cols[1] if len(cat_cols) > 1 else cat_cols[0] if cat_cols else None,
        )
        cat2_col = st.selectbox(
            "Category Column",
            cat_cols,
            index=cat_cols.index(default_cat2) if default_cat2 and default_cat2 in cat_cols else 0,
        ) if cat_cols else None

    st.divider()

    # ── Download cleaned data ─────────────────
    if not df.empty:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Cleaned Data",
            data=csv_bytes,
            file_name="cleaned_data.csv",
            mime="text/csv",
        )

# ─────────────────────────────────────────────
# GUARD: empty dataframe
# ─────────────────────────────────────────────
if df.empty:
    st.warning("⚠️ No data loaded. Upload a CSV file or add 'superstore.csv' to the project folder.")
    st.stop()

# ─────────────────────────────────────────────
# MODULE 1 — OVERVIEW (Analytics)
# ─────────────────────────────────────────────
if view == "Overview (Analytics)":
    st.subheader("📈 Key Performance Indicators")

    total_value  = df[value_col].sum()
    total_orders = df.shape[0]
    avg_order    = df[value_col].mean()
    total_profit = df[profit_col].sum() if profit_col else 0
    margin_pct   = (total_profit / total_value * 100) if (profit_col and total_value) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue",  f"${total_value:,.2f}")
    c2.metric("Net Profit",     f"${total_profit:,.2f}" if profit_col else "N/A")
    c3.metric("Total Records",  f"{total_orders:,}")
    c4.metric("Profit Margin",  f"{margin_pct:.1f}%" if profit_col else "N/A")

    st.divider()

    # ── Data Quality Report ───────────────────
    with st.expander("🔍 Data Quality Report", expanded=False):
        nulls      = df.isnull().sum().sum()
        dups_rmvd  = df.attrs.get("duplicates_removed", 0)
        date_found = df.attrs.get("date_col", "None")

        qcol1, qcol2, qcol3, qcol4 = st.columns(4)
        qcol1.metric("Rows",              f"{df.shape[0]:,}")
        qcol2.metric("Columns",           f"{df.shape[1]}")
        qcol3.metric("Missing Values",    f"{nulls:,}")
        qcol4.metric("Duplicates Removed",f"{dups_rmvd:,}")

        st.write(f"**Date column detected:** `{date_found}`")
        st.write(f"**Numeric columns:** {', '.join(df.select_dtypes('number').columns.tolist())}")

        with st.expander("📋 Raw Data Preview (first 50 rows)"):
            st.dataframe(df.head(50), use_container_width=True)

        # Null heatmap per column
        null_df = df.isnull().sum().reset_index()
        null_df.columns = ["Column", "Null Count"]
        null_df = null_df[null_df["Null Count"] > 0]
        if not null_df.empty:
            fig_null = px.bar(
                null_df, x="Column", y="Null Count",
                title="Missing Values per Column",
                template=plotly_template,
                color="Null Count",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig_null, use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")

    st.divider()

    # ── Charts Row ────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        if group_col and cat2_col and group_col != cat2_col:
            fig = px.sunburst(
                df,
                path=[group_col, cat2_col],
                values=value_col,
                title=f"{value_col} by {group_col} → {cat2_col}",
                template=plotly_template,
            )
        elif group_col:
            grp_df = df.groupby(group_col)[value_col].sum().reset_index()
            fig = px.pie(
                grp_df, names=group_col, values=value_col,
                title=f"{value_col} Distribution by {group_col}",
                template=plotly_template,
            )
        else:
            fig = None
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if profit_col and cat2_col:
            fig2 = px.box(
                df, x=cat2_col, y=profit_col, color=cat2_col,
                title=f"Profit Distribution by {cat2_col}",
                template=plotly_template,
            )
            st.plotly_chart(fig2, use_container_width=True)
        elif group_col:
            bar_df = df.groupby(group_col)[value_col].mean().reset_index()
            fig2 = px.bar(
                bar_df, x=group_col, y=value_col,
                title=f"Average {value_col} by {group_col}",
                template=plotly_template,
                color=group_col,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Top 10 Table ──────────────────────────
    if group_col:
        st.markdown(f"#### 🏆 Top 10 by {value_col}")
        top10 = (
            df.groupby(group_col)[value_col]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top10[value_col] = top10[value_col].map(lambda x: f"${x:,.2f}")
        st.dataframe(top10, use_container_width=True)

# ─────────────────────────────────────────────
# MODULE 2 — SALES TRENDS
# ─────────────────────────────────────────────
elif view == "Sales Trends":
    st.subheader("📅 Temporal Analysis")

    if "YearMonth" not in df.columns:
        st.warning("⚠️ No date column detected in this dataset. Cannot show time-series trends.")
        st.stop()

    # ── Monthly trend ─────────────────────────
    trend_df = df.groupby("YearMonth")[value_col].sum().reset_index()
    fig_trend = px.line(
        trend_df, x="YearMonth", y=value_col,
        markers=True,
        title=f"Monthly {value_col} Trend",
        template=plotly_template,
    )
    fig_trend.update_traces(line_color="#7c3aed", marker_color="#ec4899")
    st.plotly_chart(fig_trend, use_container_width=True)

    col1, col2 = st.columns(2)

    # ── Year-over-year bar ─────────────────────
    with col1:
        if "Year" in df.columns:
            yoy_df = df.groupby("Year")[value_col].sum().reset_index()
            fig_yoy = px.bar(
                yoy_df, x="Year", y=value_col,
                title=f"Year-over-Year {value_col}",
                template=plotly_template,
                color=value_col,
                color_continuous_scale=["#7c3aed", "#ec4899"],
            )
            st.plotly_chart(fig_yoy, use_container_width=True)

    # ── Monthly seasonality ────────────────────
    with col2:
        if "Month" in df.columns:
            month_names = {
                1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec",
            }
            season_df = df.groupby("Month")[value_col].mean().reset_index()
            season_df["Month_Name"] = season_df["Month"].map(month_names)
            fig_season = px.bar(
                season_df, x="Month_Name", y=value_col,
                title=f"Average {value_col} by Month (Seasonality)",
                template=plotly_template,
                color=value_col,
                color_continuous_scale=["#7c3aed", "#ec4899"],
            )
            st.plotly_chart(fig_season, use_container_width=True)

    # ── Group trend breakdown ──────────────────
    if group_col and "YearMonth" in df.columns:
        st.markdown(f"#### {value_col} Trend by {group_col}")
        grp_trend = df.groupby(["YearMonth", group_col])[value_col].sum().reset_index()
        fig_grp = px.line(
            grp_trend, x="YearMonth", y=value_col, color=group_col,
            title=f"Monthly {value_col} by {group_col}",
            template=plotly_template, markers=True,
        )
        st.plotly_chart(fig_grp, use_container_width=True)

# ─────────────────────────────────────────────
# MODULE 3 — CORRELATION & EDA
# ─────────────────────────────────────────────
elif view == "Correlation & EDA":
    st.subheader("🔬 Exploratory Data Analysis")

    num_df = df.select_dtypes(include="number")

    if num_df.shape[1] < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
    else:
        # ── Correlation heatmap ─────────────────
        corr = num_df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            title="Feature Correlation Heatmap",
            template=plotly_template,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.divider()

        col1, col2 = st.columns(2)

        # ── Distribution histogram ──────────────
        with col1:
            hist_col = st.selectbox("Select column for distribution", num_df.columns.tolist())
            fig_hist = px.histogram(
                df, x=hist_col, nbins=40,
                title=f"Distribution of {hist_col}",
                template=plotly_template,
                color_discrete_sequence=["#7c3aed"],
                marginal="box",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Scatter plot ────────────────────────
        with col2:
            num_cols_list = num_df.columns.tolist()
            x_col = st.selectbox("Scatter X axis", num_cols_list, index=0)
            y_col = st.selectbox(
                "Scatter Y axis",
                num_cols_list,
                index=min(1, len(num_cols_list) - 1),
            )
            scatter_color = group_col if group_col else None
            fig_scatter = px.scatter(
                df, x=x_col, y=y_col,
                color=scatter_color,
                title=f"{x_col} vs {y_col}",
                template=plotly_template,
                trendline="ols",
                opacity=0.6,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Descriptive statistics ──────────────
        st.markdown("#### 📊 Descriptive Statistics")
        st.dataframe(num_df.describe().T.style.background_gradient(cmap="Purples"), use_container_width=True)

# ─────────────────────────────────────────────
# MODULE 4 — ML PREDICTIONS
# ─────────────────────────────────────────────
elif view == "ML Predictions":
    st.subheader("🤖 Monthly Revenue Forecasting")

    if "YearMonth" not in df.columns:
        st.warning("⚠️ No date column detected — cannot build a time-series forecast.")
        st.stop()

    # ── Aggregate by month ────────────────────
    ml_df = df.groupby("YearMonth")[value_col].sum().reset_index()
    ml_df["Month_Index"] = range(len(ml_df))

    if len(ml_df) < 8:
        st.warning("⚠️ Need at least 8 months of data for a meaningful train/test split.")
        st.stop()

    # ── FIX: Proper train/test split ─────────
    # Hold out last 20% of time periods as the test set
    split_idx  = int(len(ml_df) * 0.8)
    train_df   = ml_df.iloc[:split_idx]
    test_df    = ml_df.iloc[split_idx:]

    X_train = train_df[["Month_Index"]].values
    y_train = train_df[value_col].values
    X_test  = test_df[["Month_Index"]].values
    y_test  = test_df[value_col].values

    model = LinearRegression().fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    r2_train = r2_score(y_train, train_preds)
    r2_test  = r2_score(y_test, test_preds)
    mae      = mean_absolute_error(y_test, test_preds)

    # ── Forecast next 6 months ────────────────
    n_future     = 6
    future_idx   = np.array(range(len(ml_df), len(ml_df) + n_future)).reshape(-1, 1)
    future_preds = model.predict(future_idx)

    last_date    = pd.to_datetime(ml_df["YearMonth"].iloc[-1])
    future_dates = [
        (last_date + pd.DateOffset(months=i)).strftime("%Y-%m")
        for i in range(1, n_future + 1)
    ]

    # ── Model metrics ─────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Training R²",        f"{r2_train:.4f}")
    m2.metric("Hold-out R² (test)", f"{r2_test:.4f}")
    m3.metric("Test MAE",           f"${mae:,.2f}")

    st.info(
        "ℹ️ **Training R²** measures fit on seen data. "
        "**Hold-out R²** is the honest accuracy on unseen months. "
        "For a real deployment, consider ARIMA or Prophet for better seasonality modeling."
    )

    # ── Forecast chart ────────────────────────
    fig_fc = go.Figure()

    # Historical
    fig_fc.add_trace(go.Scatter(
        x=train_df["YearMonth"], y=y_train,
        name="Training Data", mode="lines+markers",
        line=dict(color="#7c3aed"),
    ))
    # Test actual
    fig_fc.add_trace(go.Scatter(
        x=test_df["YearMonth"], y=y_test,
        name="Actual (Test Period)", mode="lines+markers",
        line=dict(color="#10b981"),
    ))
    # Test predicted
    fig_fc.add_trace(go.Scatter(
        x=test_df["YearMonth"], y=test_preds,
        name="Model (Test Period)", mode="lines+markers",
        line=dict(color="#f59e0b", dash="dot"),
    ))
    # Forecast
    fig_fc.add_trace(go.Scatter(
        x=future_dates, y=future_preds,
        name="6-Month Forecast", mode="lines+markers",
        line=dict(color="#ef4444", dash="dash"),
    ))

    fig_fc.update_layout(
        title=f"6-Month {value_col} Forecast with Train/Test Evaluation",
        xaxis_title="Month",
        yaxis_title=value_col,
        template=plotly_template,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── Forecast table ────────────────────────
    st.markdown("#### 📋 Forecast Values")
    forecast_tbl = pd.DataFrame({
        "Month":          future_dates,
        f"Forecasted {value_col}": [f"${v:,.2f}" for v in future_preds],
    })
    st.dataframe(forecast_tbl, use_container_width=True)

    # ── Residual plot ─────────────────────────
    residuals = y_test - test_preds
    fig_res = px.bar(
        x=test_df["YearMonth"], y=residuals,
        title="Residuals on Test Set (Actual − Predicted)",
        labels={"x": "Month", "y": "Residual"},
        template=plotly_template,
        color=residuals,
        color_continuous_scale=["#ef4444", "#ffffff", "#10b981"],
    )
    st.plotly_chart(fig_res, use_container_width=True)

# ─────────────────────────────────────────────
# MODULE 5 — SYSTEM DESIGN  (FIX: was 3-member but listed 4)
# ─────────────────────────────────────────────
elif view == "System Design":
    st.subheader("🏗️ System Architecture")

    st.markdown("""
    ### 4-Member Team Workflow

    | Member | Role | Contribution |
    |---|---|---|
    | **Member 1** | Project Lead | Streamlit routing, UI theme, CSS styling, dark/light mode |
    | **Member 2** | Data Engineer | `load_and_clean_data` pipeline — bytes-safe caching, dedup, date parsing |
    | **Member 3** | Data Analyst | KPI cards, Plotly charts, EDA module, correlation heatmap |
    | **Member 4** | ML Engineer | Linear Regression forecast with train/test split, residual analysis |

    ---

    ### Architecture Flow
    ```
    CSV Upload / Local File
          │
          ▼
    load_and_clean_data()
    ├─ Deduplication
    ├─ Date Parsing (auto-detected)
    ├─ Null Handling
    └─ Derived Columns (YearMonth, Year, Month, Ordinal)
          │
          ▼
    Sidebar Column Mapper
    (auto-detects value / profit / group columns for any dataset)
          │
          ├──▶ Overview Module   → KPIs + Sunburst + Box Plot + Top 10
          ├──▶ Sales Trends      → Monthly Line, YoY Bar, Seasonality, Group Breakdown
          ├──▶ EDA Module        → Correlation Heatmap, Distribution, Scatter, Stats
          └──▶ ML Module         → LinearRegression + Train/Test Split + 6M Forecast + Residuals

    """)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; opacity:0.5; font-size:0.8rem;'>"
    "Data-Driven Insight Generation • Built with Streamlit & Plotly</p>",
    unsafe_allow_html=True,
)
