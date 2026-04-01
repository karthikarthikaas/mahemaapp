import os
import time
import base64
import urllib.parse
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dash import Dash, html, dcc, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc

try:
    from databricks import sql as dbsql
    DBSQL_AVAILABLE = True
except Exception:
    dbsql = None
    DBSQL_AVAILABLE = False


# =========================================================
# APP
# =========================================================
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server
app.title = "MahemaNex OS"


# =========================================================
# THEME
# =========================================================
ORANGE = "#E36A38"
BG = "#F6F7F9"
TEXT = "#111827"
MUTED = "#6B7280"
BORDER = "#E5E7EB"
WHITE = "#FFFFFF"
CARD_SHADOW = "0 4px 14px rgba(15, 23, 42, 0.06)"

DARK_BG = "#0F172A"
DARK_CARD = "#111827"
DARK_TEXT = "#F8FAFC"
DARK_MUTED = "#CBD5E1"
DARK_BORDER = "#334155"

SUGAR_COLOR = "#E36A38"
FINANCE_COLOR = "#2563EB"
ABT_COLOR = "#10B981"
PURPLE_COLOR = "#7C3AED"


def theme_colors(theme_mode: str):
    if theme_mode == "dark":
        return {
            "bg": DARK_BG,
            "card": DARK_CARD,
            "text": DARK_TEXT,
            "muted": DARK_MUTED,
            "border": DARK_BORDER,
            "sidebar": "#020617",
            "sidebar_card": "#111827",
            "button_inactive": "#1E293B",
            "table_header": "#1E293B",
            "grid": "#334155",
            "paper": DARK_CARD,
            "plot": DARK_CARD,
        }
    return {
        "bg": BG,
        "card": WHITE,
        "text": TEXT,
        "muted": MUTED,
        "border": BORDER,
        "sidebar": WHITE,
        "sidebar_card": "#F8FAFC",
        "button_inactive": "#F3F4F6",
        "table_header": "#F3F4F6",
        "grid": "#E5E7EB",
        "paper": WHITE,
        "plot": WHITE,
    }


# =========================================================
# ENV / TABLES
# =========================================================
DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "").strip()
DATABRICKS_HTTP_PATH = os.environ.get("DATABRICKS_HTTP_PATH", "").strip()
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "").strip()

SUGAR_TABLE = "workspace.public.sugar_data"
FINANCE_TABLE = "workspace.public.finance_data"
ABT_TABLE = "workspace.public.abt_data"


# =========================================================
# CACHE
# =========================================================
CACHE_TTL_SECONDS = 1800

_data_cache = {
    "sugars_ceo": {"ts": 0, "df": pd.DataFrame()},
    "finance_ceo": {"ts": 0, "df": pd.DataFrame()},
    "abt_ceo": {"ts": 0, "df": pd.DataFrame()},
}

_prepared_cache = {
    "sugars_ceo": {"ts": 0, "data": None},
    "finance_ceo": {"ts": 0, "data": None},
    "abt_ceo": {"ts": 0, "data": None},
}


# =========================================================
# ROUTES / USERS
# =========================================================
ROUTE_TO_VIEW = {
    "/group": "supreme_ceo",
    "/sugars": "sugars_ceo",
    "/finance": "finance_ceo",
    "/abt": "abt_ceo",
}
VIEW_TO_ROUTE = {v: k for k, v in ROUTE_TO_VIEW.items()}

PERSONAS = {
    "supreme_ceo": {
        "name": "Supreme CEO",
        "role": "Group Chairman",
        "permissions": ["Enterprise", "Sugars", "Finance", "ABT"],
        "title": "Group Enterprise Dashboard",
        "subtitle": "Cross Sector Performance Overview",
    },
    "sugars_ceo": {
        "name": "Mahema Sugars",
        "role": "Sector CEO",
        "permissions": ["Sugars"],
        "title": "Mahema Sugars Performance Console",
        "subtitle": "Production, Recovery and Profitability Monitoring",
    },
    "finance_ceo": {
        "name": "Mahema Finance",
        "role": "Sector CEO",
        "permissions": ["Finance"],
        "title": "Mahema Finance Performance Console",
        "subtitle": "Portfolio, Asset Quality and Collections Monitoring",
    },
    "abt_ceo": {
        "name": "ABT Logistics",
        "role": "Sector CEO",
        "permissions": ["ABT"],
        "title": "ABT Logistics Performance Console",
        "subtitle": "Revenue, Utilization and Delivery Monitoring",
    },
}


# =========================================================
# LOGO
# =========================================================
BASE_DIR = Path(__file__).resolve().parent


def load_logo_data_uri():
    candidates = [
        BASE_DIR / "assets.png",
        BASE_DIR / "logo.png",
        BASE_DIR / "assets.jpg",
        BASE_DIR / "assets.jpeg",
        BASE_DIR / "assets.webp",
    ]
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }

    for path in candidates:
        if path.exists():
            try:
                encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
                mime = mime_map.get(path.suffix.lower(), "image/png")
                return f"data:{mime};base64,{encoded}"
            except Exception:
                pass
    return ""


LOGO_SRC = load_logo_data_uri()


def logo(height=42):
    if LOGO_SRC:
        return html.Img(
            src=LOGO_SRC,
            style={"height": f"{height}px", "width": "auto", "display": "block"},
        )
    return html.Div(
        "S",
        style={
            "height": f"{height}px",
            "width": f"{height}px",
            "borderRadius": "12px",
            "backgroundColor": ORANGE,
            "color": "white",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "fontWeight": "800",
            "fontSize": "18px",
        },
    )


# =========================================================
# HELPERS
# =========================================================
def numeric_value(val, default=0.0):
    try:
        if val is None:
            return default
        if isinstance(val, str):
            cleaned = (
                val.replace(",", "")
                .replace("%", "")
                .replace("₹", "")
                .replace("Cr", "")
                .replace("L", "")
                .strip()
            )
            return float(cleaned)
        return float(val)
    except Exception:
        return default


def format_inr(value):
    try:
        return f"₹{float(value):,.0f}"
    except Exception:
        return "₹0"


def format_inr_cr(value):
    try:
        return f"₹{float(value):,.2f} Cr"
    except Exception:
        return "₹0.00 Cr"


def format_inr_lakhs(value):
    try:
        return f"₹{float(value) / 100000:,.1f} L"
    except Exception:
        return "₹0.0 L"


def format_pct(value):
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "0.0%"


def parse_user_from_search(search: str):
    if not search:
        return None, "light"
    query = urllib.parse.parse_qs(search.lstrip("?"))
    user = query.get("user", [None])[0]
    theme_mode = query.get("theme", ["light"])[0]
    if theme_mode not in ["light", "dark"]:
        theme_mode = "light"
    if user in PERSONAS:
        return user, theme_mode
    return None, theme_mode


def build_href(path: str, user: str, theme_mode: str = "light"):
    return f"{path}?user={user}&theme={theme_mode}"


def db_ready():
    return all([
        DBSQL_AVAILABLE,
        bool(DATABRICKS_HOST),
        bool(DATABRICKS_HTTP_PATH),
        bool(DATABRICKS_TOKEN),
    ])


def db_connection():
    host = DATABRICKS_HOST.replace("https://", "").replace("http://", "").strip("/")
    return dbsql.connect(
        server_hostname=host,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
    )


def query_df(query: str) -> pd.DataFrame:
    if not db_ready():
        return pd.DataFrame()

    try:
        with db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                cols = [c[0] for c in cursor.description] if cursor.description else []
                return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        print(f"SQL query failed: {e}")
        return pd.DataFrame()


def clear_all_caches():
    for key in _data_cache:
        _data_cache[key] = {"ts": 0, "df": pd.DataFrame()}
    for key in _prepared_cache:
        _prepared_cache[key] = {"ts": 0, "data": None}


def get_all_rows_from_db(table_name: str) -> pd.DataFrame:
    return query_df(f"SELECT * FROM {table_name}")


def get_domain_df(view: str) -> pd.DataFrame:
    if view == "sugars_ceo":
        return get_all_rows_from_db(SUGAR_TABLE)
    if view == "finance_ceo":
        return get_all_rows_from_db(FINANCE_TABLE)
    if view == "abt_ceo":
        return get_all_rows_from_db(ABT_TABLE)
    return pd.DataFrame()


def get_cached_domain_df(view: str) -> pd.DataFrame:
    now = time.time()
    cached = _data_cache.get(view)
    if cached and (now - cached["ts"] < CACHE_TTL_SECONDS):
        return cached["df"].copy()

    df = get_domain_df(view)
    print(f"Fetched {len(df)} rows for {view}")
    print(f"Columns for {view}: {list(df.columns)}")
    _data_cache[view] = {"ts": now, "df": df.copy()}
    return df.copy()


def get_quarter_from_month(month_num):
    try:
        month_num = int(month_num)
    except Exception:
        return "Q1"
    if month_num in [1, 2, 3]:
        return "Q1"
    if month_num in [4, 5, 6]:
        return "Q2"
    if month_num in [7, 8, 9]:
        return "Q3"
    return "Q4"


def pick_existing_column(df: pd.DataFrame, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def parse_mixed_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    s = series.astype(str).str.strip()

    parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)

    formats = [
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
    ]

    for fmt in formats:
        mask = parsed.isna()
        if not mask.any():
            break
        try:
            parsed.loc[mask] = pd.to_datetime(s.loc[mask], format=fmt, errors="coerce")
        except Exception:
            pass

    return parsed


def ensure_month_year_columns(df: pd.DataFrame, preferred_date_cols):
    if df.empty:
        return df, None

    date_col = pick_existing_column(df, preferred_date_cols)

    if date_col:
        df[date_col] = parse_mixed_datetime(df[date_col])
        df["year_num"] = df[date_col].dt.year
        df["month_num"] = df[date_col].dt.month
        df["sort_date"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    else:
        year_col = pick_existing_column(df, ["year", "yr", "fiscal_year", "reporting_year"])
        month_col = pick_existing_column(df, ["month", "mnth", "reporting_month", "period_month"])

        if year_col is None or month_col is None:
            return df, None

        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }

        if df[month_col].dtype == object:
            df["month_num"] = df[month_col].astype(str).str[:3].str.lower().map(month_map)
        else:
            df["month_num"] = pd.to_numeric(df[month_col], errors="coerce")

        df["year_num"] = pd.to_numeric(df[year_col], errors="coerce")
        df["sort_date"] = pd.to_datetime(
            dict(
                year=df["year_num"].fillna(1900).astype(int),
                month=df["month_num"].fillna(1).astype(int),
                day=1
            ),
            errors="coerce"
        )

    df = df[df["sort_date"].notna()].copy()
    df["quarter"] = df["month_num"].apply(get_quarter_from_month)
    df["month_label"] = df["sort_date"].dt.strftime("%b %Y")
    return df, date_col


def compact_graph(graph_id, fig, height="250px"):
    return dcc.Graph(
        id=graph_id,
        figure=fig,
        config={
            "displayModeBar": False,
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": False,
            "modeBarButtonsToRemove": [
                "zoom", "pan", "select", "lasso2d", "zoomIn", "zoomOut",
                "autoScale", "resetScale", "toImage", "hoverClosestCartesian",
                "hoverCompareCartesian", "toggleSpikelines"
            ],
        },
        style={"height": height, "cursor": "pointer"},
        clickData=None,
    )


def modal_graph(graph_id):
    return dcc.Graph(
        id=graph_id,
        config={
            "displayModeBar": False,
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": False,
            "modeBarButtonsToRemove": [
                "zoom", "pan", "select", "lasso2d", "zoomIn", "zoomOut",
                "autoScale", "resetScale", "toImage", "hoverClosestCartesian",
                "hoverCompareCartesian", "toggleSpikelines"
            ],
        },
        style={"height": "220px"},
    )


def apply_chart_theme(fig, theme_mode="light"):
    colors = theme_colors(theme_mode)
    fig.update_layout(
        paper_bgcolor=colors["paper"],
        plot_bgcolor=colors["plot"],
        font=dict(color=colors["text"], family="Arial", size=10),
        margin=dict(l=40, r=18, t=42, b=42),
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=9),
            bgcolor="rgba(0,0,0,0)"
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=colors["border"],
        tickfont=dict(color=colors["text"], size=9),
        title_font=dict(color=colors["muted"], size=10),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=colors["grid"],
        zeroline=False,
        tickfont=dict(color=colors["text"], size=9),
        title_font=dict(color=colors["muted"], size=10),
    )
    return fig


def message_figure(message: str, theme_mode="light"):
    colors = theme_colors(theme_mode)
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        text=message,
        showarrow=False,
        font=dict(size=15, color=colors["muted"])
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        paper_bgcolor=colors["paper"],
        plot_bgcolor=colors["plot"],
        margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig


def latest_growth_pct(series: pd.Series) -> float:
    if series is None or len(series.dropna()) < 2:
        return 0.0
    s = series.dropna().astype(float)
    prev = s.iloc[-2]
    curr = s.iloc[-1]
    if prev == 0:
        return 0.0
    return ((curr - prev) / prev) * 100.0


# =========================================================
# EMPTY PREP
# =========================================================
def empty_prepared_sugar():
    return {
        "overall": {},
        "latest": {},
        "monthly": pd.DataFrame(),
        "quarterly": pd.DataFrame(),
        "yearly": pd.DataFrame(),
        "raw": pd.DataFrame(),
        "total_rows": 0,
        "is_live": False,
        "date_col": None,
    }


def empty_prepared_finance():
    return {
        "overall": {},
        "latest": {},
        "monthly": pd.DataFrame(),
        "quarterly": pd.DataFrame(),
        "yearly": pd.DataFrame(),
        "raw": pd.DataFrame(),
        "total_rows": 0,
        "is_live": False,
        "date_col": None,
    }


def empty_prepared_abt():
    return {
        "overall": {},
        "latest": {},
        "monthly": pd.DataFrame(),
        "quarterly": pd.DataFrame(),
        "yearly": pd.DataFrame(),
        "raw": pd.DataFrame(),
        "total_rows": 0,
        "is_live": False,
        "date_col": None,
    }


# =========================================================
# PREP DATA
# =========================================================
def prepare_sugar_data():
    df = get_cached_domain_df("sugars_ceo").copy()
    if df.empty:
        return empty_prepared_sugar()

    df, date_col = ensure_month_year_columns(df, [
        "report_date", "date", "period", "transaction_date", "created_at"
    ])
    if "sort_date" not in df.columns or df.empty:
        return empty_prepared_sugar()

    col_map = {
        "cane_crushed_tons": pick_existing_column(df, ["cane_crushed_tons", "cane_crushed", "cane_crushed_qty"]),
        "sugar_produced_tons": pick_existing_column(df, ["sugar_produced_tons", "sugar_produced", "sugar_qty"]),
        "installed_crushing_capacity": pick_existing_column(df, ["installed_crushing_capacity", "installed_capacity"]),
        "actual_crushing_capacity": pick_existing_column(df, ["actual_crushing_capacity", "actual_capacity", "capacity_used"]),
        "ebitda": pick_existing_column(df, ["ebitda"]),
        "sugar_sold_quintal": pick_existing_column(df, ["sugar_sold_quintal", "sugar_sold_qty"]),
        "sugar_sales_revenue": pick_existing_column(df, ["sugar_sales_revenue", "sales_revenue", "revenue"]),
    }

    for k, v in col_map.items():
        if v is None:
            df[k] = 0.0
        else:
            df[k] = pd.to_numeric(df[v], errors="coerce").fillna(0.0)

    monthly = (
        df.groupby(["year_num", "month_num", "month_label", "sort_date", "quarter"], as_index=False)
        .agg({
            "cane_crushed_tons": "sum",
            "sugar_produced_tons": "sum",
            "installed_crushing_capacity": "sum",
            "actual_crushing_capacity": "sum",
            "ebitda": "sum",
            "sugar_sold_quintal": "sum",
            "sugar_sales_revenue": "sum",
        })
        .sort_values("sort_date")
        .reset_index(drop=True)
    )

    monthly["recovery_pct"] = np.where(
        monthly["cane_crushed_tons"] > 0,
        (monthly["sugar_produced_tons"] / monthly["cane_crushed_tons"]) * 100,
        0
    )
    monthly["crushing_capacity_utilization_pct"] = np.where(
        monthly["installed_crushing_capacity"] > 0,
        (monthly["actual_crushing_capacity"] / monthly["installed_crushing_capacity"]) * 100,
        0
    )
    monthly["ebitda_per_ton"] = np.where(
        monthly["cane_crushed_tons"] > 0,
        monthly["ebitda"] / monthly["cane_crushed_tons"],
        0
    )
    monthly["avg_realization_per_quintal"] = np.where(
        monthly["sugar_sold_quintal"] > 0,
        monthly["sugar_sales_revenue"] / monthly["sugar_sold_quintal"],
        0
    )

    quarterly = (
        monthly.groupby(["year_num", "quarter"], as_index=False)
        .agg({
            "ebitda": "sum",
            "recovery_pct": "mean",
            "crushing_capacity_utilization_pct": "mean",
            "ebitda_per_ton": "mean",
            "avg_realization_per_quintal": "mean",
        })
    )
    quarterly["quarter_label"] = quarterly["year_num"].astype(int).astype(str) + " " + quarterly["quarter"]
    quarterly["sort_order"] = quarterly["year_num"] * 10 + quarterly["quarter"].map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4})
    quarterly = quarterly.sort_values("sort_order")

    yearly = (
        monthly.groupby(["year_num"], as_index=False)
        .agg({
            "ebitda": "sum",
            "recovery_pct": "mean",
            "crushing_capacity_utilization_pct": "mean",
            "ebitda_per_ton": "mean",
            "avg_realization_per_quintal": "mean",
        })
        .sort_values("year_num")
    )
    yearly["year_label"] = yearly["year_num"].astype(int).astype(str)

    latest = monthly.iloc[-1] if not monthly.empty else pd.Series()

    total_cane = df["cane_crushed_tons"].sum()
    total_sugar = df["sugar_produced_tons"].sum()
    total_installed = df["installed_crushing_capacity"].sum()
    total_actual = df["actual_crushing_capacity"].sum()
    total_ebitda = df["ebitda"].sum()
    total_sold = df["sugar_sold_quintal"].sum()
    total_sales_revenue = df["sugar_sales_revenue"].sum()

    overall = {
        "Recovery %": (total_sugar / total_cane) * 100 if total_cane > 0 else 0,
        "Capacity Utilization %": (total_actual / total_installed) * 100 if total_installed > 0 else 0,
        "EBITDA": total_ebitda,
        "EBITDA per Ton": (total_ebitda / total_cane) if total_cane > 0 else 0,
        "Avg Realization": (total_sales_revenue / total_sold) if total_sold > 0 else 0,
    }

    return {
        "overall": overall,
        "latest": {
            "Recovery %": numeric_value(latest.get("recovery_pct", 0)),
            "Capacity Utilization %": numeric_value(latest.get("crushing_capacity_utilization_pct", 0)),
            "EBITDA": numeric_value(latest.get("ebitda", 0)),
            "EBITDA per Ton": numeric_value(latest.get("ebitda_per_ton", 0)),
            "Avg Realization": numeric_value(latest.get("avg_realization_per_quintal", 0)),
        },
        "monthly": monthly,
        "quarterly": quarterly,
        "yearly": yearly,
        "raw": df.sort_values("sort_date", ascending=False),
        "total_rows": len(df),
        "is_live": True,
        "date_col": date_col,
    }


def prepare_finance_data():
    df = get_cached_domain_df("finance_ceo").copy()
    if df.empty:
        return empty_prepared_finance()

    df, date_col = ensure_month_year_columns(df, [
        "disbursement_date", "loan_date", "sanction_date", "report_date",
        "period", "date", "created_at", "npa_date"
    ])
    if "sort_date" not in df.columns or df.empty:
        return empty_prepared_finance()

    col_map = {
        "outstanding_principal": pick_existing_column(df, ["outstanding_principal", "principal_outstanding", "outstanding_amount"]),
        "accrued_interest": pick_existing_column(df, ["accrued_interest", "interest_accrued"]),
        "principal_due": pick_existing_column(df, ["principal_due"]),
        "interest_due": pick_existing_column(df, ["interest_due"]),
        "principal_paid": pick_existing_column(df, ["principal_paid"]),
        "interest_paid": pick_existing_column(df, ["interest_paid"]),
        "loan_amount_disbursed": pick_existing_column(df, ["loan_amount_disbursed", "disbursement_amount", "loan_amount"]),
        "gross_npa_amount": pick_existing_column(df, ["gross_npa_amount", "npa_amount"]),
    }

    for k, v in col_map.items():
        if v is None:
            df[k] = 0.0
        else:
            df[k] = pd.to_numeric(df[v], errors="coerce").fillna(0.0)

    df["aum_value"] = df["outstanding_principal"] + df["accrued_interest"]
    df["collection_due"] = df["principal_due"] + df["interest_due"]
    df["collection_paid"] = df["principal_paid"] + df["interest_paid"]

    monthly = (
        df.groupby(["year_num", "month_num", "month_label", "sort_date", "quarter"], as_index=False)
        .agg({
            "aum_value": "sum",
            "loan_amount_disbursed": "sum",
            "gross_npa_amount": "sum",
            "collection_due": "sum",
            "collection_paid": "sum",
        })
        .sort_values("sort_date")
        .reset_index(drop=True)
    )

    monthly["aum_growth_pct"] = monthly["aum_value"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    monthly["gross_npa_pct"] = np.where(
        monthly["aum_value"] > 0,
        (monthly["gross_npa_amount"] / monthly["aum_value"]) * 100,
        0
    )
    monthly["collection_efficiency_pct"] = np.where(
        monthly["collection_due"] > 0,
        (monthly["collection_paid"] / monthly["collection_due"]) * 100,
        0
    )
    monthly.rename(columns={"aum_value": "aum", "loan_amount_disbursed": "disbursement_volume"}, inplace=True)

    quarterly = (
        monthly.groupby(["year_num", "quarter"], as_index=False)
        .agg({
            "aum": "max",
            "disbursement_volume": "sum",
            "gross_npa_pct": "mean",
            "collection_efficiency_pct": "mean",
        })
    )
    quarterly["quarter_label"] = quarterly["year_num"].astype(int).astype(str) + " " + quarterly["quarter"]
    quarterly["sort_order"] = quarterly["year_num"] * 10 + quarterly["quarter"].map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4})
    quarterly = quarterly.sort_values("sort_order")

    yearly = (
        monthly.groupby(["year_num"], as_index=False)
        .agg({
            "aum": "max",
            "disbursement_volume": "sum",
            "gross_npa_pct": "mean",
            "collection_efficiency_pct": "mean",
        })
        .sort_values("year_num")
    )
    yearly["year_label"] = yearly["year_num"].astype(int).astype(str)

    latest = monthly.iloc[-1] if not monthly.empty else pd.Series()

    total_aum = df["aum_value"].sum()
    total_disbursement = df["loan_amount_disbursed"].sum()
    total_gross_npa = df["gross_npa_amount"].sum()
    total_collection_due = df["collection_due"].sum()
    total_collection_paid = df["collection_paid"].sum()

    overall = {
        "AUM": total_aum,
        "AUM Growth %": latest_growth_pct(monthly["aum"]) if not monthly.empty else 0,
        "Disbursement": total_disbursement,
        "Gross NPA %": (total_gross_npa / total_aum) * 100 if total_aum > 0 else 0,
        "Collection Efficiency %": (total_collection_paid / total_collection_due) * 100 if total_collection_due > 0 else 0,
    }

    return {
        "overall": overall,
        "latest": {
            "AUM": numeric_value(latest.get("aum", 0)),
            "AUM Growth %": numeric_value(latest.get("aum_growth_pct", 0)),
            "Disbursement": numeric_value(latest.get("disbursement_volume", 0)),
            "Gross NPA %": numeric_value(latest.get("gross_npa_pct", 0)),
            "Collection Efficiency %": numeric_value(latest.get("collection_efficiency_pct", 0)),
        },
        "monthly": monthly,
        "quarterly": quarterly,
        "yearly": yearly,
        "raw": df.sort_values("sort_date", ascending=False),
        "total_rows": len(df),
        "is_live": True,
        "date_col": date_col,
    }


def prepare_abt_data():
    df = get_cached_domain_df("abt_ceo").copy()
    if df.empty:
        return empty_prepared_abt()

    df, date_col = ensure_month_year_columns(df, [
        "period", "trip_date", "date", "dispatch_date", "delivery_date",
        "report_date", "created_at"
    ])
    if "sort_date" not in df.columns or df.empty:
        return empty_prepared_abt()

    col_map = {
        "total_revenue": pick_existing_column(df, ["total_revenue", "revenue", "trip_revenue"]),
        "active_vehicle_hours": pick_existing_column(df, ["active_vehicle_hours"]),
        "available_vehicle_hours": pick_existing_column(df, ["available_vehicle_hours"]),
        "deliveries_on_time": pick_existing_column(df, ["deliveries_on_time", "on_time_deliveries"]),
        "total_deliveries": pick_existing_column(df, ["total_deliveries", "deliveries"]),
        "vehicle_id": pick_existing_column(df, ["vehicle_id", "truck_id", "fleet_id"]),
    }

    for k, v in col_map.items():
        if k == "vehicle_id":
            if v is None:
                df["vehicle_id"] = df.index.astype(str)
            else:
                df["vehicle_id"] = df[v].astype(str)
        else:
            if v is None:
                df[k] = 0.0
            else:
                df[k] = pd.to_numeric(df[v], errors="coerce").fillna(0.0)

    monthly = (
        df.groupby(["year_num", "month_num", "month_label", "sort_date", "quarter"], as_index=False)
        .agg({
            "total_revenue": "sum",
            "active_vehicle_hours": "sum",
            "available_vehicle_hours": "sum",
            "deliveries_on_time": "sum",
            "total_deliveries": "sum",
            "vehicle_id": pd.Series.nunique,
        })
        .sort_values("sort_date")
        .reset_index(drop=True)
    )

    monthly["revenue_growth_pct"] = monthly["total_revenue"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    monthly["fleet_utilization_pct"] = np.where(
        monthly["available_vehicle_hours"] > 0,
        (monthly["active_vehicle_hours"] / monthly["available_vehicle_hours"]) * 100,
        0
    )
    monthly["revenue_per_vehicle"] = np.where(
        monthly["vehicle_id"] > 0,
        monthly["total_revenue"] / monthly["vehicle_id"],
        0
    )
    monthly["on_time_delivery_pct"] = np.where(
        monthly["total_deliveries"] > 0,
        (monthly["deliveries_on_time"] / monthly["total_deliveries"]) * 100,
        0
    )

    quarterly = (
        monthly.groupby(["year_num", "quarter"], as_index=False)
        .agg({
            "total_revenue": "sum",
            "fleet_utilization_pct": "mean",
            "on_time_delivery_pct": "mean",
            "revenue_per_vehicle": "mean",
        })
    )
    quarterly["quarter_label"] = quarterly["year_num"].astype(int).astype(str) + " " + quarterly["quarter"]
    quarterly["sort_order"] = quarterly["year_num"] * 10 + quarterly["quarter"].map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4})
    quarterly = quarterly.sort_values("sort_order")

    yearly = (
        monthly.groupby(["year_num"], as_index=False)
        .agg({
            "total_revenue": "sum",
            "fleet_utilization_pct": "mean",
            "on_time_delivery_pct": "mean",
            "revenue_per_vehicle": "mean",
        })
        .sort_values("year_num")
    )
    yearly["year_label"] = yearly["year_num"].astype(int).astype(str)

    latest = monthly.iloc[-1] if not monthly.empty else pd.Series()

    total_revenue = df["total_revenue"].sum()
    total_active_hours = df["active_vehicle_hours"].sum()
    total_available_hours = df["available_vehicle_hours"].sum()
    total_on_time = df["deliveries_on_time"].sum()
    total_deliveries = df["total_deliveries"].sum()
    distinct_vehicles = df["vehicle_id"].nunique()

    overall = {
        "Total Revenue": total_revenue,
        "Revenue Growth %": latest_growth_pct(monthly["total_revenue"]) if not monthly.empty else 0,
        "Fleet Utilization %": (total_active_hours / total_available_hours) * 100 if total_available_hours > 0 else 0,
        "Revenue per Vehicle": (total_revenue / distinct_vehicles) if distinct_vehicles > 0 else 0,
        "On-Time Delivery %": (total_on_time / total_deliveries) * 100 if total_deliveries > 0 else 0,
    }

    return {
        "overall": overall,
        "latest": {
            "Total Revenue": numeric_value(latest.get("total_revenue", 0)),
            "Revenue Growth %": numeric_value(latest.get("revenue_growth_pct", 0)),
            "Fleet Utilization %": numeric_value(latest.get("fleet_utilization_pct", 0)),
            "Revenue per Vehicle": numeric_value(latest.get("revenue_per_vehicle", 0)),
            "On-Time Delivery %": numeric_value(latest.get("on_time_delivery_pct", 0)),
        },
        "monthly": monthly,
        "quarterly": quarterly,
        "yearly": yearly,
        "raw": df.sort_values("sort_date", ascending=False),
        "total_rows": len(df),
        "is_live": True,
        "date_col": date_col,
    }


def get_prepared_sugar_data():
    now = time.time()
    cached = _prepared_cache["sugars_ceo"]
    if cached["data"] is not None and (now - cached["ts"] < CACHE_TTL_SECONDS):
        return cached["data"]
    data = prepare_sugar_data()
    _prepared_cache["sugars_ceo"] = {"ts": now, "data": data}
    return data


def get_prepared_finance_data():
    now = time.time()
    cached = _prepared_cache["finance_ceo"]
    if cached["data"] is not None and (now - cached["ts"] < CACHE_TTL_SECONDS):
        return cached["data"]
    data = prepare_finance_data()
    _prepared_cache["finance_ceo"] = {"ts": now, "data": data}
    return data


def get_prepared_abt_data():
    now = time.time()
    cached = _prepared_cache["abt_ceo"]
    if cached["data"] is not None and (now - cached["ts"] < CACHE_TTL_SECONDS):
        return cached["data"]
    data = prepare_abt_data()
    _prepared_cache["abt_ceo"] = {"ts": now, "data": data}
    return data


# =========================================================
# KPI BUILDERS
# =========================================================
def build_sugar_metrics():
    data = get_prepared_sugar_data()
    overall = data["overall"]
    return [
        {"label": "Recovery %", "value": format_pct(overall.get("Recovery %", 0))},
        {"label": "Capacity Utilization %", "value": format_pct(overall.get("Capacity Utilization %", 0))},
        {"label": "EBITDA", "value": format_inr_cr(overall.get("EBITDA", 0))},
        {"label": "EBITDA / Ton", "value": format_inr(overall.get("EBITDA per Ton", 0))},
        {"label": "Avg Realization / Quintal", "value": format_inr(overall.get("Avg Realization", 0))},
    ], data["total_rows"], data["date_col"]


def build_finance_metrics():
    data = get_prepared_finance_data()
    overall = data["overall"]
    return [
        {"label": "AUM", "value": format_inr_cr(overall.get("AUM", 0))},
        {"label": "AUM Growth %", "value": format_pct(overall.get("AUM Growth %", 0))},
        {"label": "Disbursement", "value": format_inr_cr(overall.get("Disbursement", 0))},
        {"label": "Gross NPA %", "value": format_pct(overall.get("Gross NPA %", 0))},
        {"label": "Collection Efficiency %", "value": format_pct(overall.get("Collection Efficiency %", 0))},
    ], data["total_rows"], data["date_col"]


def build_abt_metrics():
    data = get_prepared_abt_data()
    overall = data["overall"]
    return [
        {"label": "Total Revenue", "value": format_inr_cr(overall.get("Total Revenue", 0))},
        {"label": "Revenue Growth %", "value": format_pct(overall.get("Revenue Growth %", 0))},
        {"label": "Fleet Utilization %", "value": format_pct(overall.get("Fleet Utilization %", 0))},
        {"label": "Revenue / Vehicle", "value": format_inr_lakhs(overall.get("Revenue per Vehicle", 0))},
        {"label": "On-Time Delivery %", "value": format_pct(overall.get("On-Time Delivery %", 0))},
    ], data["total_rows"], data["date_col"]


def build_group_metrics():
    sugar = get_prepared_sugar_data()
    finance = get_prepared_finance_data()
    abt = get_prepared_abt_data()

    total_rows = sugar["total_rows"] + finance["total_rows"] + abt["total_rows"]

    metrics = [
        {"label": "Sugar EBITDA", "value": format_inr_cr(sugar["overall"].get("EBITDA", 0))},
        {"label": "Finance AUM", "value": format_inr_cr(finance["overall"].get("AUM", 0))},
        {"label": "ABT Revenue", "value": format_inr_cr(abt["overall"].get("Total Revenue", 0))},
        {"label": "Sugar Recovery %", "value": format_pct(sugar["overall"].get("Recovery %", 0))},
        {"label": "Finance Collection %", "value": format_pct(finance["overall"].get("Collection Efficiency %", 0))},
        {"label": "ABT On-Time %", "value": format_pct(abt["overall"].get("On-Time Delivery %", 0))},
    ]
    return metrics, total_rows, sugar, finance, abt


# =========================================================
# UI COMPONENTS
# =========================================================
def kpi_card(label, value, theme_mode="light"):
    colors = theme_colors(theme_mode)
    return dbc.Col(
        dbc.Card(
            dbc.CardBody([
                html.Div(
                    label,
                    style={
                        "fontSize": "10px",
                        "color": colors["muted"],
                        "fontWeight": "700",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.06em",
                        "marginBottom": "6px"
                    }
                ),
                html.Div(
                    value,
                    style={
                        "fontSize": "18px",
                        "fontWeight": "800",
                        "color": colors["text"],
                        "lineHeight": "1.15",
                        "whiteSpace": "nowrap",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis"
                    }
                ),
            ]),
            style={
                "border": f"1px solid {colors['border']}",
                "borderRadius": "12px",
                "backgroundColor": colors["card"],
                "boxShadow": CARD_SHADOW,
                "minHeight": "80px"
            }
        ),
        md=6, lg=4, xl=2, className="mb-2"
    )


def chart_card(title, graph_component, theme_mode="light"):
    colors = theme_colors(theme_mode)
    return dbc.Card(
        dbc.CardBody([
            html.Div(
                title,
                style={
                    "fontSize": "13px",
                    "fontWeight": "700",
                    "color": colors["text"],
                    "marginBottom": "6px"
                }
            ),
            graph_component
        ]),
        style={
            "border": f"1px solid {colors['border']}",
            "borderRadius": "12px",
            "backgroundColor": colors["card"],
            "boxShadow": CARD_SHADOW,
            "height": "320px"
        }
    )


def modal_chart(modal_id, chart_title, theme_mode="light"):
    colors = theme_colors(theme_mode)
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(f"{chart_title} Trend Analysis")),
            dbc.ModalBody([
                html.H6("Yearly", style={"color": colors["text"], "fontSize": "13px"}),
                modal_graph(f"{modal_id}-yearly-chart"),

                html.H6("Quarterly", style={"color": colors["text"], "fontSize": "13px", "marginTop": "12px"}),
                modal_graph(f"{modal_id}-quarterly-chart"),

                html.H6("Monthly", style={"color": colors["text"], "fontSize": "13px", "marginTop": "12px"}),
                modal_graph(f"{modal_id}-monthly-chart"),
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id=f"close-{modal_id}", className="ms-auto", n_clicks=0, size="sm")
            ),
        ],
        id=modal_id,
        size="xl",
        is_open=False,
    )


def nav_link(label, href, active=False, theme_mode="light"):
    colors = theme_colors(theme_mode)
    return dcc.Link(
        label,
        href=href,
        refresh=False,
        style={
            "display": "block",
            "width": "100%",
            "padding": "8px 12px",
            "marginBottom": "6px",
            "background": ORANGE if active else colors["button_inactive"],
            "color": "white" if active else colors["text"],
            "borderRadius": "8px",
            "fontWeight": "600",
            "textAlign": "left",
            "textDecoration": "none",
            "fontSize": "13px"
        }
    )


def login_card(label, href):
    return dcc.Link(
        [
            html.Div(label, style={"fontSize": "18px", "fontWeight": "600", "color": "#1F2937"}),
            html.Div("Click to continue", style={"fontSize": "12px", "color": MUTED, "marginTop": "2px"})
        ],
        href=href,
        refresh=False,
        style={
            "display": "block",
            "width": "100%",
            "padding": "15px 18px",
            "marginBottom": "12px",
            "background": "#F8F8F9",
            "border": f"1px solid {BORDER}",
            "borderRadius": "12px",
            "textAlign": "left",
            "textDecoration": "none"
        }
    )


def table_card(title, df, theme_mode="light", total_rows=None):
    colors = theme_colors(theme_mode)
    if df is None or df.empty:
        df = pd.DataFrame([{"Status": "No data found"}])
        total_rows_display = 0
    else:
        total_rows_display = total_rows if total_rows is not None else len(df)

    display_cols = list(df.columns[:8]) if len(df.columns) > 8 else list(df.columns)
    df_display = df[display_cols] if display_cols else df
    columns = [{"name": c, "id": c} for c in df_display.columns]

    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Div(
                    title,
                    style={
                        "fontSize": "13px",
                        "fontWeight": "700",
                        "color": colors["text"],
                        "display": "inline-block"
                    }
                ),
                html.Span(
                    f" ({total_rows_display:,} rows)",
                    style={"fontSize": "11px", "color": colors["muted"], "marginLeft": "6px"}
                ),
            ], style={"marginBottom": "8px"}),
            dash_table.DataTable(
                data=df_display.to_dict("records"),
                columns=columns,
                page_size=8,
                page_action="native",
                filter_action="native",
                sort_action="native",
                style_table={"overflowX": "auto", "maxHeight": "245px", "overflowY": "auto"},
                style_cell={
                    "padding": "6px",
                    "fontFamily": "Arial",
                    "fontSize": "11px",
                    "border": "none",
                    "textAlign": "left",
                    "maxWidth": "140px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "backgroundColor": colors["card"],
                    "color": colors["text"]
                },
                style_header={
                    "fontWeight": "700",
                    "backgroundColor": colors["table_header"],
                    "color": colors["text"],
                    "border": "none",
                    "position": "sticky",
                    "top": 0,
                    "fontSize": "11px"
                },
                style_data={"backgroundColor": colors["card"], "color": colors["text"], "border": "none"},
                style_data_conditional=[
                    {
                        "if": {"row_index": "odd"},
                        "backgroundColor": colors["table_header"] if theme_mode == "light" else "#1E293B",
                    }
                ],
            )
        ]),
        style={
            "border": f"1px solid {colors['border']}",
            "borderRadius": "12px",
            "backgroundColor": colors["card"],
            "boxShadow": CARD_SHADOW,
            "height": "320px"
        }
    )


# =========================================================
# CHARTS
# =========================================================
def group_charts(theme_mode="light"):
    sugar = get_prepared_sugar_data()
    finance = get_prepared_finance_data()
    abt = get_prepared_abt_data()

    fig1 = go.Figure()
    labels = ["Sugars EBITDA", "Finance AUM", "ABT Revenue"]
    values = [
        sugar["overall"].get("EBITDA", 0),
        finance["overall"].get("AUM", 0),
        abt["overall"].get("Total Revenue", 0)
    ]
    fig1.add_trace(go.Bar(
        x=labels,
        y=values,
        marker=dict(color=[SUGAR_COLOR, FINANCE_COLOR, ABT_COLOR], line=dict(width=0)),
        text=[format_inr_cr(v) for v in values],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", size=10),
        hovertemplate="%{x}<br>%{text}<extra></extra>"
    ))
    fig1.update_layout(title="Business Unit Value Snapshot", yaxis_title="Value")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatterpolar(
        r=[
            sugar["overall"].get("Recovery %", 0),
            finance["overall"].get("Collection Efficiency %", 0),
            abt["overall"].get("On-Time Delivery %", 0),
        ],
        theta=["Sugar Recovery %", "Finance Collection %", "ABT On-Time %"],
        fill="toself",
        line=dict(color=ORANGE, width=3),
        fillcolor="rgba(227,106,56,0.25)",
        hovertemplate="%{theta}: %{r:.1f}%<extra></extra>"
    ))
    fig2.update_layout(
        title="Operating Efficiency Radar",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False
    )

    fig3 = go.Figure()
    row_labels = ["Sugars", "Finance", "ABT"]
    row_vals = [sugar["total_rows"], finance["total_rows"], abt["total_rows"]]
    fig3.add_trace(go.Bar(
        x=row_labels,
        y=row_vals,
        marker=dict(color=[SUGAR_COLOR, FINANCE_COLOR, ABT_COLOR], line=dict(width=0)),
        text=[f"{int(v):,}" for v in row_vals],
        textposition="inside",
        textfont=dict(color="white", size=10),
        hovertemplate="%{x}<br>Rows: %{y:,}<extra></extra>"
    ))
    fig3.update_layout(title="Rows Loaded by Business Unit", yaxis_title="Rows")

    return apply_chart_theme(fig1, theme_mode), apply_chart_theme(fig2, theme_mode), apply_chart_theme(fig3, theme_mode), sugar, finance, abt


def sugar_figures(theme_mode="light"):
    prepared = get_prepared_sugar_data()
    monthly = prepared["monthly"]
    yearly = prepared["yearly"]
    raw = prepared["raw"]

    if yearly.empty:
        fig1 = message_figure("No yearly data available", theme_mode)
    else:
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=yearly["year_label"],
            y=yearly["ebitda"],
            marker=dict(color=SUGAR_COLOR),
            text=[format_inr_cr(v) for v in yearly["ebitda"]],
            textposition="inside",
            textfont=dict(color="white", size=10),
            hovertemplate="Year %{x}<br>%{text}<extra></extra>"
        ))
        fig1.update_layout(title="EBITDA by Year", yaxis_title="EBITDA")

    if monthly.empty:
        fig2 = message_figure("No monthly data available", theme_mode)
    else:
        show_df = monthly.tail(12)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=show_df["month_label"],
            y=show_df["recovery_pct"],
            mode="lines+markers",
            name="Recovery %",
            line=dict(color=SUGAR_COLOR, width=3),
            fill="tozeroy",
            fillcolor="rgba(227,106,56,0.12)"
        ))
        fig2.add_trace(go.Scatter(
            x=show_df["month_label"],
            y=show_df["crushing_capacity_utilization_pct"],
            mode="lines+markers",
            name="Capacity Utilization %",
            line=dict(color=FINANCE_COLOR, width=3),
        ))
        fig2.update_layout(title="Recovery vs Capacity Utilization", yaxis_title="%")

    if monthly.empty:
        fig3 = message_figure("No monthly data available", theme_mode)
    else:
        show_df = monthly.tail(12)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=show_df["month_label"],
            y=show_df["ebitda_per_ton"],
            marker=dict(color=SUGAR_COLOR),
            text=[format_inr(v) for v in show_df["ebitda_per_ton"]],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>"
        ))
        fig3.update_layout(title="EBITDA per Ton Trend", yaxis_title="₹")

    return apply_chart_theme(fig1, theme_mode), apply_chart_theme(fig2, theme_mode), apply_chart_theme(fig3, theme_mode), raw, monthly, prepared["quarterly"], yearly


def finance_figures(theme_mode="light"):
    prepared = get_prepared_finance_data()
    monthly = prepared["monthly"]
    yearly = prepared["yearly"]
    raw = prepared["raw"]

    if yearly.empty:
        fig1 = message_figure("No yearly data available", theme_mode)
    else:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=yearly["year_label"],
            y=yearly["aum"],
            mode="lines+markers",
            line=dict(color=FINANCE_COLOR, width=4),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.16)",
            marker=dict(size=8),
            hovertemplate="Year %{x}<br>AUM: %{y:,.2f}<extra></extra>"
        ))
        fig1.update_layout(title="AUM Trend", yaxis_title="AUM")

    if monthly.empty:
        fig2 = message_figure("No monthly data available", theme_mode)
    else:
        show_df = monthly.tail(12)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=show_df["month_label"],
            y=show_df["disbursement_volume"],
            name="Disbursement",
            marker=dict(color=FINANCE_COLOR),
            opacity=0.90
        ))
        fig2.add_trace(go.Scatter(
            x=show_df["month_label"],
            y=show_df["gross_npa_pct"],
            name="Gross NPA %",
            mode="lines+markers",
            line=dict(color=ORANGE, width=3),
            yaxis="y2"
        ))
        fig2.update_layout(
            title="Disbursement vs Gross NPA",
            yaxis=dict(title="Disbursement"),
            yaxis2=dict(title="Gross NPA %", overlaying="y", side="right", showgrid=False)
        )

    if monthly.empty:
        fig3 = message_figure("No monthly data available", theme_mode)
    else:
        show_df = monthly.tail(12)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=show_df["month_label"],
            y=show_df["collection_efficiency_pct"],
            mode="lines+markers",
            line=dict(color=PURPLE_COLOR, width=4),
            fill="tozeroy",
            fillcolor="rgba(124,58,237,0.14)",
            hovertemplate="%{x}<br>Collection Efficiency: %{y:.1f}%<extra></extra>"
        ))
        fig3.update_layout(title="Collection Efficiency Trend", yaxis_title="%")

    return apply_chart_theme(fig1, theme_mode), apply_chart_theme(fig2, theme_mode), apply_chart_theme(fig3, theme_mode), raw, monthly, prepared["quarterly"], yearly


def abt_figures(theme_mode="light"):
    prepared = get_prepared_abt_data()
    monthly = prepared["monthly"]
    yearly = prepared["yearly"]
    raw = prepared["raw"]

    if yearly.empty:
        fig1 = message_figure("No yearly data available", theme_mode)
    else:
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=yearly["year_label"],
            y=yearly["total_revenue"],
            marker=dict(color=ABT_COLOR),
            text=[format_inr_cr(v) for v in yearly["total_revenue"]],
            textposition="inside",
            textfont=dict(color="white", size=10),
            hovertemplate="Year %{x}<br>%{text}<extra></extra>"
        ))
        fig1.update_layout(title="Revenue by Year", yaxis_title="Revenue")

    if monthly.empty:
        fig2 = message_figure("No monthly data available", theme_mode)
    else:
        show_df = monthly.tail(12)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=show_df["month_label"],
            y=show_df["fleet_utilization_pct"],
            mode="lines+markers",
            name="Fleet Utilization %",
            line=dict(color=ABT_COLOR, width=3),
            fill="tozeroy",
            fillcolor="rgba(16,185,129,0.12)"
        ))
        fig2.add_trace(go.Scatter(
            x=show_df["month_label"],
            y=show_df["on_time_delivery_pct"],
            mode="lines+markers",
            name="On-Time Delivery %",
            line=dict(color=ORANGE, width=3),
        ))
        fig2.update_layout(title="Fleet Utilization vs On-Time Delivery", yaxis_title="%")

    if monthly.empty:
        fig3 = message_figure("No monthly data available", theme_mode)
    else:
        show_df = monthly.tail(12)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=show_df["month_label"],
            y=show_df["revenue_per_vehicle"],
            marker=dict(color=ABT_COLOR),
            text=[format_inr_lakhs(v) for v in show_df["revenue_per_vehicle"]],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>"
        ))
        fig3.update_layout(title="Revenue per Vehicle Trend", yaxis_title="₹ Lakhs")

    return apply_chart_theme(fig1, theme_mode), apply_chart_theme(fig2, theme_mode), apply_chart_theme(fig3, theme_mode), raw, monthly, prepared["quarterly"], yearly


def get_modal_charts(view, chart_type, theme_mode):
    if view == "sugars_ceo":
        prepared = get_prepared_sugar_data()
        if prepared["monthly"].empty:
            msg = message_figure("No data available", theme_mode)
            return msg, msg, msg

        if chart_type == "chart-one":
            fig_yearly = go.Figure([go.Bar(x=prepared["yearly"]["year_label"], y=prepared["yearly"]["ebitda"], marker_color=SUGAR_COLOR)])
            fig_quarterly = go.Figure([go.Bar(x=prepared["quarterly"]["quarter_label"], y=prepared["quarterly"]["ebitda"], marker_color=SUGAR_COLOR)])
            fig_monthly = go.Figure([go.Scatter(x=prepared["monthly"]["month_label"], y=prepared["monthly"]["ebitda"], mode="lines+markers", line=dict(color=SUGAR_COLOR, width=3))])
            fig_yearly.update_layout(title="EBITDA by Year")
            fig_quarterly.update_layout(title="EBITDA by Quarter")
            fig_monthly.update_layout(title="EBITDA by Month")

        elif chart_type == "chart-two":
            fig_yearly = go.Figure([go.Bar(x=prepared["yearly"]["year_label"], y=prepared["yearly"]["recovery_pct"], marker_color=SUGAR_COLOR)])
            fig_quarterly = go.Figure([go.Bar(x=prepared["quarterly"]["quarter_label"], y=prepared["quarterly"]["recovery_pct"], marker_color=SUGAR_COLOR)])
            fig_monthly = go.Figure([go.Scatter(x=prepared["monthly"]["month_label"], y=prepared["monthly"]["recovery_pct"], mode="lines+markers", line=dict(color=SUGAR_COLOR, width=3))])
            fig_yearly.update_layout(title="Recovery % by Year")
            fig_quarterly.update_layout(title="Recovery % by Quarter")
            fig_monthly.update_layout(title="Recovery % by Month")

        else:
            fig_yearly = go.Figure([go.Bar(x=prepared["yearly"]["year_label"], y=prepared["yearly"]["ebitda_per_ton"], marker_color=SUGAR_COLOR)])
            fig_quarterly = go.Figure([go.Bar(x=prepared["quarterly"]["quarter_label"], y=prepared["quarterly"]["ebitda_per_ton"], marker_color=SUGAR_COLOR)])
            fig_monthly = go.Figure([go.Scatter(x=prepared["monthly"]["month_label"], y=prepared["monthly"]["ebitda_per_ton"], mode="lines+markers", line=dict(color=SUGAR_COLOR, width=3))])
            fig_yearly.update_layout(title="EBITDA / Ton by Year")
            fig_quarterly.update_layout(title="EBITDA / Ton by Quarter")
            fig_monthly.update_layout(title="EBITDA / Ton by Month")

    elif view == "finance_ceo":
        prepared = get_prepared_finance_data()
        if prepared["monthly"].empty:
            msg = message_figure("No data available", theme_mode)
            return msg, msg, msg

        if chart_type == "chart-one":
            fig_yearly = go.Figure([go.Bar(x=prepared["yearly"]["year_label"], y=prepared["yearly"]["aum"], marker_color=FINANCE_COLOR)])
            fig_quarterly = go.Figure([go.Bar(x=prepared["quarterly"]["quarter_label"], y=prepared["quarterly"]["aum"], marker_color=FINANCE_COLOR)])
            fig_monthly = go.Figure([go.Scatter(x=prepared["monthly"]["month_label"], y=prepared["monthly"]["aum"], mode="lines+markers", line=dict(color=FINANCE_COLOR, width=3))])
            fig_yearly.update_layout(title="AUM by Year")
            fig_quarterly.update_layout(title="AUM by Quarter")
            fig_monthly.update_layout(title="AUM by Month")

        elif chart_type == "chart-two":
            fig_yearly = go.Figure([go.Bar(x=prepared["yearly"]["year_label"], y=prepared["yearly"]["disbursement_volume"], marker_color=FINANCE_COLOR)])
            fig_quarterly = go.Figure([go.Bar(x=prepared["quarterly"]["quarter_label"], y=prepared["quarterly"]["disbursement_volume"], marker_color=FINANCE_COLOR)])
            fig_monthly = go.Figure([go.Scatter(x=prepared["monthly"]["month_label"], y=prepared["monthly"]["disbursement_volume"], mode="lines+markers", line=dict(color=FINANCE_COLOR, width=3))])
            fig_yearly.update_layout(title="Disbursement by Year")
            fig_quarterly.update_layout(title="Disbursement by Quarter")
            fig_monthly.update_layout(title="Disbursement by Month")

        else:
            fig_yearly = go.Figure([go.Bar(x=prepared["yearly"]["year_label"], y=prepared["yearly"]["collection_efficiency_pct"], marker_color=PURPLE_COLOR)])
            fig_quarterly = go.Figure([go.Bar(x=prepared["quarterly"]["quarter_label"], y=prepared["quarterly"]["collection_efficiency_pct"], marker_color=PURPLE_COLOR)])
            fig_monthly = go.Figure([go.Scatter(x=prepared["monthly"]["month_label"], y=prepared["monthly"]["collection_efficiency_pct"], mode="lines+markers", line=dict(color=PURPLE_COLOR, width=3))])
            fig_yearly.update_layout(title="Collection Efficiency by Year")
            fig_quarterly.update_layout(title="Collection Efficiency by Quarter")
            fig_monthly.update_layout(title="Collection Efficiency by Month")

    elif view == "abt_ceo":
        prepared = get_prepared_abt_data()
        if prepared["monthly"].empty:
            msg = message_figure("No data available", theme_mode)
            return msg, msg, msg

        if chart_type == "chart-one":
            fig_yearly = go.Figure([go.Bar(x=prepared["yearly"]["year_label"], y=prepared["yearly"]["total_revenue"], marker_color=ABT_COLOR)])
            fig_quarterly = go.Figure([go.Bar(x=prepared["quarterly"]["quarter_label"], y=prepared["quarterly"]["total_revenue"], marker_color=ABT_COLOR)])
            fig_monthly = go.Figure([go.Scatter(x=prepared["monthly"]["month_label"], y=prepared["monthly"]["total_revenue"], mode="lines+markers", line=dict(color=ABT_COLOR, width=3))])
            fig_yearly.update_layout(title="Revenue by Year")
            fig_quarterly.update_layout(title="Revenue by Quarter")
            fig_monthly.update_layout(title="Revenue by Month")

        elif chart_type == "chart-two":
            fig_yearly = go.Figure([go.Bar(x=prepared["yearly"]["year_label"], y=prepared["yearly"]["fleet_utilization_pct"], marker_color=ABT_COLOR)])
            fig_quarterly = go.Figure([go.Bar(x=prepared["quarterly"]["quarter_label"], y=prepared["quarterly"]["fleet_utilization_pct"], marker_color=ABT_COLOR)])
            fig_monthly = go.Figure([go.Scatter(x=prepared["monthly"]["month_label"], y=prepared["monthly"]["fleet_utilization_pct"], mode="lines+markers", line=dict(color=ABT_COLOR, width=3))])
            fig_yearly.update_layout(title="Fleet Utilization % by Year")
            fig_quarterly.update_layout(title="Fleet Utilization % by Quarter")
            fig_monthly.update_layout(title="Fleet Utilization % by Month")

        else:
            fig_yearly = go.Figure([go.Bar(x=prepared["yearly"]["year_label"], y=prepared["yearly"]["revenue_per_vehicle"], marker_color=ABT_COLOR)])
            fig_quarterly = go.Figure([go.Bar(x=prepared["quarterly"]["quarter_label"], y=prepared["quarterly"]["revenue_per_vehicle"], marker_color=ABT_COLOR)])
            fig_monthly = go.Figure([go.Scatter(x=prepared["monthly"]["month_label"], y=prepared["monthly"]["revenue_per_vehicle"], mode="lines+markers", line=dict(color=ABT_COLOR, width=3))])
            fig_yearly.update_layout(title="Revenue per Vehicle by Year")
            fig_quarterly.update_layout(title="Revenue per Vehicle by Quarter")
            fig_monthly.update_layout(title="Revenue per Vehicle by Month")

    else:
        sugar, finance, abt = get_prepared_sugar_data(), get_prepared_finance_data(), get_prepared_abt_data()
        fig_yearly = go.Figure([go.Bar(
            x=["Sugars EBITDA", "Finance AUM", "ABT Revenue"],
            y=[sugar["overall"].get("EBITDA", 0), finance["overall"].get("AUM", 0), abt["overall"].get("Total Revenue", 0)],
            marker_color=[SUGAR_COLOR, FINANCE_COLOR, ABT_COLOR]
        )])
        fig_yearly.update_layout(title="Business Unit Value Snapshot")
        fig_quarterly = message_figure("Open a sector console for quarterly drill-down", theme_mode)
        fig_monthly = message_figure("Open a sector console for monthly drill-down", theme_mode)

    return apply_chart_theme(fig_yearly, theme_mode), apply_chart_theme(fig_quarterly, theme_mode), apply_chart_theme(fig_monthly, theme_mode)


# =========================================================
# PAGE LAYOUTS
# =========================================================
def login_page():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [logo(56), html.Span("MahemaNex OS", style={"fontSize": "24px", "fontWeight": "700", "color": "#273142", "marginLeft": "12px"})],
                        style={"display": "flex", "alignItems": "center", "marginBottom": "60px"}
                    ),
                    html.Div("Fueling the Future", style={"fontSize": "42px", "fontWeight": "700", "color": "#0F172A", "lineHeight": "1.1"}),
                    html.Div("with Tech", style={"fontSize": "42px", "fontWeight": "700", "color": ORANGE, "lineHeight": "1.1", "marginTop": "4px"}),
                    html.Div("Enterprise Operating System", style={"fontSize": "16px", "color": MUTED, "lineHeight": "1.6", "marginTop": "20px", "maxWidth": "480px"}),
                ],
                style={"width": "50%", "padding": "40px 50px", "borderRight": f"1px solid {BORDER}", "backgroundColor": BG}
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Enterprise Portal", style={"fontSize": "28px", "fontWeight": "700", "color": "#273142", "textAlign": "center"}),
                            html.Div("Select your console", style={"fontSize": "14px", "color": MUTED, "textAlign": "center", "marginTop": "6px", "marginBottom": "28px"}),
                            login_card("Supreme CEO", build_href("/group", "supreme_ceo", "light")),
                            login_card("Mahema Sugars", build_href("/sugars", "sugars_ceo", "light")),
                            login_card("Mahema Finance", build_href("/finance", "finance_ceo", "light")),
                            login_card("ABT Logistics", build_href("/abt", "abt_ceo", "light")),
                        ],
                        style={"width": "100%", "maxWidth": "420px", "padding": "30px", "backgroundColor": "#FAFAFB", "border": f"1px solid {BORDER}", "borderRadius": "18px", "boxShadow": "0 6px 18px rgba(0,0,0,0.05)"}
                    )
                ],
                style={"width": "50%", "display": "flex", "alignItems": "center", "justifyContent": "center", "padding": "30px", "backgroundColor": BG}
            )
        ],
        style={"display": "flex", "minHeight": "100vh", "backgroundColor": BG, "fontFamily": "Arial, sans-serif"}
    )


def sidebar(user, active_view, theme_mode="light"):
    persona = PERSONAS[user]
    colors = theme_colors(theme_mode)

    if "Enterprise" in persona["permissions"]:
        buttons = [
            nav_link("Group Overview", build_href("/group", user, theme_mode), active_view == "supreme_ceo", theme_mode),
            nav_link("Mahema Sugars", build_href("/sugars", user, theme_mode), active_view == "sugars_ceo", theme_mode),
            nav_link("Mahema Finance", build_href("/finance", user, theme_mode), active_view == "finance_ceo", theme_mode),
            nav_link("ABT Logistics", build_href("/abt", user, theme_mode), active_view == "abt_ceo", theme_mode),
        ]
    else:
        buttons = [nav_link(persona["name"], build_href(VIEW_TO_ROUTE[user], user, theme_mode), True, theme_mode)]

    return html.Div(
        [
            html.Div(
                [logo(36), html.Span("MahemaNex", style={"fontSize": "20px", "fontWeight": "700", "color": colors["text"], "marginLeft": "8px"})],
                style={"display": "flex", "alignItems": "center", "marginBottom": "22px"}
            ),
            html.Div("CONSOLES", style={"fontSize": "10px", "fontWeight": "700", "letterSpacing": "0.08em", "textTransform": "uppercase", "color": colors["muted"], "marginBottom": "10px"}),
            html.Div(buttons),
            html.Div(style={"flex": "1"}),
            html.Div(
                [
                    html.Div(persona["name"], style={"fontSize": "14px", "fontWeight": "700", "color": colors["text"]}),
                    html.Div(persona["role"], style={"fontSize": "11px", "fontWeight": "500", "color": colors["muted"], "marginTop": "2px"}),
                ],
                style={"padding": "12px", "borderRadius": "10px", "backgroundColor": colors["sidebar_card"], "border": f"1px solid {colors['border']}", "marginBottom": "8px"}
            ),
            dcc.Link(
                "Logout",
                href="/",
                refresh=False,
                style={"display": "block", "width": "100%", "padding": "10px", "borderRadius": "8px", "backgroundColor": "#FEF2F2", "color": "#DC2626", "fontWeight": "700", "textAlign": "center", "textDecoration": "none", "fontSize": "13px"}
            )
        ],
        style={"width": "210px", "minHeight": "100vh", "padding": "16px 12px", "borderRight": f"1px solid {colors['border']}", "backgroundColor": colors["sidebar"], "display": "flex", "flexDirection": "column"}
    )


def top_theme_toggle(user, active_view, theme_mode="light"):
    colors = theme_colors(theme_mode)
    toggle_target = "dark" if theme_mode == "light" else "light"
    toggle_icon = "☾" if theme_mode == "light" else "☀"

    return dcc.Link(
        toggle_icon,
        href=build_href(VIEW_TO_ROUTE.get(active_view, "/group"), user, toggle_target),
        refresh=False,
        style={"position": "fixed", "top": "12px", "right": "16px", "width": "36px", "height": "36px", "borderRadius": "50%", "display": "flex", "alignItems": "center", "justifyContent": "center", "textDecoration": "none", "fontSize": "18px", "color": colors["text"], "backgroundColor": colors["card"], "border": f"1px solid {colors['border']}", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)", "zIndex": "9999"}
    )


def dashboard_shell(view, theme_mode="light"):
    colors = theme_colors(theme_mode)
    persona = PERSONAS[view]

    if view == "sugars_ceo":
        chart_titles = ["EBITDA by Year", "Recovery vs Capacity Utilization", "EBITDA per Ton Trend"]
    elif view == "finance_ceo":
        chart_titles = ["AUM Trend", "Disbursement vs Gross NPA", "Collection Efficiency Trend"]
    elif view == "abt_ceo":
        chart_titles = ["Revenue by Year", "Fleet Utilization vs On-Time Delivery", "Revenue per Vehicle Trend"]
    else:
        chart_titles = ["Business Unit Value Snapshot", "Operating Efficiency Radar", "Rows Loaded by Business Unit"]

    return html.Div([
        html.Div([
            html.Div(persona["title"], style={"fontSize": "24px", "fontWeight": "700", "color": colors["text"], "lineHeight": "1.2"}),
            html.Div(persona["subtitle"], style={"fontSize": "12px", "fontWeight": "500", "color": colors["muted"], "marginTop": "2px"}),
        ], style={"marginBottom": "12px"}),

        html.Div(id="status-text", style={"fontSize": "12px", "color": ORANGE, "marginBottom": "10px"}),

        html.Div(id="kpi-row", style={"marginBottom": "6px"}),

        dbc.Row([
            dbc.Col(html.Div(id="chart-one-wrap"), lg=4, className="mb-2"),
            dbc.Col(html.Div(id="chart-two-wrap"), lg=4, className="mb-2"),
            dbc.Col(html.Div(id="chart-three-wrap"), lg=4, className="mb-2"),
        ], className="g-2"),

        dbc.Row([
            dbc.Col(html.Div(id="table-wrap"), lg=12, className="mb-2"),
        ], className="g-2"),

        modal_chart("modal-chart-one", chart_titles[0], theme_mode),
        modal_chart("modal-chart-two", chart_titles[1], theme_mode),
        modal_chart("modal-chart-three", chart_titles[2], theme_mode),

        dcc.Store(id="current-view", data=view),
        dcc.Store(id="current-theme", data=theme_mode),
    ])


# =========================================================
# APP LAYOUT
# =========================================================
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Interval(id="auto-refresh", interval=30 * 60 * 1000, n_intervals=0),
    html.Div(id="page")
])


# =========================================================
# PAGE RENDER
# =========================================================
@app.callback(
    Output("page", "children"),
    Input("url", "pathname"),
    Input("url", "search"),
)
def render_page(pathname, search):
    user, theme_mode = parse_user_from_search(search)

    if pathname == "/" or user is None:
        return login_page()

    view = ROUTE_TO_VIEW.get(pathname, "supreme_ceo")
    if user != "supreme_ceo":
        view = user

    colors = theme_colors(theme_mode)

    return html.Div(
        [
            sidebar(user, view, theme_mode),
            html.Div(
                dashboard_shell(view, theme_mode),
                style={"padding": "18px", "flex": "1", "backgroundColor": colors["bg"], "minHeight": "100vh"}
            ),
            top_theme_toggle(user, view, theme_mode),
        ],
        style={"display": "flex", "fontFamily": "Arial, sans-serif", "backgroundColor": colors["bg"]}
    )


# =========================================================
# DASHBOARD CONTENT
# =========================================================
@app.callback(
    Output("status-text", "children"),
    Output("kpi-row", "children"),
    Output("chart-one-wrap", "children"),
    Output("chart-two-wrap", "children"),
    Output("chart-three-wrap", "children"),
    Output("table-wrap", "children"),
    Output("current-view", "data"),
    Output("current-theme", "data"),
    Input("url", "pathname"),
    Input("url", "search"),
    Input("auto-refresh", "n_intervals"),
)
def update_dashboard_content(pathname, search, n_intervals):
    if n_intervals and n_intervals > 0:
        clear_all_caches()

    user, theme_mode = parse_user_from_search(search)
    if pathname == "/" or user is None:
        blank = html.Div()
        return "", blank, blank, blank, blank, blank, None, theme_mode

    view = ROUTE_TO_VIEW.get(pathname, "supreme_ceo")
    if user != "supreme_ceo":
        view = user

    if view == "supreme_ceo":
        metrics, total_rows, sugar, finance, abt = build_group_metrics()
        kpis = dbc.Row([kpi_card(m["label"], m["value"], theme_mode) for m in metrics], className="g-2")

        fig1, fig2, fig3, sugar, finance, abt = group_charts(theme_mode)

        chart1 = chart_card("Business Unit Value Snapshot", compact_graph("chart-one", fig1), theme_mode)
        chart2 = chart_card("Operating Efficiency Radar", compact_graph("chart-two", fig2), theme_mode)
        chart3 = chart_card("Rows Loaded by Business Unit", compact_graph("chart-three", fig3), theme_mode)

        summary_data = [
            {
                "Business Unit": "Sugars",
                "Primary KPI": "EBITDA",
                "Value": format_inr_cr(sugar["overall"].get("EBITDA", 0)),
                "Rows": sugar["total_rows"],
                "Date Column Used": sugar.get("date_col") or "month/year"
            },
            {
                "Business Unit": "Finance",
                "Primary KPI": "AUM",
                "Value": format_inr_cr(finance["overall"].get("AUM", 0)),
                "Rows": finance["total_rows"],
                "Date Column Used": finance.get("date_col") or "month/year"
            },
            {
                "Business Unit": "ABT Logistics",
                "Primary KPI": "Revenue",
                "Value": format_inr_cr(abt["overall"].get("Total Revenue", 0)),
                "Rows": abt["total_rows"],
                "Date Column Used": abt.get("date_col") or "month/year"
            },
        ]
        table_box = table_card("Group KPI Summary", pd.DataFrame(summary_data), theme_mode, total_rows)
        status = f"✅ Live Data loaded from all tables | Total rows: {total_rows:,}"

        return status, kpis, chart1, chart2, chart3, table_box, view, theme_mode

    elif view == "sugars_ceo":
        metrics, total_rows, date_col = build_sugar_metrics()
        kpis = dbc.Row([kpi_card(m["label"], m["value"], theme_mode) for m in metrics], className="g-2")

        fig1, fig2, fig3, raw_df, monthly, quarterly, yearly = sugar_figures(theme_mode)

        chart1 = chart_card("EBITDA by Year", compact_graph("chart-one", fig1), theme_mode)
        chart2 = chart_card("Recovery vs Capacity Utilization", compact_graph("chart-two", fig2), theme_mode)
        chart3 = chart_card("EBITDA per Ton Trend", compact_graph("chart-three", fig3), theme_mode)

        table_box = table_card("Sugar Raw Data", raw_df, theme_mode, total_rows)
        status = f"✅ Live Data loaded from sugar table | Total rows: {total_rows:,} | Date column: {date_col or 'month/year'}"

        return status, kpis, chart1, chart2, chart3, table_box, view, theme_mode

    elif view == "finance_ceo":
        metrics, total_rows, date_col = build_finance_metrics()
        kpis = dbc.Row([kpi_card(m["label"], m["value"], theme_mode) for m in metrics], className="g-2")

        fig1, fig2, fig3, raw_df, monthly, quarterly, yearly = finance_figures(theme_mode)

        chart1 = chart_card("AUM Trend", compact_graph("chart-one", fig1), theme_mode)
        chart2 = chart_card("Disbursement vs Gross NPA", compact_graph("chart-two", fig2), theme_mode)
        chart3 = chart_card("Collection Efficiency Trend", compact_graph("chart-three", fig3), theme_mode)

        table_box = table_card("Finance Raw Data", raw_df, theme_mode, total_rows)
        status = f"✅ Live Data loaded from finance table | Total rows: {total_rows:,} | Date column: {date_col or 'month/year'}"

        return status, kpis, chart1, chart2, chart3, table_box, view, theme_mode

    else:
        metrics, total_rows, date_col = build_abt_metrics()
        kpis = dbc.Row([kpi_card(m["label"], m["value"], theme_mode) for m in metrics], className="g-2")

        fig1, fig2, fig3, raw_df, monthly, quarterly, yearly = abt_figures(theme_mode)

        chart1 = chart_card("Revenue by Year", compact_graph("chart-one", fig1), theme_mode)
        chart2 = chart_card("Fleet Utilization vs On-Time Delivery", compact_graph("chart-two", fig2), theme_mode)
        chart3 = chart_card("Revenue per Vehicle Trend", compact_graph("chart-three", fig3), theme_mode)

        table_box = table_card("ABT Raw Data", raw_df, theme_mode, total_rows)
        status = f"✅ Live Data loaded from ABT table | Total rows: {total_rows:,} | Date column: {date_col or 'month/year'}"

        return status, kpis, chart1, chart2, chart3, table_box, view, theme_mode


# =========================================================
# MODALS
# =========================================================
@app.callback(
    Output("modal-chart-one", "is_open"),
    [Input("chart-one", "clickData"), Input("close-modal-chart-one", "n_clicks")],
    [State("modal-chart-one", "is_open")],
)
def toggle_modal_one(click_data, n_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return is_open
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "chart-one" and click_data:
        return True
    if trigger_id == "close-modal-chart-one" and n_clicks:
        return False
    return is_open


@app.callback(
    Output("modal-chart-two", "is_open"),
    [Input("chart-two", "clickData"), Input("close-modal-chart-two", "n_clicks")],
    [State("modal-chart-two", "is_open")],
)
def toggle_modal_two(click_data, n_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return is_open
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "chart-two" and click_data:
        return True
    if trigger_id == "close-modal-chart-two" and n_clicks:
        return False
    return is_open


@app.callback(
    Output("modal-chart-three", "is_open"),
    [Input("chart-three", "clickData"), Input("close-modal-chart-three", "n_clicks")],
    [State("modal-chart-three", "is_open")],
)
def toggle_modal_three(click_data, n_clicks, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return is_open
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "chart-three" and click_data:
        return True
    if trigger_id == "close-modal-chart-three" and n_clicks:
        return False
    return is_open


@app.callback(
    Output("modal-chart-one-yearly-chart", "figure"),
    Output("modal-chart-one-quarterly-chart", "figure"),
    Output("modal-chart-one-monthly-chart", "figure"),
    Input("modal-chart-one", "is_open"),
    Input("current-view", "data"),
    Input("current-theme", "data"),
    prevent_initial_call=True,
)
def update_modal_one_charts(is_open, view, theme_mode):
    if not is_open or not view:
        msg = message_figure("No data", theme_mode)
        return msg, msg, msg
    return get_modal_charts(view, "chart-one", theme_mode)


@app.callback(
    Output("modal-chart-two-yearly-chart", "figure"),
    Output("modal-chart-two-quarterly-chart", "figure"),
    Output("modal-chart-two-monthly-chart", "figure"),
    Input("modal-chart-two", "is_open"),
    Input("current-view", "data"),
    Input("current-theme", "data"),
    prevent_initial_call=True,
)
def update_modal_two_charts(is_open, view, theme_mode):
    if not is_open or not view:
        msg = message_figure("No data", theme_mode)
        return msg, msg, msg
    return get_modal_charts(view, "chart-two", theme_mode)


@app.callback(
    Output("modal-chart-three-yearly-chart", "figure"),
    Output("modal-chart-three-quarterly-chart", "figure"),
    Output("modal-chart-three-monthly-chart", "figure"),
    Input("modal-chart-three", "is_open"),
    Input("current-view", "data"),
    Input("current-theme", "data"),
    prevent_initial_call=True,
)
def update_modal_three_charts(is_open, view, theme_mode):
    if not is_open or not view:
        msg = message_figure("No data", theme_mode)
        return msg, msg, msg
    return get_modal_charts(view, "chart-three", theme_mode)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
