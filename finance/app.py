import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    confusion_matrix,
    roc_curve
)

# ---------------------------
# Global styling / Page setup
# ---------------------------
st.set_page_config(
    page_title="Vesta Fraud Risk Dashboard",
    layout="wide"
)

# Light, clean seaborn style (kept in case we use matplotlib anywhere)
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (5, 3)

# Custom CSS to make things look more "dashboardy" and aesthetic
st.markdown(
    """
    <style>
        /* Main background */
        .main {
            background-color: #f5f7fb;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
        }

        /* Container padding */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border-radius: 0.75rem;
            padding: 0.75rem 0.75rem 0.5rem 0.75rem;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
        }

        /* Headings */
        h1, h2, h3 {
            font-family: "Segoe UI", system-ui, sans-serif;
        }

        .stMarkdown h3 {
            margin-top: 0.75rem;
            margin-bottom: 0.25rem;
        }

        /* DataFrame styling wrapper */
        .stDataFrame {
            border-radius: 0.75rem;
            overflow: hidden;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Load data & model
# =========================
@st.cache_data
def load_main_data():
    # final condensed Vesta dataset with isFraud + engineered features
    return pd.read_csv("data/vesta_condensed_final.csv")

@st.cache_data
def load_validation_data():
    X_valid_enc = pd.read_csv("data/X_valid_enc.csv")
    y_valid = pd.read_csv("data/y_valid.csv")["isFraud"]
    return X_valid_enc, y_valid

@st.cache_resource
def load_model():
    model = joblib.load("models/final_lgbm.pkl")
    return model


df = load_main_data()
X_valid_enc, y_valid = load_validation_data()
model = load_model()

# ================================================
# Sidebar Navigation
# ================================================
st.sidebar.title("Navigation")
PAGES = [
    "Executive Summary",
    "EDA",
    "Model Performance"
]
page = st.sidebar.radio("", PAGES)

# ================================================
# PAGE 1: Executive Summary
# ================================================
if page == "Executive Summary":
    st.markdown("## Vesta Fraud Risk – Overview")

    if "isFraud" in df.columns:
        total_tx = len(df)
        total_fraud = df["isFraud"].sum()
        fraud_rate = (total_fraud / total_tx) * 100

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Transactions", f"{total_tx:,}")
        with c2:
            st.metric("Fraudulent Transactions", f"{int(total_fraud):,}")
        with c3:
            st.metric("Overall Fraud Rate", f"{fraud_rate:.2f}%")
    else:
        st.error("Column 'isFraud' not found in dataset.")

    st.markdown("---")

    st.markdown("### Sample from Final Engineered Vesta Dataset")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown(
        """
        ### Setup

        This dashboard is built on the IEEE‑CIS / Vesta transaction dataset with:
        - Engineered dataset: `data/vesta_condensed_final.csv`  
        - Trained model: LightGBM (`models/final_lgbm.pkl`)  
        """
    )

# ================================================
# PAGE 2: EDA (interactive, Plotly)
# ================================================
elif page == "EDA":
    st.markdown("## Exploratory Data Analysis")

    # ---- Row 1: Class balance interactive donut + text ----
    if "isFraud" in df.columns:
        st.markdown("### Class Distribution (Fraud vs Non‑Fraud)")
        col_pie, col_text = st.columns([2, 1])

        counts = df["isFraud"].value_counts().reindex([0, 1], fill_value=0)
        total_tx = int(counts.sum())

        pie_df = pd.DataFrame({
            "Class": ["Non-Fraud", "Fraud"],
            "Count": [counts[0], counts[1]]
        })

        with col_pie:
            fig_pie = px.pie(
                pie_df,
                names="Class",
                values="Count",
                hole=0.7,
                color="Class",
                color_discrete_map={
                    "Non-Fraud": "#4e79a7",
                    "Fraud": "#f28e2b"
                }
            )
            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent",
                hovertemplate="Class: %{label}<br>Count: %{value}<br>Percent: %{percent}"
            )
            fig_pie.add_annotation(
                dict(
                    text=f"Total<br>{total_tx:,}<br>transactions",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    x=0.5, y=0.5, xanchor="center", yanchor="middle"
                )
            )
            fig_pie.update_layout(
                showlegend=True,
                margin=dict(l=0, r=0, t=30, b=0),
                title=dict(text="Fraud Distribution (Donut Chart)", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_text:
            fraud_rate = (counts[1] / total_tx) * 100 if total_tx > 0 else 0
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
            st.write(
                "Hover over each segment to see the exact count and percentage. "
                "The donut center shows the total transaction volume in the Vesta dataset."
            )

    st.markdown("---")

    # ---- Row 2: Transaction amount + ProductCD ----
    st.markdown("### Transaction and Product Overview")
    col1, col2 = st.columns(2)

    with col1:
        if "TransactionAmt" in df.columns:
            st.caption("Transaction Amount (log‑scale)")
            temp = df.copy()
            temp["log_TransactionAmt"] = np.log1p(temp["TransactionAmt"])
            fig_amt = px.histogram(
                temp,
                x="log_TransactionAmt",
                nbins=60,
                labels={"log_TransactionAmt": "log(TransactionAmt + 1)"},
            )
            fig_amt.update_layout(margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_amt, use_container_width=True)
        else:
            st.warning("Column 'TransactionAmt' not found.")

    with col2:
        if "ProductCD" in df.columns and "isFraud" in df.columns:
            st.caption("Fraud Rate by Product Category (ProductCD)")
            fraud_by_product = (
                df.groupby("ProductCD")["isFraud"]
                .mean()
                .reset_index()
                .sort_values("isFraud", ascending=False)
            )
            fig_prod = px.bar(
                fraud_by_product,
                x="ProductCD",
                y="isFraud",
                labels={"isFraud": "Fraud Rate"},
            )
            fig_prod.update_layout(margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_prod, use_container_width=True)
        else:
            st.info("Columns 'ProductCD' or 'isFraud' missing.")

    st.markdown("---")

    # ---- Row 3: Card type & Device type ----
    st.markdown("### Card and Device Risk Overview")
    col3, col4 = st.columns(2)

    with col3:
        if "card4" in df.columns and "isFraud" in df.columns:
            st.caption("Fraud Rate by card4 (Card Network / Type)")
            card4_fraud = (
                df.groupby("card4")["isFraud"]
                .mean()
                .reset_index()
                .dropna()
            )
            fig_c4 = px.bar(
                card4_fraud,
                x="card4",
                y="isFraud",
                labels={"isFraud": "Fraud Rate", "card4": "Card Type"},
            )
            fig_c4.update_layout(margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_c4, use_container_width=True)
        else:
            st.info("Column 'card4' not available.")

    with col4:
        if "DeviceType" in df.columns and "isFraud" in df.columns:
            st.caption("Fraud Rate by Device Type")
            dev_fraud = (
                df.groupby("DeviceType")["isFraud"]
                .mean()
                .reset_index()
                .dropna()
            )
            fig_dev = px.bar(
                dev_fraud,
                x="DeviceType",
                y="isFraud",
                labels={"isFraud": "Fraud Rate"},
            )
            fig_dev.update_layout(margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_dev, use_container_width=True)
        else:
            st.info("Column 'DeviceType' not available.")

    st.markdown("---")

    # ---- Row 4 & 5: C / D / M engineered risk features (compact + minimal) ----
    st.markdown("### Engineered Risk Features (C / D / M)")

    color_map = ["#6a8caf", "#e27c7c"]  # soft blue + soft red
    small_height = 250  # compact charts

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # C_extreme_risk_flag
    with row1_col1:
        if "C_extreme_risk_flag" in df.columns:
            st.caption("C_extreme_risk_flag")
            c_ext = df.groupby("C_extreme_risk_flag")["isFraud"].mean().reset_index()
            c_ext["C_extreme_risk_flag"] = c_ext["C_extreme_risk_flag"].map(
                {0: "No Extreme Risk", 1: "Extreme Risk"}
            )
            fig_c = px.bar(
                c_ext,
                x="C_extreme_risk_flag",
                y="isFraud",
                color="C_extreme_risk_flag",
                color_discrete_sequence=color_map,
                labels={"isFraud": "Fraud Rate", "C_extreme_risk_flag": "Flag"},
                height=small_height
            )
            fig_c.update_layout(margin=dict(l=0, r=0, t=25, b=0), showlegend=False)
            st.plotly_chart(fig_c, use_container_width=True)

    # D_extreme_risk_flag
    with row1_col2:
        if "D_extreme_risk_flag" in df.columns:
            st.caption("D_extreme_risk_flag")
            d_ext = df.groupby("D_extreme_risk_flag")["isFraud"].mean().reset_index()
            d_ext["D_extreme_risk_flag"] = d_ext["D_extreme_risk_flag"].map(
                {0: "No Extreme Risk", 1: "Extreme Risk"}
            )
            fig_d1 = px.bar(
                d_ext,
                x="D_extreme_risk_flag",
                y="isFraud",
                color="D_extreme_risk_flag",
                color_discrete_sequence=color_map,
                labels={"isFraud": "Fraud Rate", "D_extreme_risk_flag": "Flag"},
                height=small_height
            )
            fig_d1.update_layout(margin=dict(l=0, r=0, t=25, b=0), showlegend=False)
            st.plotly_chart(fig_d1, use_container_width=True)

    # D_new_account
    with row2_col1:
        if "D_new_account" in df.columns:
            st.caption("D_new_account")
            d_new = df.groupby("D_new_account")["isFraud"].mean().reset_index()
            d_new["D_new_account"] = d_new["D_new_account"].map(
                {0: "Existing Account", 1: "New Account"}
            )
            fig_d2 = px.bar(
                d_new,
                x="D_new_account",
                y="isFraud",
                color="D_new_account",
                color_discrete_sequence=color_map,
                labels={"isFraud": "Fraud Rate", "D_new_account": "Account Type"},
                height=small_height
            )
            fig_d2.update_layout(margin=dict(l=0, r=0, t=25, b=0), showlegend=False)
            st.plotly_chart(fig_d2, use_container_width=True)

    # M_risk_category
    with row2_col2:
        if "M_risk_category" in df.columns:
            st.caption("M_risk_category")
            m_risk = df.groupby("M_risk_category")["isFraud"].mean().reset_index()
            fig_m = px.bar(
                m_risk,
                x="M_risk_category",
                y="isFraud",
                color="M_risk_category",
                color_discrete_sequence=px.colors.sequential.Plasma_r,
                labels={"isFraud": "Fraud Rate"},
                height=small_height
            )
            fig_m.update_layout(margin=dict(l=0, r=0, t=25, b=0), showlegend=False)
            st.plotly_chart(fig_m, use_container_width=True)

# ================================================
# PAGE 3: Model Performance
# ================================================
elif page == "Model Performance":
    st.markdown("## Model Performance – LightGBM on Vesta Data")

    st.markdown("### Validation Performance")

    # Probabilities
    y_proba = model.predict_proba(X_valid_enc)[:, 1]
    roc = roc_auc_score(y_valid, y_proba)

    # Fixed threshold for recall and confusion matrix
    threshold = 0.30
    y_pred = (y_proba >= threshold).astype(int)
    rec = recall_score(y_valid, y_pred, zero_division=0)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Validation ROC–AUC", f"{roc:.3f}")
    with c2:
        st.metric(f"Recall (threshold = {threshold:.2f})", f"{rec:.3f}")

    st.markdown(
        """
        ROC–AUC summarizes how well the model separates fraudulent and non‑fraudulent
        transactions across all possible thresholds.  
        Recall here is computed at a fixed probability threshold of 0.30.
        """
    )

    st.markdown("---")

    # ---- ROC curve + Confusion matrix side by side ----
    st.markdown("### ROC Curve and Confusion Matrix")

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_valid, y_proba)
    roc_df = pd.DataFrame({
        "False Positive Rate": fpr,
        "True Positive Rate": tpr
    })

    fig_roc = px.area(
        roc_df,
        x="False Positive Rate",
        y="True Positive Rate",
        title="ROC Curve",
        labels={
            "False Positive Rate": "False Positive Rate",
            "True Positive Rate": "True Positive Rate"
        },
        range_x=[0, 1],
        range_y=[0, 1],
        height=450
    )
    fig_roc.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(dash="dash", color="gray")
    )
    fig_roc.update_layout(margin=dict(l=40, r=20, t=40, b=40))

    # Confusion matrix for same threshold
    cm = confusion_matrix(y_valid, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0 (Non‑Fraud)", "Actual 1 (Fraud)"],
        columns=["Pred 0 (Non‑Fraud)", "Pred 1 (Fraud)"]
    )
    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        title="Confusion Matrix (threshold = 0.30)",
        height=450
    )
    fig_cm.update_layout(
        xaxis_title="Predicted label",
        yaxis_title="True label",
        margin=dict(l=40, r=20, t=40, b=40)
    )

    col_roc, col_cm = st.columns(2)
    with col_roc:
        st.plotly_chart(fig_roc, use_container_width=True)
    with col_cm:
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.markdown("### Note:")
    st.write(
        "This view combines a global ranking metric (ROC–AUC) with recall and a "
        "confusion matrix at a fixed threshold of 0.30, giving a clear picture of "
        "how well the model detects fraud cases."
    )
