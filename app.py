import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WebGuard Analytics",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# THEME SELECTOR
# ============================================================

with st.sidebar:
    theme_choice = st.selectbox(
        "🎨 Appearance",
        ["System", "Light", "Dark"],
        index=0
    )


# ============================================================
# THEME STYLING
# ============================================================

if theme_choice == "Light":

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f8fafc;
            color: #172033;
        }

        section[data-testid="stSidebar"] {
            background-color: #eef2ff;
        }

        .dashboard-banner {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            padding: 26px 30px;
            border-radius: 18px;
            margin-bottom: 25px;
            box-shadow: 0 8px 24px rgba(79,70,229,0.18);
        }

        .dashboard-heading {
            color: white !important;
            font-size: 35px;
            font-weight: 750;
            margin-bottom: 5px;
        }

        .dashboard-subtitle {
            color: rgba(255,255,255,0.88) !important;
            font-size: 15px;
        }

        .section-title {
            font-size: 22px;
            font-weight: 700;
            margin-top: 22px;
            margin-bottom: 12px;
            color: #172033;
        }

        .summary-tile {
            padding: 18px;
            border-radius: 14px;
            background-color: #ffffff;
            color: #172033;
            border: 1px solid #dbe3f0;
            min-height: 100px;
            box-shadow: 0 4px 14px rgba(15,23,42,0.08);
        }

        .tile-label {
            font-size: 12px;
            font-weight: 650;
            color: #64748b;
            margin-bottom: 7px;
        }

        .tile-value {
            font-size: 24px;
            font-weight: 750;
            color: #172033;
        }

        .insight-box {
            padding: 18px 20px;
            border-radius: 14px;
            background-color: #eef2ff;
            border: 1px solid #c7d2fe;
            color: #172033;
            margin-top: 18px;
            margin-bottom: 22px;
        }

        .insight-title {
            font-size: 17px;
            font-weight: 700;
            color: #312e81;
        }

        .insight-text {
            color: #475569;
            margin-top: 6px;
            line-height: 1.5;
        }

        .stButton > button {
            border-radius: 10px;
            font-weight: 650;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

elif theme_choice == "Dark":

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0f172a;
            color: #f8fafc;
        }

        section[data-testid="stSidebar"] {
            background-color: #111827;
        }

        .dashboard-banner {
            background: linear-gradient(135deg, #4338ca, #6d28d9);
            padding: 26px 30px;
            border-radius: 18px;
            margin-bottom: 25px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.28);
        }

        .dashboard-heading {
            color: white !important;
            font-size: 35px;
            font-weight: 750;
            margin-bottom: 5px;
        }

        .dashboard-subtitle {
            color: rgba(255,255,255,0.88) !important;
            font-size: 15px;
        }

        .section-title {
            font-size: 22px;
            font-weight: 700;
            margin-top: 22px;
            margin-bottom: 12px;
            color: #f8fafc;
        }

        .summary-tile {
            padding: 18px;
            border-radius: 14px;
            background-color: #1e293b;
            color: #f8fafc;
            border: 1px solid #334155;
            min-height: 100px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
        }

        .tile-label {
            font-size: 12px;
            font-weight: 650;
            color: #94a3b8;
            margin-bottom: 7px;
        }

        .tile-value {
            font-size: 24px;
            font-weight: 750;
            color: #f8fafc;
        }

        .insight-box {
            padding: 18px 20px;
            border-radius: 14px;
            background-color: #1e1b4b;
            border: 1px solid #4338ca;
            color: #f8fafc;
            margin-top: 18px;
            margin-bottom: 22px;
        }

        .insight-title {
            font-size: 17px;
            font-weight: 700;
            color: #c4b5fd;
        }

        .insight-text {
            color: #cbd5e1;
            margin-top: 6px;
            line-height: 1.5;
        }

        .stButton > button {
            border-radius: 10px;
            font-weight: 650;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <style>
        .dashboard-banner {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            padding: 26px 30px;
            border-radius: 18px;
            margin-bottom: 25px;
            box-shadow: 0 8px 24px rgba(79,70,229,0.18);
        }

        .dashboard-heading {
            color: white !important;
            font-size: 35px;
            font-weight: 750;
            margin-bottom: 5px;
        }

        .dashboard-subtitle {
            color: rgba(255,255,255,0.88) !important;
            font-size: 15px;
        }

        .section-title {
            font-size: 22px;
            font-weight: 700;
            margin-top: 22px;
            margin-bottom: 12px;
        }

        .summary-tile {
            padding: 18px;
            border-radius: 14px;
            background-color: transparent;
            border: 1px solid rgba(100,116,139,0.30);
            min-height: 100px;
        }

        .tile-label {
            font-size: 12px;
            font-weight: 650;
            opacity: 0.68;
            margin-bottom: 7px;
        }

        .tile-value {
            font-size: 24px;
            font-weight: 750;
        }

        .insight-box {
            padding: 18px 20px;
            border-radius: 14px;
            background-color: rgba(79,70,229,0.08);
            border: 1px solid rgba(79,70,229,0.30);
            margin-top: 18px;
            margin-bottom: 22px;
        }

        .insight-title {
            font-size: 17px;
            font-weight: 700;
        }

        .insight-text {
            margin-top: 6px;
            opacity: 0.78;
            line-height: 1.5;
        }

        .stButton > button {
            border-radius: 10px;
            font-weight: 650;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-banner">
        <div class="dashboard-heading">🔎 WebGuard Analytics</div>
        <div class="dashboard-subtitle">
            Machine Learning Based Phishing Website Classification & Evaluation
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest": "model/random_forest.joblib"
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧠 Model Studio")

    selected_model = st.selectbox(
        "Select analysis model",
        list(MODEL_PATHS.keys())
    )

    st.markdown("---")

    st.markdown("### Classification")

    st.markdown(
        """
        **-1** → Phishing  
        **+1** → Legitimate
        """
    )

    st.markdown("---")

    st.caption(
        "Machine learning models trained for phishing website classification."
    )


# ============================================================
# DATA UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">1. Dataset Input</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"],
    help="Upload the test dataset used for model evaluation."
)


if uploaded_file is None:

    st.info(
        "Upload the test CSV file to evaluate the selected machine learning model."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    data = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(f"Unable to read the CSV file: {error}")
    st.stop()


# ============================================================
# DATA VALIDATION
# ============================================================

if "result" not in data.columns:

    st.error(
        "The uploaded dataset must contain the 'result' target column."
    )

    st.stop()


feature_columns = [
    column for column in data.columns
    if column != "result"
]


# ============================================================
# DATASET SUMMARY
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        f"""
        <div class="summary-tile">
            <div class="tile-label">TEST INSTANCES</div>
            <div class="tile-value">{len(data):,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="summary-tile">
            <div class="tile-label">FEATURES</div>
            <div class="tile-value">{len(feature_columns)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="summary-tile">
            <div class="tile-label">MISSING VALUES</div>
            <div class="tile-value">{data.isnull().sum().sum()}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="summary-tile">
            <div class="tile-label">SELECTED MODEL</div>
            <div class="tile-value">{selected_model}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander("📄 View uploaded dataset"):

    st.dataframe(
        data.head(10),
        width="stretch"
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(MODEL_PATHS[selected_model])

except Exception as error:

    st.error(
        f"Unable to load {selected_model}: {error}"
    )

    st.stop()


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

X_test = data[feature_columns]

y_test = data["result"]


# ============================================================
# PREDICTION
# ============================================================

try:

    predictions = model.predict(X_test)

except Exception as error:

    st.error(
        f"Prediction failed. Please check that the uploaded dataset "
        f"contains the correct 30 features.\n\n{error}"
    )

    st.stop()


# ============================================================
# PROBABILITY / AUC
# ============================================================

try:

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_test)[:, 1]

        auc_score = roc_auc_score(
            y_test,
            probabilities
        )

    else:

        auc_score = roc_auc_score(
            y_test,
            predictions
        )

except Exception:

    auc_score = 0.0


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label=1,
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    predictions
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">2. Model Scorecard</div>',
    unsafe_allow_html=True
)

metric1, metric2, metric3 = st.columns(3)

with metric1:

    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )


with metric2:

    st.metric(
        "AUC",
        f"{auc_score:.4f}"
    )


with metric3:

    st.metric(
        "F1 Score",
        f"{f1:.4f}"
    )


metric4, metric5, metric6 = st.columns(3)

with metric4:

    st.metric(
        "Precision",
        f"{precision:.4f}"
    )


with metric5:

    st.metric(
        "Recall",
        f"{recall:.4f}"
    )


with metric6:

    st.metric(
        "MCC",
        f"{mcc:.4f}"
    )


# ============================================================
# MODEL VERDICT
# ============================================================

correct_predictions = int(
    np.sum(predictions == y_test)
)

total_predictions = len(y_test)

if selected_model == "Random Forest":

    verdict_text = (
        "Random Forest achieved the strongest overall performance "
        "among the evaluated models in this experiment."
    )

elif selected_model == "Naive Bayes":

    verdict_text = (
        "Naive Bayes shows high precision but substantially lower "
        "recall compared with the other evaluated models."
    )

else:

    verdict_text = (
        f"{selected_model} provides a useful baseline for comparing "
        "classification performance on the phishing dataset."
    )


st.markdown(
    f"""
    <div class="insight-box">
        <div class="insight-title">
            {selected_model} — Model Verdict
        </div>
        <div class="insight-text">
            {verdict_text}
        </div>
        <div class="insight-text">
            Correct predictions: {correct_predictions:,} / {total_predictions:,}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.markdown(
    '<div class="section-title">3. Diagnostic View</div>',
    unsafe_allow_html=True
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[-1, 1]
)

left, right = st.columns([1, 1])


with left:

    st.markdown("### Confusion Matrix")

    fig, ax = plt.subplots(figsize=(5, 4))

    image = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Phishing (-1)", "Legitimate (+1)"]
    )

    ax.set_yticklabels(
        ["Phishing (-1)", "Legitimate (+1)"]
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_title(selected_model)

    for row in range(2):

        for column in range(2):

            ax.text(
                column,
                row,
                cm[row, column],
                ha="center",
                va="center",
                fontsize=14
            )

    fig.colorbar(image, ax=ax)

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

with right:

    st.markdown("### Classification Report")

    report = classification_report(
        y_test,
        predictions,
        labels=[-1, 1],
        target_names=[
            "Phishing (-1)",
            "Legitimate (+1)"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        width="stretch"
    )


# ============================================================
# PREDICTION SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">4. Risk Summary</div>',
    unsafe_allow_html=True
)

phishing_count = int(
    np.sum(predictions == -1)
)

legitimate_count = int(
    np.sum(predictions == 1)
)

summary1, summary2 = st.columns(2)

with summary1:

    st.metric(
        "Predicted Phishing",
        phishing_count
    )


with summary2:

    st.metric(
        "Predicted Legitimate",
        legitimate_count
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "WebGuard Analytics | Machine Model Learning Evaluation"
)
