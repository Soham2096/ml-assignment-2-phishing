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
    page_title="WebGuard Analytics Dashboard",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background-color: #f4f7fb;
    }

    /* Main content */
    .main {
        padding-top: 1rem;
    }

    /* Header */
    .dashboard-banner {
        padding: 24px 28px;
        border-radius: 14px;
        background: #ffffff;
        border: 1px solid #c7d2fe;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.08);
    }

    .dashboard-heading {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .dashboard-tagline {
        font-size: 16px;
        color: #475569;
    }

    /* Section headings */
    .panel-heading {
        font-size: 23px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    /* Information cards */
    .summary-tile {
        padding: 18px;
        border-radius: 12px;
        background: #ffffff;
        border: 1px solid #dbe3ef;
        min-height: 100px;
    }

    .tile-label {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 6px;
    }

    .tile-value {
        font-size: 25px;
        font-weight: 700;
    }

    /* Model verdict */
    .insight-box {
        padding: 18px 20px;
        border-radius: 12px;
        background: #eef2ff;
        border: 1px solid #6366f1;
        margin-top: 18px;
        margin-bottom: 20px;
    }

    .insight-box-title {
        font-size: 18px;
        font-weight: 650;
    }

    .insight-box-text {
        color: #475569;
        margin-top: 5px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
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
        <div class="dashboard-tagline">
            Intelligent Website Risk Analysis & Model Evaluation
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

    st.markdown("## Model Studio")

    selected_model = st.selectbox(
        "Select analysis model",
        list(MODEL_PATHS.keys())
    )

    st.markdown("---")

    st.markdown("### Prediction Labels")

    st.markdown(
        """
        **-1** → Suspicious / Phishing  
        **+1** → Safe / Legitimate
        """
    )

    st.markdown("---")

    st.caption(
        "Trained classifiers available for website risk analysis."
    )


# ============================================================
# DATA UPLOAD
# ============================================================

st.markdown(
    '<div class="panel-heading">1. Dataset Input</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Import evaluation CSV",
    type=["csv"],
    help="Upload the test dataset used for model evaluation."
)


if uploaded_file is None:

    st.info(
        "Import a CSV dataset to start the website risk analysis."
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
            <div class="tile-label">RECORDS</div>
            <div class="tile-value">{len(data):,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="summary-tile">
            <div class="tile-label">INPUT FEATURES</div>
            <div class="tile-value">{len(feature_columns)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="summary-tile">
            <div class="tile-label">NULL VALUES</div>
            <div class="tile-value">{data.isnull().sum().sum()}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="summary-tile">
            <div class="tile-label">ACTIVE MODEL</div>
            <div class="tile-value">{selected_model}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander("Preview evaluation data"):

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
# PREPARE INPUT FEATURES AND TARGET
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
    '<div class="panel-heading">2. Model Scorecard</div>',
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
        <div class="insight-heading">
            {selected_model} — Model Verdict
        </div>
        <div class="insight-copy">
            {verdict_text}
        </div>
        <div class="insight-copy">
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
    '<div class="panel-heading">3. Diagnostic View</div>',
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

    st.markdown("### Prediction Labels Report")

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
    '<div class="panel-heading">4. Risk Summary</div>',
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
        "Flagged Websites",
        phishing_count
    )

with summary2:

    st.metric(
        "Safe Websites",
        legitimate_count
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Web Guard Analytics | Machine Learning Evaluation"
)