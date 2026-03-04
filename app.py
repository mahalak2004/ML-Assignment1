import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    silhouette_score
)

st.set_page_config(page_title="Customer Segmentation ML App", layout="wide")

st.title("🛍️ Customer Segmentation using RFM + ML")

# ===============================
# FILE UPLOADER
# ===============================
uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx"],
    key="dataset_uploader"
)

if uploaded_file is not None:

    # ===============================
    # READ FILE
    # ===============================
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")

    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    # ===============================
    # RFM CALCULATION
    # ===============================
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm.dropna()

    st.subheader("📊 RFM Table")
    st.dataframe(rfm.head())

    # ===============================
    # SCALING
    # ===============================
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # ===============================
    # SELECT NUMBER OF CLUSTERS
    # ===============================
    st.subheader("🔢 Select Number of Clusters")
    n_clusters = st.slider("Choose Clusters", 2, 10, 4)

    # ===============================
    # ELBOW METHOD
    # ===============================
    st.subheader("📉 Elbow Method")
    inertia = []
    K = range(2, 11)

    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(rfm_scaled)
        inertia.append(km.inertia_)

    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(K, inertia, marker='o')
    ax_elbow.set_xlabel("Number of Clusters")
    ax_elbow.set_ylabel("Inertia")
    st.pyplot(fig_elbow)

    # ===============================
    # KMEANS
    # ===============================
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm['KMeans_Cluster'] = kmeans.fit_predict(rfm_scaled)

    kmeans_score = silhouette_score(rfm_scaled, rfm['KMeans_Cluster'])

    st.subheader("🔵 KMeans Results")
    st.metric("Silhouette Score (KMeans)", round(kmeans_score, 4))

    fig_km, ax_km = plt.subplots()
    sns.scatterplot(
        x=rfm['Recency'],
        y=rfm['Monetary'],
        hue=rfm['KMeans_Cluster'],
        palette="Set2",
        ax=ax_km
    )
    ax_km.set_title("KMeans Clusters")
    st.pyplot(fig_km)

    # ===============================
    # AGGLOMERATIVE
    # ===============================
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    rfm['Agglomerative_Cluster'] = agg.fit_predict(rfm_scaled)

    agg_score = silhouette_score(rfm_scaled, rfm['Agglomerative_Cluster'])

    st.subheader("🟣 Agglomerative Results")
    st.metric("Silhouette Score (Agglomerative)", round(agg_score, 4))

    fig_agg, ax_agg = plt.subplots()
    sns.scatterplot(
        x=rfm['Recency'],
        y=rfm['Monetary'],
        hue=rfm['Agglomerative_Cluster'],
        palette="Set1",
        ax=ax_agg
    )
    ax_agg.set_title("Agglomerative Clusters")
    st.pyplot(fig_agg)

    # ===============================
    # SILHOUETTE COMPARISON
    # ===============================
    st.subheader("📊 Silhouette Score Comparison")

    fig_score, ax_score = plt.subplots()
    ax_score.bar(["KMeans", "Agglomerative"], [kmeans_score, agg_score])
    ax_score.set_ylabel("Silhouette Score")
    st.pyplot(fig_score)

    # ===============================
    # SUPERVISED LEARNING
    # ===============================
    X = rfm[['Recency', 'Frequency', 'Monetary']]
    y = rfm['KMeans_Cluster']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # ===============================
    # METRICS
    # ===============================
    st.subheader("📈 Supervised Model Evaluation")

    st.markdown("### 🔹 Logistic Regression")
    st.write("Accuracy:", round(accuracy_score(y_test, y_pred_lr), 4))
    st.write("Precision:", round(precision_score(y_test, y_pred_lr, average='weighted'), 4))
    st.write("Recall:", round(recall_score(y_test, y_pred_lr, average='weighted'), 4))
    st.write("F1 Score:", round(f1_score(y_test, y_pred_lr, average='weighted'), 4))

    fig1, ax1 = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_lr)).plot(ax=ax1)
    st.pyplot(fig1)

    st.markdown("### 🔹 Random Forest")
    st.write("Accuracy:", round(accuracy_score(y_test, y_pred_rf), 4))
    st.write("Precision:", round(precision_score(y_test, y_pred_rf, average='weighted'), 4))
    st.write("Recall:", round(recall_score(y_test, y_pred_rf, average='weighted'), 4))
    st.write("F1 Score:", round(f1_score(y_test, y_pred_rf, average='weighted'), 4))

    fig2, ax2 = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf)).plot(ax=ax2)
    st.pyplot(fig2)

    # ===============================
    # ACCURACY COMPARISON
    # ===============================
    st.subheader("📊 Model Accuracy Comparison")

    fig_acc, ax_acc = plt.subplots()
    ax_acc.bar(
        ["Logistic Regression", "Random Forest"],
        [accuracy_score(y_test, y_pred_lr),
         accuracy_score(y_test, y_pred_rf)]
    )
    ax_acc.set_ylim(0, 1)
    st.pyplot(fig_acc)

    # ===============================
    # PREDICTION
    # ===============================
    st.subheader("🔮 Predict Customer Segment")

    recency_input = st.number_input("Recency", min_value=0)
    frequency_input = st.number_input("Frequency", min_value=0)
    monetary_input = st.number_input("Monetary", min_value=0.0)

    if st.button("Predict Segment"):

        new_data = pd.DataFrame({
            'Recency': [recency_input],
            'Frequency': [frequency_input],
            'Monetary': [monetary_input]
        })

        lr_pred = lr.predict(new_data)[0]
        rf_pred = rf.predict(new_data)[0]

        col1, col2 = st.columns(2)
        col1.success(f"Logistic Regression Prediction: Cluster {lr_pred}")
        col2.success(f"Random Forest Prediction: Cluster {rf_pred}")

else:
    st.info("👆 Please upload a dataset to begin.")