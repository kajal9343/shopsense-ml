import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

from src.utils import get_logger, DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, save_model, save_json

logger = get_logger("MLPipeline")

def train_customer_segmentation():
    """
    Trains KMeans clustering on customer aggregated features (RFM + Engagement + Price Sensitivity)
    and maps actionable customer personas.
    """
    logger.info("Starting Customer Behavioral Segmentation training...")
    agg_path = DATA_DIR / "customer_features_aggregated.csv"
    if not agg_path.exists():
        raise FileNotFoundError(f"Customer aggregated data not found at {agg_path}")

    df_agg = pd.read_csv(agg_path)
    
    feature_cols = [
        "total_orders",
        "total_spend",
        "avg_order_value",
        "avg_discount_received",
        "return_rate",
        "avg_delivery_delay",
        "avg_rating",
        "avg_session_duration",
        "avg_page_views",
        "avg_cart_additions",
        "total_visits",
        "tenure_months"
    ]
    
    X = df_agg[feature_cols].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit KMeans (k=5)
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_agg["cluster"] = clusters
    
    # Fit PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df_agg["pca_x"] = pca_coords[:, 0]
    df_agg["pca_y"] = pca_coords[:, 1]
    
    # Characterize clusters to name personas
    cluster_profiles = df_agg.groupby("cluster")[feature_cols].mean()
    logger.info(f"Cluster profiles:\n{cluster_profiles}")

    # Persona mapping heuristic based on profiles
    # Find cluster with highest spend & low return -> Champions
    # Find cluster with highest return rate -> High Return Risk
    # Find cluster with highest discount -> Bargain Hunters
    # Find cluster with lowest orders & highest visits -> Casual Browsers
    # Remaining -> Steady Core
    persona_names = {}
    return_rates = cluster_profiles["return_rate"].to_dict()
    spends = cluster_profiles["total_spend"].to_dict()
    discounts = cluster_profiles["avg_discount_received"].to_dict()
    orders = cluster_profiles["total_orders"].to_dict()

    # Identify by dominant metric
    max_spend_cl = max(spends, key=spends.get)
    max_return_cl = max(return_rates, key=return_rates.get)
    max_disc_cl = max(discounts, key=discounts.get)
    min_orders_cl = min(orders, key=orders.get)

    assigned = set()
    persona_names[max_spend_cl] = "High-Value Loyal Champions"
    assigned.add(max_spend_cl)

    if max_return_cl not in assigned:
        persona_names[max_return_cl] = "High-Return Risk Shoppers"
        assigned.add(max_return_cl)

    if max_disc_cl not in assigned:
        persona_names[max_disc_cl] = "Discount-Driven Bargain Hunters"
        assigned.add(max_disc_cl)

    if min_orders_cl not in assigned:
        persona_names[min_orders_cl] = "Casual Window Browsers"
        assigned.add(min_orders_cl)

    for c in range(n_clusters):
        if c not in persona_names:
            persona_names[c] = "Steady Quality Buyers"

    df_agg["persona"] = df_agg["cluster"].map(persona_names)
    
    # Generate visualization
    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("tab10", n_clusters)
    for c in range(n_clusters):
        sub = df_agg[df_agg["cluster"] == c]
        plt.scatter(
            sub["pca_x"], sub["pca_y"],
            label=f"Cluster {c}: {persona_names[c]} ({len(sub):,} users)",
            alpha=0.6,
            s=40,
            color=palette[c]
        )
    plt.title("Customer Behavioral Segmentation (PCA Projection of 12 RFM-E Dimensions)", fontsize=13, weight="bold")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_customer_segments_pca.png", dpi=250)
    plt.close()

    # Save artifacts
    segmenter_bundle = {
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "feature_cols": feature_cols,
        "persona_map": persona_names,
        "cluster_stats": cluster_profiles.to_dict(orient="index")
    }
    save_model(segmenter_bundle, "customer_segmenter.joblib")
    
    # Save segmented dataframe
    df_agg.to_csv(DATA_DIR / "customer_features_with_segments.csv", index=False)
    logger.info("Customer segmentation artifacts and figures saved.")
    return segmenter_bundle

def train_return_prediction_model():
    """
    Trains a Gradient Boosting / Random Forest classifier to predict return probability
    from transactional and journey features.
    """
    logger.info("Training Return Risk Prediction Model...")
    raw_path = DATA_DIR / "raw_ecommerce_journey.csv"
    df = pd.read_csv(raw_path)
    
    # Filter completed orders only
    orders = df[df["order_completed"] == 1].copy()
    
    # Features
    categorical_features = ["category", "payment_method", "city_tier", "region", "device_type", "carrier"]
    numerical_features = [
        "final_price",
        "discount_percent",
        "session_duration_min",
        "page_views",
        "cart_additions",
        "estimated_delivery_days",
        "delivery_delay_days",
        "customer_tenure_months",
        "prior_orders",
        "prior_return_ratio"
    ]
    
    X = orders[categorical_features + numerical_features]
    y = orders["is_returned"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_split=8,
        random_state=42,
        class_weight="balanced"
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluation
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    auc = round(float(roc_auc_score(y_test, y_proba)), 4)
    clf_rep = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    logger.info(f"Return Prediction ROC-AUC: {auc:.4f}")
    
    # Feature Importance Extraction
    cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = numerical_features + encoded_cat_names
    importances = pipeline.named_steps["classifier"].feature_importances_
    
    feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
    
    # Plot top 12 features
    plt.figure(figsize=(10, 6))
    top_feat = feat_imp.head(12)
    sns.barplot(x=top_feat.values, y=top_feat.index, palette="viridis")
    plt.title("Key Drivers of Product Returns (Feature Importance)", fontsize=13, weight="bold")
    plt.xlabel("Importance Weight")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_return_model_feature_importance.png", dpi=250)
    plt.close()

    metrics = {
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": clf_rep,
        "top_features": feat_imp.head(10).to_dict()
    }
    
    save_model(pipeline, "return_prediction_model.joblib")
    save_json(metrics, "return_model_metrics.json")
    return pipeline, metrics

def train_csat_rating_model():
    """
    Trains a regressor to predict customer ratings (1 to 5) based on journey experience.
    """
    logger.info("Training CSAT Rating Prediction Model...")
    raw_path = DATA_DIR / "raw_ecommerce_journey.csv"
    df = pd.read_csv(raw_path)
    
    # Only completed orders with non-null ratings
    orders = df[(df["order_completed"] == 1) & (df["customer_rating"].notna())].copy()
    
    categorical_features = ["category", "payment_method", "city_tier", "carrier"]
    numerical_features = [
        "final_price",
        "discount_percent",
        "delivery_delay_days",
        "delivery_ontime",
        "is_returned",
        "customer_tenure_months"
    ]
    
    X = orders[categorical_features + numerical_features]
    y = orders["customer_rating"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    
    regressor = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
    mae = round(float(mean_absolute_error(y_test, y_pred)), 4)
    r2 = round(float(r2_score(y_test, y_pred)), 4)
    
    logger.info(f"CSAT Model Performance: RMSE={rmse}, MAE={mae}, R2={r2}")
    
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    save_model(pipeline, "csat_rating_model.joblib")
    save_json(metrics, "csat_model_metrics.json")
    return pipeline, metrics

def train_conversion_propensity_model():
    """
    Trains a model to predict cart-to-order conversion propensity.
    """
    logger.info("Training Purchase Conversion Propensity Model...")
    raw_path = DATA_DIR / "raw_ecommerce_journey.csv"
    df = pd.read_csv(raw_path)
    
    categorical_features = ["category", "device_type", "city_tier"]
    numerical_features = [
        "website_visits_30d",
        "session_duration_min",
        "page_views",
        "cart_additions",
        "base_price",
        "discount_percent"
    ]
    
    X = df[categorical_features + numerical_features]
    y = df["order_completed"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    
    clf = GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = round(float(roc_auc_score(y_test, y_proba)), 4)
    
    logger.info(f"Conversion Propensity ROC-AUC: {auc}")
    save_model(pipeline, "conversion_model.joblib")
    save_json({"roc_auc": auc}, "conversion_model_metrics.json")
    return pipeline, {"roc_auc": auc}

def run_all_training():
    """
    Executes training for all models and generates all artifacts.
    """
    train_customer_segmentation()
    train_return_prediction_model()
    train_csat_rating_model()
    train_conversion_propensity_model()
    logger.info("All Machine Learning pipelines successfully executed.")

if __name__ == "__main__":
    run_all_training()
