import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

from src.utils import DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, load_json, load_model
from src.optimizer import QuadGoalOptimizer

app = FastAPI(
    title="ShopSense - Beyond the Purchase AI Platform",
    description="Intelligent Customer-Behavior & Revenue-Optimization System",
    version="1.0.0"
)

# Template and static paths
WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

# Load models and reports on startup
optimizer_instance = None
return_model = None
csat_model = None
segmenter = None

def get_resources():
    global optimizer_instance, return_model, csat_model, segmenter
    if optimizer_instance is None:
        optimizer_instance = QuadGoalOptimizer()
    if return_model is None:
        return_model = load_model("return_prediction_model.joblib")
    if csat_model is None:
        csat_model = load_model("csat_rating_model.joblib")
    if segmenter is None:
        segmenter = load_model("customer_segmenter.joblib")
    return optimizer_instance, return_model, csat_model, segmenter

class PredictionInput(BaseModel):
    category: str
    payment_method: str
    city_tier: str
    region: str
    device_type: str
    carrier: str
    final_price: float
    discount_percent: float
    session_duration_min: float
    page_views: int
    cart_additions: int
    estimated_delivery_days: float
    delivery_delay_days: float
    customer_tenure_months: int
    prior_orders: int
    prior_return_ratio: float

class SimulationInput(BaseModel):
    cod_intervention_rate: float = 0.60
    fit_advisor_adoption: float = 0.65
    logistics_reroute_intensity: float = 0.75
    discount_strategy: str = "targeted"

@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    eda_summary = load_json("eda_summary.json")
    sim_summary = load_json("optimization_simulation.json")
    ret_metrics = load_json("return_model_metrics.json")
    csat_metrics = load_json("csat_model_metrics.json")
    
    # Load segment info
    _, _, _, seg = get_resources()
    persona_map = seg.get("persona_map", {})
    cluster_stats = seg.get("cluster_stats", {})

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "eda": eda_summary,
            "sim": sim_summary,
            "ret_metrics": ret_metrics,
            "csat_metrics": csat_metrics,
            "persona_map": persona_map,
            "cluster_stats": cluster_stats
        }
    )

@app.get("/api/kpis")
async def get_kpis():
    eda = load_json("eda_summary.json")
    return JSONResponse(content=eda.get("kpis", {}))

@app.get("/api/clusters")
async def get_clusters():
    _, _, _, seg = get_resources()
    return JSONResponse(content={
        "persona_map": seg.get("persona_map", {}),
        "cluster_stats": seg.get("cluster_stats", {})
    })

@app.post("/api/predict")
async def predict_journey(input_data: PredictionInput):
    _, ret_pipe, csat_pipe, _ = get_resources()
    
    df_in = pd.DataFrame([{
        "category": input_data.category,
        "payment_method": input_data.payment_method,
        "city_tier": input_data.city_tier,
        "region": input_data.region,
        "device_type": input_data.device_type,
        "carrier": input_data.carrier,
        "final_price": input_data.final_price,
        "discount_percent": input_data.discount_percent,
        "session_duration_min": input_data.session_duration_min,
        "page_views": input_data.page_views,
        "cart_additions": input_data.cart_additions,
        "estimated_delivery_days": input_data.estimated_delivery_days,
        "delivery_delay_days": input_data.delivery_delay_days,
        "customer_tenure_months": input_data.customer_tenure_months,
        "prior_orders": input_data.prior_orders,
        "prior_return_ratio": input_data.prior_return_ratio
    }])

    # Return prediction
    return_proba = float(ret_pipe.predict_proba(df_in)[0, 1])
    is_return_pred = int(return_proba > 0.45)
    
    # CSAT prediction input features
    df_csat_in = pd.DataFrame([{
        "category": input_data.category,
        "payment_method": input_data.payment_method,
        "city_tier": input_data.city_tier,
        "carrier": input_data.carrier,
        "final_price": input_data.final_price,
        "discount_percent": input_data.discount_percent,
        "delivery_delay_days": input_data.delivery_delay_days,
        "delivery_ontime": 1 if input_data.delivery_delay_days == 0 else 0,
        "is_returned": is_return_pred,
        "customer_tenure_months": input_data.customer_tenure_months
    }])
    
    pred_rating = float(csat_pipe.predict(df_csat_in)[0])
    pred_rating = round(np.clip(pred_rating, 1.0, 5.0), 2)

    # Actionable prescription
    prescriptions = []
    if return_proba > 0.50:
        if "COD" in input_data.payment_method:
            prescriptions.append("Trigger Smart Pre-pay Conversion: Offer 3% Instant UPI Discount or Mandatory Mobile OTP.")
        if input_data.category == "Fashion & Apparel":
            prescriptions.append("Active Size Verification: Trigger AI interactive sizing popup before dispatch.")
        if input_data.delivery_delay_days > 0 or input_data.estimated_delivery_days > 4:
            prescriptions.append("Logistics Upgrade: Reroute to SwiftExpress / BlueDart to prevent late-delivery cancellation.")
    else:
        prescriptions.append("Low Return Risk: Process standard automated fulfillment.")

    if pred_rating < 3.2:
        prescriptions.append("CSAT Risk Alert: Proactive VIP customer support check-in post-delivery.")

    return JSONResponse(content={
        "return_risk_probability": round(return_proba, 4),
        "return_risk_level": "High" if return_proba > 0.50 else ("Medium" if return_proba > 0.30 else "Low"),
        "predicted_rating": pred_rating,
        "prescriptions": prescriptions
    })

@app.post("/api/simulate")
async def simulate_quad_goal(sim_in: SimulationInput):
    opt, _, _, _ = get_resources()
    result = opt.simulate_policy(
        cod_intervention_rate=sim_in.cod_intervention_rate,
        fit_advisor_adoption=sim_in.fit_advisor_adoption,
        logistics_reroute_intensity=sim_in.logistics_reroute_intensity,
        discount_strategy=sim_in.discount_strategy
    )
    return JSONResponse(content=result)

@app.get("/api/customer/{cust_id}")
async def get_customer_profile(cust_id: str):
    seg_path = DATA_DIR / "customer_features_with_segments.csv"
    raw_path = DATA_DIR / "raw_ecommerce_journey.csv"
    if not seg_path.exists() or not raw_path.exists():
        return JSONResponse(status_code=404, content={"error": "Data not found"})
    
    df_seg = pd.read_csv(seg_path)
    cust_row = df_seg[df_seg["customer_id"].str.upper() == cust_id.upper()]
    if cust_row.empty:
        return JSONResponse(status_code=404, content={"error": f"Customer '{cust_id}' not found"})
    
    profile = cust_row.iloc[0].to_dict()
    # Replace NaN
    profile = {k: (None if pd.isna(v) else v) for k, v in profile.items()}
    
    df_raw = pd.read_csv(raw_path)
    orders = df_raw[df_raw["customer_id"].str.upper() == cust_id.upper()].sort_values("timestamp", ascending=False)
    order_list = orders.fillna("").head(8).to_dict(orient="records")
    
    return JSONResponse(content={
        "profile": profile,
        "recent_orders": order_list
    })

@app.get("/api/sample_customers")
async def get_sample_customers():
    seg_path = DATA_DIR / "customer_features_with_segments.csv"
    if not seg_path.exists():
        return JSONResponse(content=[])
    df_seg = pd.read_csv(seg_path)
    # Pick 1 representative from each persona
    samples = []
    for persona, grp in df_seg.groupby("persona"):
        cid = grp.iloc[0]["customer_id"]
        samples.append({"customer_id": cid, "persona": persona})
    return JSONResponse(content=samples)

