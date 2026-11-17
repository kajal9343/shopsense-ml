import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils import get_logger, DATA_DIR, REPORTS_DIR, FIGURES_DIR, save_json

logger = get_logger("EDA")

# Apply styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.titleweight": "bold"
})

def run_exploratory_analysis() -> dict:
    """
    Executes end-to-end exploratory data analysis, computes key e-commerce KPIs,
    and generates high-resolution figures.
    """
    raw_path = DATA_DIR / "raw_ecommerce_journey.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found at {raw_path}")

    logger.info("Loading raw e-commerce dataset for EDA...")
    df = pd.read_csv(raw_path)
    
    completed = df[df["order_completed"] == 1].copy()
    abandoned = df[df["cart_abandoned"] == 1].copy()

    # 1. Funnel & High-level KPIs
    total_sessions = len(df)
    total_orders = len(completed)
    total_abandoned = len(abandoned)
    conversion_rate = round((total_orders / total_sessions) * 100, 2)
    
    total_gross_revenue = round(completed["final_price"].sum(), 2)
    total_discount_given = round((completed["base_price"] - completed["final_price"]).sum(), 2)
    total_returns = int(completed["is_returned"].sum())
    overall_return_rate = round((total_returns / total_orders) * 100, 2)
    total_return_costs = round(completed["return_cost"].sum(), 2)
    net_realized_revenue = round(total_gross_revenue - (completed[completed["is_returned"] == 1]["final_price"].sum()) - total_return_costs, 2)
    
    avg_order_value = round(completed["final_price"].mean(), 2)
    avg_csat = round(completed["customer_rating"].dropna().mean(), 2)
    avg_delivery_delay = round(completed["delivery_delay_days"].mean(), 2)
    ontime_delivery_rate = round((completed["delivery_ontime"].mean()) * 100, 2)

    logger.info(f"Summary: Sessions={total_sessions}, Conversion={conversion_rate}%, Orders={total_orders}, ReturnRate={overall_return_rate}%, CSAT={avg_csat}")

    # 2. Category Analytics
    cat_summary = completed.groupby("category").agg(
        orders=("transaction_id", "count"),
        gross_sales=("final_price", "sum"),
        avg_price=("final_price", "mean"),
        avg_discount=("discount_percent", "mean"),
        return_count=("is_returned", "sum"),
        return_rate=("is_returned", "mean"),
        avg_rating=("customer_rating", "mean")
    ).reset_index()
    cat_summary["return_rate"] = (cat_summary["return_rate"] * 100).round(2)
    cat_summary["avg_rating"] = cat_summary["avg_rating"].round(2)
    cat_summary["gross_sales"] = cat_summary["gross_sales"].round(2)
    cat_summary["avg_price"] = cat_summary["avg_price"].round(2)
    cat_summary["avg_discount"] = cat_summary["avg_discount"].round(2)

    # 3. Payment Method Return Analysis
    payment_summary = completed.groupby("payment_method").agg(
        orders=("transaction_id", "count"),
        gross_sales=("final_price", "sum"),
        returns=("is_returned", "sum"),
        return_rate=("is_returned", "mean")
    ).reset_index()
    payment_summary["return_rate"] = (payment_summary["return_rate"] * 100).round(2)
    payment_summary["share_pct"] = ((payment_summary["orders"] / total_orders) * 100).round(2)

    # 4. Carrier Logistics Performance
    carrier_summary = completed.groupby("carrier").agg(
        shipments=("transaction_id", "count"),
        avg_est_days=("estimated_delivery_days", "mean"),
        avg_actual_days=("actual_delivery_days", "mean"),
        avg_delay=("delivery_delay_days", "mean"),
        ontime_pct=("delivery_ontime", "mean"),
        avg_csat=("customer_rating", "mean")
    ).reset_index()
    carrier_summary["ontime_pct"] = (carrier_summary["ontime_pct"] * 100).round(2)
    carrier_summary["avg_delay"] = carrier_summary["avg_delay"].round(2)
    carrier_summary["avg_csat"] = carrier_summary["avg_csat"].round(2)

    # 5. Geographic Summary
    geo_summary = completed.groupby(["city_tier", "region"]).agg(
        orders=("transaction_id", "count"),
        gross_sales=("final_price", "sum"),
        return_rate=("is_returned", "mean"),
        avg_delay=("delivery_delay_days", "mean")
    ).reset_index()
    geo_summary["return_rate"] = (geo_summary["return_rate"] * 100).round(2)
    geo_summary["avg_delay"] = geo_summary["avg_delay"].round(2)

    # 6. Return Reasons Breakdown
    returned_items = completed[completed["is_returned"] == 1]
    reason_breakdown = returned_items["return_reason"].value_counts(normalize=True).mul(100).round(2).to_dict()

    # Compile structured summary
    eda_report = {
        "kpis": {
            "total_sessions": total_sessions,
            "total_orders": total_orders,
            "conversion_rate_pct": conversion_rate,
            "total_gross_revenue": total_gross_revenue,
            "net_realized_revenue": net_realized_revenue,
            "total_discount_given": total_discount_given,
            "total_returns": total_returns,
            "overall_return_rate_pct": overall_return_rate,
            "total_return_costs": total_return_costs,
            "avg_order_value": avg_order_value,
            "avg_csat_rating": avg_csat,
            "ontime_delivery_rate_pct": ontime_delivery_rate,
            "avg_delivery_delay_days": avg_delivery_delay
        },
        "category_metrics": cat_summary.to_dict(orient="records"),
        "payment_metrics": payment_summary.to_dict(orient="records"),
        "carrier_metrics": carrier_summary.to_dict(orient="records"),
        "return_reasons": reason_breakdown
    }

    save_json(eda_report, "eda_summary.json")
    
    # ---------------- GENERATE CHARTS ----------------
    logger.info("Generating publication-ready figures...")

    # Plot 1: Journey Funnel, Discount vs Conversion & Cart Behavior
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Funnel
    funnel_data = pd.Series({"Sessions / Visits": total_sessions, "Cart Additions": len(df[df["cart_additions"] > 0]), "Completed Orders": total_orders})
    sns.barplot(x=funnel_data.index, y=funnel_data.values, ax=axes[0], palette="Blues_d")
    axes[0].set_title("Customer Journey Conversion Funnel")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(funnel_data.values):
        axes[0].text(i, v + total_sessions*0.015, f"{v:,}\n({(v/total_sessions*100):.1f}%)", ha="center", fontweight="bold")
    axes[0].set_ylim(0, total_sessions * 1.15)

    # Discount vs Conversion Rate
    df["discount_bin"] = pd.cut(df["discount_percent"], bins=[-1, 5, 15, 25, 35, 60], labels=["0-5%", "6-15%", "16-25%", "26-35%", "36%+"])
    disc_conv = df.groupby("discount_bin", observed=True)["order_completed"].mean() * 100
    sns.lineplot(x=disc_conv.index.astype(str), y=disc_conv.values, ax=axes[1], marker="o", color="#2563eb", linewidth=2.5, markersize=8)
    axes[1].set_title("Discount Depth vs. Purchase Conversion Rate")
    axes[1].set_xlabel("Discount Tier")
    axes[1].set_ylabel("Conversion Rate (%)")
    for i, v in enumerate(disc_conv.values):
        axes[1].text(i, v + 0.8, f"{v:.1f}%", ha="center", fontweight="bold", color="#1d4ed8")
    axes[1].set_ylim(0, max(disc_conv.values) * 1.25)

    # Final Price Distribution by Category
    sns.boxplot(x="category", y="final_price", data=completed, ax=axes[2], palette="Set2")
    axes[2].set_title("Order Value Distribution by Category")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Final Price ($)")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "01_journey_funnel_and_discount.png", dpi=250)
    plt.close(fig)

    # Plot 2: Returns Dynamics by Payment Method & Category Reason
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Payment Method Return Rate
    pm_sorted = payment_summary.sort_values(by="return_rate", ascending=False)
    colors = ["#ef4444" if "COD" in p else "#3b82f6" for p in pm_sorted["payment_method"]]
    sns.barplot(x="return_rate", y="payment_method", data=pm_sorted, ax=axes[0], palette=colors)
    axes[0].set_title("Return & Rejection Rate by Payment Method (COD Hazard)")
    axes[0].set_xlabel("Return Rate (%)")
    axes[0].set_ylabel("")
    for i, v in enumerate(pm_sorted["return_rate"]):
        axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center", fontweight="bold")
    axes[0].set_xlim(0, max(pm_sorted["return_rate"]) * 1.2)

    # Return Reasons in Returned Orders
    reasons_df = pd.DataFrame(list(reason_breakdown.items()), columns=["Reason", "Percentage"])
    sns.barplot(x="Percentage", y="Reason", data=reasons_df, ax=axes[1], palette="crest_r")
    axes[1].set_title("Primary Root Causes of Product Returns")
    axes[1].set_xlabel("Share of Returns (%)")
    axes[1].set_ylabel("")
    for i, v in enumerate(reasons_df["Percentage"]):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontweight="bold")
    axes[1].set_xlim(0, max(reasons_df["Percentage"]) * 1.2)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "02_return_reasons_and_payment.png", dpi=250)
    plt.close(fig)

    # Plot 3: Delivery Delay vs CSAT Rating & Carrier Benchmarking
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Delivery delay impact on rating
    delay_rating = completed.groupby("delivery_delay_days", observed=True)["customer_rating"].mean().reset_index()
    delay_rating = delay_rating[delay_rating["delivery_delay_days"] <= 6]
    sns.lineplot(x="delivery_delay_days", y="customer_rating", data=delay_rating, ax=axes[0], marker="s", color="#dc2626", linewidth=3, markersize=8)
    axes[0].set_title("Logistics Friction: Delivery Delay vs. CSAT Rating")
    axes[0].set_xlabel("Delivery Delay (Days Beyond SLA)")
    axes[0].set_ylabel("Average Rating (1 to 5)")
    axes[0].set_ylim(1.0, 5.0)
    for i, row in delay_rating.iterrows():
        axes[0].text(row["delivery_delay_days"], row["customer_rating"] + 0.15, f"{row['customer_rating']:.2f}★", ha="center", fontweight="bold")

    # Carrier On-time Performance
    carrier_sorted = carrier_summary.sort_values(by="ontime_pct", ascending=False)
    sns.barplot(x="carrier", y="ontime_pct", data=carrier_sorted, ax=axes[1], palette="mako")
    axes[1].set_title("Logistics Carrier On-Time Performance Benchmark (%)")
    axes[1].set_xlabel("Logistics Partner")
    axes[1].set_ylabel("On-Time SLA Rate (%)")
    axes[1].set_ylim(0, 110)
    for i, v in enumerate(carrier_sorted["ontime_pct"]):
        axes[1].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "03_delivery_vs_ratings.png", dpi=250)
    plt.close(fig)

    # Plot 4: Geographic Patterns (City Tier vs Return Rate & Delays)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    tier_summary = completed.groupby("city_tier").agg(
        return_rate=("is_returned", "mean"),
        avg_delay=("delivery_delay_days", "mean")
    ).reset_index()
    tier_summary["return_rate"] = tier_summary["return_rate"] * 100

    sns.barplot(x="city_tier", y="return_rate", data=tier_summary, ax=axes[0], palette="coolwarm")
    axes[0].set_title("Return Rate Across Geographic City Tiers")
    axes[0].set_xlabel("City Tier")
    axes[0].set_ylabel("Return Rate (%)")
    for i, v in enumerate(tier_summary["return_rate"]):
        axes[0].text(i, v + 0.4, f"{v:.1f}%", ha="center", fontweight="bold")

    sns.barplot(x="city_tier", y="avg_delay", data=tier_summary, ax=axes[1], palette="flare")
    axes[1].set_title("Average Delivery Delay Across City Tiers (Days)")
    axes[1].set_xlabel("City Tier")
    axes[1].set_ylabel("Average Delay (Days)")
    for i, v in enumerate(tier_summary["avg_delay"]):
        axes[1].text(i, v + 0.05, f"{v:.2f} d", ha="center", fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04_geographic_insights.png", dpi=250)
    plt.close(fig)

    logger.info("All EDA figures and summaries created successfully.")
    return eda_report

if __name__ == "__main__":
    run_exploratory_analysis()
