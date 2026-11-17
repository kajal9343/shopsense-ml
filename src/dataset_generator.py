import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from src.utils import get_logger, DATA_DIR

logger = get_logger("DatasetGenerator")

def generate_shopsense_dataset(
    n_samples: int = 25000,
    n_customers: int = 4500,
    random_seed: int = 42,
    save_path: Path = None
) -> pd.DataFrame:
    """
    Generates a high-fidelity, realistic, noisy e-commerce transactional dataset
    reflecting complete customer journey dynamics.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    logger.info(f"Generating synthetic dataset with {n_samples} interactions across {n_customers} unique customers...")

    # Customer Base profiles
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
    cust_tenure = {cid: np.random.randint(1, 48) for cid in customer_ids} # months
    cust_loyalty = {
        cid: np.random.choice(["Bronze", "Silver", "Gold", "Platinum"], p=[0.50, 0.28, 0.16, 0.06])
        for cid in customer_ids
    }
    cust_base_city_tier = {
        cid: np.random.choice(["Tier-1", "Tier-2", "Tier-3"], p=[0.42, 0.38, 0.20])
        for cid in customer_ids
    }
    cust_base_region = {
        cid: np.random.choice(["North", "South", "East", "West", "Central"], p=[0.28, 0.26, 0.16, 0.20, 0.10])
        for cid in customer_ids
    }
    # Customer baseline return risk inclination (unobserved heterogeneity)
    cust_return_tendency = {cid: np.clip(np.random.beta(2, 8), 0.02, 0.65) for cid in customer_ids}

    # Categories and pricing priors
    categories = [
        "Fashion & Apparel",
        "Electronics & Gadgets",
        "Home & Kitchen",
        "Beauty & Health",
        "Sports & Fitness",
        "Books & Media"
    ]
    cat_base_prices = {
        "Fashion & Apparel": (25.0, 180.0, 65.0),       # min, max, median
        "Electronics & Gadgets": (40.0, 1200.0, 240.0),
        "Home & Kitchen": (15.0, 450.0, 85.0),
        "Beauty & Health": (10.0, 110.0, 32.0),
        "Sports & Fitness": (20.0, 320.0, 75.0),
        "Books & Media": (8.0, 75.0, 22.0)
    }

    carriers = ["SwiftExpress", "BlueDart", "Delhivery", "SpeedPost", "ApexLogistics"]
    payment_methods = ["UPI / Digital Wallet", "Credit Card", "Cash on Delivery (COD)", "Debit Card", "BNPL (Pay Later)"]
    devices = ["Mobile App", "Mobile Web", "Desktop Web", "Tablet"]

    start_date = datetime(2025, 1, 1)
    
    rows = []
    
    # Track historical orders per customer
    cust_order_history = {cid: {"orders": 0, "returns": 0} for cid in customer_ids}

    for i in range(n_samples):
        trans_id = f"TXN_{i+1:07d}"
        cid = np.random.choice(customer_ids)
        session_id = f"SES_{i+1:07d}"
        
        # Temporal aspect
        days_offset = np.random.randint(0, 365)
        minute_offset = np.random.randint(0, 1440)
        txn_timestamp = start_date + timedelta(days=days_offset, minutes=minute_offset)
        
        # Category selection
        category = np.random.choice(
            categories, 
            p=[0.32, 0.24, 0.18, 0.12, 0.09, 0.05]
        )
        
        # Pricing & discount logic
        min_p, max_p, med_p = cat_base_prices[category]
        base_price = round(float(np.random.lognormal(mean=np.log(med_p), sigma=0.45)), 2)
        base_price = float(np.clip(base_price, min_p, max_p))
        
        # Discount policy: Fashion & Electronics frequently discounted
        if category in ["Fashion & Apparel", "Electronics & Gadgets"]:
            discount_pct = float(np.random.choice([0, 10, 15, 20, 25, 30, 40, 50], p=[0.15, 0.15, 0.20, 0.20, 0.12, 0.10, 0.05, 0.03]))
        else:
            discount_pct = float(np.random.choice([0, 5, 10, 15, 20, 25, 30], p=[0.30, 0.25, 0.20, 0.12, 0.08, 0.03, 0.02]))
            
        final_price = round(base_price * (1.0 - discount_pct / 100.0), 2)
        
        # Engagement & Browsing Journey
        device = np.random.choice(devices, p=[0.62, 0.18, 0.16, 0.04])
        visits_last_30d = int(np.random.poisson(lam=6) + 1)
        session_duration_min = round(float(np.random.gamma(shape=2.5, scale=4.0)), 1)
        page_views = max(1, int(session_duration_min * np.random.uniform(0.8, 2.2) + np.random.randint(1, 5)))
        cart_additions = int(np.clip(np.random.poisson(lam=1.8) + 1, 1, 9))
        
        # Cart abandonment & purchase conversion probability
        # Highly engaged + high discount + low base price = higher purchase probability
        conversion_logit = -0.5 + (session_duration_min * 0.03) + (page_views * 0.02) + (discount_pct * 0.02) - (final_price * 0.001)
        conversion_prob = 1 / (1 + np.exp(-conversion_logit))
        cart_abandoned = 1 if np.random.rand() > conversion_prob else 0
        
        # If cart abandoned, no fulfillment occurs
        if cart_abandoned:
            rows.append({
                "transaction_id": trans_id,
                "customer_id": cid,
                "session_id": session_id,
                "timestamp": txn_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "device_type": device,
                "website_visits_30d": visits_last_30d,
                "session_duration_min": session_duration_min,
                "page_views": page_views,
                "cart_additions": cart_additions,
                "cart_abandoned": 1,
                "order_completed": 0,
                "category": category,
                "base_price": base_price,
                "discount_percent": discount_pct,
                "final_price": final_price,
                "payment_method": "None",
                "city_tier": cust_base_city_tier[cid],
                "region": cust_base_region[cid],
                "carrier": "None",
                "estimated_delivery_days": np.nan,
                "actual_delivery_days": np.nan,
                "delivery_delay_days": np.nan,
                "delivery_ontime": np.nan,
                "customer_rating": np.nan,
                "review_sentiment": "None",
                "is_returned": 0,
                "return_reason": "None",
                "return_cost": 0.0,
                "customer_tenure_months": cust_tenure[cid],
                "loyalty_tier": cust_loyalty[cid],
                "prior_orders": cust_order_history[cid]["orders"],
                "prior_return_ratio": (
                    cust_order_history[cid]["returns"] / cust_order_history[cid]["orders"]
                    if cust_order_history[cid]["orders"] > 0 else 0.0
                )
            })
            continue

        # Completed Orders Journey
        cust_order_history[cid]["orders"] += 1
        city_tier = cust_base_city_tier[cid]
        region = cust_base_region[cid]
        
        # Payment method distribution depends on city tier
        if city_tier == "Tier-3":
            payment_p = [0.24, 0.14, 0.46, 0.10, 0.06] # High COD in Tier-3
        elif city_tier == "Tier-2":
            payment_p = [0.36, 0.22, 0.25, 0.10, 0.07]
        else: # Tier-1
            payment_p = [0.45, 0.32, 0.10, 0.06, 0.07] # Low COD in Tier-1
            
        payment_method = np.random.choice(payment_methods, p=payment_p)
        carrier = np.random.choice(carriers, p=[0.30, 0.25, 0.22, 0.13, 0.10])
        
        # Delivery SLA & Performance
        base_est_days = 2 if city_tier == "Tier-1" else (4 if city_tier == "Tier-2" else 6)
        estimated_delivery_days = base_est_days + np.random.choice([0, 1])
        
        # Carrier reliability factor
        carrier_perf = {
            "SwiftExpress": 0.92,
            "BlueDart": 0.90,
            "Delhivery": 0.85,
            "ApexLogistics": 0.78,
            "SpeedPost": 0.70
        }
        is_delayed_prob = (1.0 - carrier_perf[carrier]) + (0.08 if city_tier == "Tier-3" else 0.0)
        
        if np.random.rand() < is_delayed_prob:
            delay = int(np.random.choice([1, 2, 3, 4, 6], p=[0.45, 0.25, 0.15, 0.10, 0.05]))
            actual_delivery_days = estimated_delivery_days + delay
        else:
            delay = 0
            # sometimes early delivery
            early = 1 if np.random.rand() < 0.20 and estimated_delivery_days > 2 else 0
            actual_delivery_days = estimated_delivery_days - early

        delivery_delay_days = max(0, actual_delivery_days - estimated_delivery_days)
        delivery_ontime = 1 if delivery_delay_days == 0 else 0

        # Return Dynamics & Likelihood
        # Real-world drivers of return:
        # 1. Category (Fashion has highest returns due to sizing)
        # 2. Payment method (COD orders have high impulse rejection)
        # 3. Delivery delay (late orders trigger cancellations/returns)
        # 4. Customer inherent tendency
        # 5. Discount depth (bargain hunting impulse)
        cat_return_bias = {
            "Fashion & Apparel": 0.24,
            "Electronics & Gadgets": 0.14,
            "Home & Kitchen": 0.11,
            "Beauty & Health": 0.05,
            "Sports & Fitness": 0.09,
            "Books & Media": 0.03
        }
        
        return_score = (
            cat_return_bias[category] * 0.40 +
            (0.18 if payment_method == "Cash on Delivery (COD)" else 0.04) +
            (min(delivery_delay_days, 5) * 0.05) +
            (cust_return_tendency[cid] * 0.35) +
            (0.04 if discount_pct >= 40 else 0.0)
        )
        # Add random noise
        return_prob = np.clip(return_score + np.random.normal(0, 0.06), 0.01, 0.95)
        is_returned = 1 if np.random.rand() < return_prob else 0

        # Return reason attribution
        if is_returned:
            cust_order_history[cid]["returns"] += 1
            if category == "Fashion & Apparel":
                reason_choices = ["Sizing / Fit Issue", "Defective / Damaged", "Delayed Delivery", "Item Not as Described", "Impulse Buy / Regret"]
                reason_weights = [0.55, 0.10, 0.12, 0.13, 0.10]
            elif delivery_delay_days >= 2:
                reason_choices = ["Delayed Delivery", "Found Better Price Elsewhere", "Defective / Damaged", "Impulse Buy / Regret"]
                reason_weights = [0.60, 0.15, 0.15, 0.10]
            elif category == "Electronics & Gadgets":
                reason_choices = ["Defective / Damaged", "Item Not as Described", "Found Better Price Elsewhere", "Impulse Buy / Regret"]
                reason_weights = [0.42, 0.28, 0.15, 0.15]
            else:
                reason_choices = ["Item Not as Described", "Defective / Damaged", "Impulse Buy / Regret", "Found Better Price Elsewhere"]
                reason_weights = [0.35, 0.30, 0.20, 0.15]
                
            return_reason = str(np.random.choice(reason_choices, p=reason_weights))
            # Reverse logistics cost: shipping return ($6-$14) + inspection/restocking ($3-$8)
            return_cost = round(float(np.random.uniform(9.0, 22.0) + (final_price * 0.03)), 2)
        else:
            return_reason = "None"
            return_cost = 0.0

        # Customer Rating & Sentiment (1 - 5 stars)
        # Governed by: on-time delivery, return occurrence, product quality, discount delight
        rating_score = 4.5
        if delivery_delay_days > 0:
            rating_score -= (delivery_delay_days * 0.65)
        if is_returned:
            rating_score -= 1.8
        if discount_pct >= 25:
            rating_score += 0.35
        if payment_method == "Cash on Delivery (COD)":
            rating_score -= 0.15
            
        rating_score = np.clip(rating_score + np.random.normal(0, 0.45), 1.0, 5.0)
        customer_rating = int(np.round(rating_score))

        # Sentiment mapping
        if customer_rating >= 4:
            review_sentiment = "Positive"
        elif customer_rating == 3:
            review_sentiment = "Neutral"
        else:
            review_sentiment = "Negative"

        # Realistic noise: ~12% of customers skip rating
        if np.random.rand() < 0.12:
            customer_rating = np.nan
            review_sentiment = "Missing"

        rows.append({
            "transaction_id": trans_id,
            "customer_id": cid,
            "session_id": session_id,
            "timestamp": txn_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "device_type": device,
            "website_visits_30d": visits_last_30d,
            "session_duration_min": session_duration_min,
            "page_views": page_views,
            "cart_additions": cart_additions,
            "cart_abandoned": 0,
            "order_completed": 1,
            "category": category,
            "base_price": base_price,
            "discount_percent": discount_pct,
            "final_price": final_price,
            "payment_method": payment_method,
            "city_tier": city_tier,
            "region": region,
            "carrier": carrier,
            "estimated_delivery_days": estimated_delivery_days,
            "actual_delivery_days": actual_delivery_days,
            "delivery_delay_days": delivery_delay_days,
            "delivery_ontime": delivery_ontime,
            "customer_rating": customer_rating,
            "review_sentiment": review_sentiment,
            "is_returned": is_returned,
            "return_reason": return_reason,
            "return_cost": return_cost,
            "customer_tenure_months": cust_tenure[cid],
            "loyalty_tier": cust_loyalty[cid],
            "prior_orders": cust_order_history[cid]["orders"],
            "prior_return_ratio": (
                cust_order_history[cid]["returns"] / cust_order_history[cid]["orders"]
                if cust_order_history[cid]["orders"] > 0 else 0.0
            )
        })

    df = pd.DataFrame(rows)
    
    # Save dataset if path provided
    if save_path is None:
        save_path = DATA_DIR / "raw_ecommerce_journey.csv"
        
    df.to_csv(save_path, index=False)
    logger.info(f"Dataset successfully created and saved to {save_path} with shape {df.shape}")
    
    # Also generate customer aggregate features for RFM & Segmentation modeling
    generate_customer_aggregates(df)
    
    return df

def generate_customer_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transactional journeys into customer-level RFM-E (Recency, Frequency, Monetary, Engagement)
    and behavioral metrics for segmentation.
    """
    logger.info("Computing customer-level behavioral aggregates...")
    
    completed = df[df["order_completed"] == 1].copy()
    
    # Customer aggregates
    cust_grp = completed.groupby("customer_id")
    
    agg = cust_grp.agg(
        total_orders=("transaction_id", "count"),
        total_spend=("final_price", "sum"),
        avg_order_value=("final_price", "mean"),
        avg_discount_received=("discount_percent", "mean"),
        total_returns=("is_returned", "sum"),
        return_rate=("is_returned", "mean"),
        avg_delivery_delay=("delivery_delay_days", "mean"),
        avg_rating=("customer_rating", "mean"),
        avg_session_duration=("session_duration_min", "mean"),
        avg_page_views=("page_views", "mean"),
        avg_cart_additions=("cart_additions", "mean"),
        total_visits=("website_visits_30d", "mean"),
        tenure_months=("customer_tenure_months", "first"),
        loyalty_tier=("loyalty_tier", "first"),
        city_tier=("city_tier", "first"),
        region=("region", "first")
    ).reset_index()
    
    # Monetary Value Net of Returns
    agg["net_revenue"] = agg["total_spend"] - (agg["total_returns"] * 15.0)
    
    # Impute missing ratings with global mean
    agg["avg_rating"] = agg["avg_rating"].fillna(agg["avg_rating"].mean().round(2))
    
    agg_path = DATA_DIR / "customer_features_aggregated.csv"
    agg.to_csv(agg_path, index=False)
    logger.info(f"Customer aggregated dataset saved to {agg_path} with shape {agg.shape}")
    return agg

if __name__ == "__main__":
    generate_shopsense_dataset()
