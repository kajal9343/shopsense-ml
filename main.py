import argparse
import os
import sys
import uvicorn
from pathlib import Path

from src.utils import get_logger
from src.dataset_generator import generate_shopsense_dataset
from src.eda import run_exploratory_analysis
from src.ml_pipeline import run_all_training
from src.optimizer import run_optimization_study

logger = get_logger("ShopSense-Main", log_file="shopsense_main.log", log_level="INFO")
def run_full_pipeline():
    logger.info("==================================================================")
    logger.info("   SHOPSENSE: BEYOND THE PURCHASE - FULL PIPELINE EXECUTION       ")
    logger.info("==================================================================")
    
    # Step 1: Generate High-Fidelity Dataset
    logger.info("\n--- STEP 1: GENERATING TRANSACTIONAL JOURNEY DATASET ---")
    generate_shopsense_dataset()

    # Step 2: Exploratory Data Analysis & Diagnostic Figures
    logger.info("\n--- STEP 2: RUNNING EXPLORATORY DATA ANALYSIS & REPORTING ---")
    run_exploratory_analysis()

    # Step 3: Machine Learning Pipelines (Segmentation, Return Risk, CSAT, Conversion)
    logger.info("\n--- STEP 3: TRAINING MACHINE LEARNING MODELS & PERSONAS ---")
    run_all_training()

    # Step 4: Quad-Pillar Multi-Objective Revenue & Experience Optimization
    logger.info("\n--- STEP 4: EXECUTING QUAD-PILLAR OPTIMIZATION SIMULATION ---")
    run_optimization_study()

    logger.info("\n==================================================================")
    logger.info("   PIPELINE COMPLETED SUCCESSFULLY! ARTIFACTS SAVED IN /reports   ")
    logger.info("==================================================================")

def start_server(host: str = "127.0.0.1", port: int = 8000):
    import socket
    # Check for cloud environment variables (e.g. Render, Railway)
    env_port = os.environ.get("PORT")
    if env_port:
        port = int(env_port)
        host = os.environ.get("HOST", "0.0.0.0")

    # Check if port is already in use (only if running on localhost)
    if host in ["127.0.0.1", "localhost"]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            is_busy = s.connect_ex((host, port)) == 0
        
        if is_busy:
            logger.warning(f"Port {port} is already in use! ShopSense server might already be running at http://{host}:{port}")
            alt_port = port + 80 if port == 8000 else port + 1
            logger.info(f"Switching automatically to fallback port http://{host}:{alt_port}")
            port = alt_port

    logger.info(f"Starting ShopSense interactive web application on http://{host}:{port}")
    uvicorn.run("web.app:app", host=host, port=port, reload=False)

def main():
    parser = argparse.ArgumentParser(description="ShopSense - Beyond the Purchase: ML & Revenue Optimization")
    parser.add_argument("--pipeline", action="store_true", help="Execute complete end-to-end ML & Optimization pipeline")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic journey dataset only")
    parser.add_argument("--eda", action="store_true", help="Run exploratory analysis and generate figures")
    parser.add_argument("--train", action="store_true", help="Train ML models and customer personas")
    parser.add_argument("--simulate", action="store_true", help="Run Quad-Goal optimization simulation")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI web dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.pipeline:
        run_full_pipeline()
    elif args.generate:
        generate_shopsense_dataset()
    elif args.eda:
        run_exploratory_analysis()
    elif args.train:
        run_all_training()
    elif args.simulate:
        run_optimization_study()

    if args.serve:
        start_server(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
