#!/usr/bin/env python3
"""
Project Nova - Quick Start Script
Fast execution version of the ML pipeline for testing and development
"""

import os
import sys
import logging
import time
from datetime import datetime

# Add src to path
sys.path.append('src')

from main_training_pipeline import CreditScoringPipeline

def quick_start_pipeline():
    """Run a quick version of the pipeline for testing"""
    
    print("="*60)
    print("PROJECT NOVA - QUICK START")
    print("="*60)
    print("This is a simplified version for quick testing.")
    print("For full pipeline with hyperparameter tuning, run: python main_training_pipeline.py")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Create pipeline instance
        pipeline = CreditScoringPipeline()
        
        # Step 1: Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        pipeline.load_and_preprocess_data()
        print("   ✓ Data loaded successfully")
        
        # Step 2: Train models (without hyperparameter tuning for speed)
        print("\n2. Training models with default parameters...")
        pipeline.train_models(use_tuned_params=False)
        print(f"   ✓ Trained {len(pipeline.drivers_models)} drivers models and {len(pipeline.merchants_models)} merchants models")
        
        # Step 3: Evaluate models
        print("\n3. Evaluating models...")
        pipeline.evaluate_models()
        print("   ✓ Model evaluation completed")
        
        # Step 4: Basic fairness analysis
        print("\n4. Analyzing fairness...")
        try:
            pipeline.analyze_fairness()
            print("   ✓ Fairness analysis completed")
        except Exception as e:
            print(f"   ⚠ Fairness analysis failed: {str(e)}")
        
        # Step 5: Create basic visualizations
        print("\n5. Creating visualizations...")
        try:
            pipeline.create_visualizations()
            print("   ✓ Visualizations created")
        except Exception as e:
            print(f"   ⚠ Visualization creation failed: {str(e)}")
        
        # Step 6: Generate report
        print("\n6. Generating report...")
        pipeline.generate_final_report()
        print("   ✓ Final report generated")
        
        # Show results
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("\n" + "="*60)
        print("QUICK START COMPLETED SUCCESSFULLY!")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Models trained: {len(pipeline.trained_models)}")
        
        # Show best models
        if pipeline.drivers_evaluation:
            best_drivers = max(
                pipeline.drivers_evaluation.items(),
                key=lambda x: x[1].get('r2_score', 0)
            )
            print(f"Best drivers model: {best_drivers[0]}")
            print(f"Drivers R² Score: {best_drivers[1].get('r2_score', 0):.4f}")
        
        if pipeline.merchants_evaluation:
            best_merchants = max(
                pipeline.merchants_evaluation.items(),
                key=lambda x: x[1].get('r2_score', 0)
            )
            print(f"Best merchants model: {best_merchants[0]}")
            print(f"Merchants R² Score: {best_merchants[1].get('r2_score', 0):.4f}")
        
        print("\nResults saved in:")
        print("  - Models: models/saved_models/")
        print("  - Metrics: results/metrics/")
        print("  - Plots: results/plots/")
        print("  - Report: results/final_report.md")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Quick start failed: {str(e)}")
        print("Check logs/main_pipeline.log for details")
        return False
    
    return True

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    try:
        import subprocess
        import sys
        
        # Install packages
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to install requirements: {str(e)}")
        print("Please install manually: pip install -r requirements.txt")
        return False

def check_data_files():
    """Check if data files exist"""
    data_files = [
        "data/grab_drivers_dataset_refined_score.csv",
        "data/grab_merchants_dataset_refined_score.csv"
    ]
    
    missing_files = []
    for file_path in data_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing data files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✓ All data files found")
    return True

def main():
    """Main function"""
    
    print("Project Nova - Credit Scoring ML Pipeline")
    print("Quick Start Script")
    print("-" * 40)
    
    # Check data files
    if not check_data_files():
        print("\nPlease ensure data files are in the correct location.")
        return
    
    # Ask user for installation
    install_deps = input("\nInstall/update requirements? (y/n): ").lower().strip()
    if install_deps == 'y':
        if not install_requirements():
            return
    
    # Ask user to proceed
    proceed = input("\nRun quick start pipeline? (y/n): ").lower().strip()
    if proceed != 'y':
        print("Exiting...")
        return
    
    # Run pipeline
    success = quick_start_pipeline()
    
    if success:
        print("\n🎉 Quick start completed successfully!")
        print("\nFor full pipeline with hyperparameter tuning:")
        print("python main_training_pipeline.py")
    else:
        print("\n❌ Quick start failed. Check logs for details.")

if __name__ == "__main__":
    main()