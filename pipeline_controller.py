#!/usr/bin/env python3
"""
Project Nova - Pipeline Controller
Control individual pipeline stages independently
"""

import os
import sys
import subprocess
from pathlib import Path

def show_menu():
    """Show the pipeline menu"""
    print("="*60)
    print("PROJECT NOVA - PIPELINE CONTROLLER")
    print("="*60)
    print("Choose which stage to run:")
    print()
    print("1. Train Models Only")
    print("2. Evaluate Models (requires trained models)")
    print("3. Analyze Fairness (requires trained models)")
    print("4. Create Visualizations (requires trained models)")
    print("5. Run All Stages (full pipeline)")
    print("6. Check Status")
    print("7. Exit")
    print()

def check_status():
    """Check the status of different pipeline stages"""
    print("\n📊 PIPELINE STATUS CHECK")
    print("-" * 40)
    
    # Check if models are trained
    models_dir = Path("models/saved_models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*"))
        drivers_models = [f for f in model_files if f.name.startswith("drivers_")]
        merchants_models = [f for f in model_files if f.name.startswith("merchants_")]
        
        print(f"✅ Models Directory: {len(model_files)} total files")
        print(f"   - Drivers models: {len(drivers_models)}")
        print(f"   - Merchants models: {len(merchants_models)}")
        
        if drivers_models:
            print("   Drivers models:")
            for model in drivers_models[:5]:  # Show first 5
                print(f"     - {model.name}")
            if len(drivers_models) > 5:
                print(f"     ... and {len(drivers_models) - 5} more")
        
        if merchants_models:
            print("   Merchants models:")
            for model in merchants_models[:5]:  # Show first 5
                print(f"     - {model.name}")
            if len(merchants_models) > 5:
                print(f"     ... and {len(merchants_models) - 5} more")
    else:
        print("❌ Models Directory: Not found")
        print("   Run option 1 to train models first")
    
    # Check evaluation results
    eval_files = [
        "results/metrics/drivers_models_evaluation.csv",
        "results/metrics/merchants_models_evaluation.csv"
    ]
    
    eval_exists = [os.path.exists(f) for f in eval_files]
    if any(eval_exists):
        print(f"✅ Evaluation Results: {sum(eval_exists)}/2 files found")
        for i, exists in enumerate(eval_exists):
            status = "✅" if exists else "❌"
            print(f"   {status} {eval_files[i]}")
    else:
        print("❌ Evaluation Results: Not found")
        print("   Run option 2 to evaluate models")
    
    # Check fairness analysis
    fairness_dir = Path("results/fairness_analysis")
    if fairness_dir.exists():
        fairness_files = list(fairness_dir.glob("*"))
        print(f"✅ Fairness Analysis: {len(fairness_files)} files found")
    else:
        print("❌ Fairness Analysis: Not found")
        print("   Run option 3 to analyze fairness")
    
    # Check visualizations
    plots_dir = Path("results/plots")
    viz_dir = Path("results/visualizations")
    
    plot_count = len(list(plots_dir.glob("*"))) if plots_dir.exists() else 0
    viz_count = len(list(viz_dir.glob("*"))) if viz_dir.exists() else 0
    
    if plot_count > 0 or viz_count > 0:
        print(f"✅ Visualizations: {plot_count} plots, {viz_count} interactive")
    else:
        print("❌ Visualizations: Not found")
        print("   Run option 4 to create visualizations")
    
    print("-" * 40)

def run_stage(stage_number):
    """Run a specific pipeline stage"""
    
    scripts = {
        1: ("train_models_only.py", "Training Models"),
        2: ("evaluate_models.py", "Evaluating Models"),
        3: ("analyze_fairness.py", "Analyzing Fairness"),
        4: ("create_visualizations.py", "Creating Visualizations"),
        5: ("main_training_pipeline.py", "Running Full Pipeline")
    }
    
    if stage_number not in scripts:
        print("❌ Invalid stage number")
        return False
    
    script_name, description = scripts[stage_number]
    
    print(f"\n🚀 {description}...")
    print(f"Running: python3 {script_name}")
    print("-" * 40)
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully!")
            return True
        else:
            print(f"\n❌ {description} failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Failed to run {script_name}: {str(e)}")
        return False

def main():
    """Main controller function"""
    
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == "7":
                print("\n👋 Goodbye!")
                break
            elif choice == "6":
                check_status()
                input("\nPress Enter to continue...")
            elif choice in ["1", "2", "3", "4", "5"]:
                stage_num = int(choice)
                
                # Check prerequisites
                if stage_num in [2, 3, 4]:  # Stages that require trained models
                    models_dir = Path("models/saved_models")
                    if not models_dir.exists() or not list(models_dir.glob("*")):
                        print("\n⚠️  WARNING: No trained models found!")
                        print("You need to run option 1 (Train Models) first.")
                        
                        train_first = input("Would you like to train models first? (y/n): ").lower().strip()
                        if train_first == 'y':
                            print("\n🚀 Training models first...")
                            if run_stage(1):
                                print(f"\n🚀 Now running stage {stage_num}...")
                                run_stage(stage_num)
                        continue
                
                # Run the selected stage
                run_stage(stage_num)
                input("\nPress Enter to continue...")
                
            else:
                print("\n❌ Invalid choice. Please enter 1-7.")
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()