"""
Setup script to automatically prepare the environment and run the application
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please install manually using: pip install -r requirements.txt")
        return False

def check_csv_file():
    """Check if CSV file exists"""
    csv_file = "multi_city_food_delivery_demand.csv"
    if os.path.exists(csv_file):
        print(f"✅ Found dataset: {csv_file}")
        return True
    else:
        print(f"❌ Dataset not found: {csv_file}")
        print("Please ensure your CSV file is named 'multi_city_food_delivery_demand.csv' and is in this directory.")
        return False

def train_model():
    """Train the machine learning model"""
    print("🤖 Training machine learning model...")
    try:
        result = subprocess.run([sys.executable, "train_model.py"], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            return True
        else:
            print("❌ Model training failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Model training timed out. Please run manually: python train_model.py")
        return False
    except Exception as e:
        print(f"❌ Error during model training: {str(e)}")
        return False

def check_model_file():
    """Check if model file exists"""
    model_file = "hyperlocal_demand_predictor.pkl"
    if os.path.exists(model_file):
        print(f"✅ Model file found: {model_file}")
        return True
    else:
        print(f"❌ Model file not found: {model_file}")
        return False

def run_streamlit():
    """Run the Streamlit application"""
    print("🚀 Starting Streamlit application...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except Exception as e:
        print(f"❌ Error running Streamlit: {str(e)}")
        print("Try running manually: streamlit run streamlit_app.py")

def main():
    """Main setup function"""
    print("🍕 Food Delivery Demand Predictor - Setup Script")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        return
    
    # Step 2: Check for CSV file
    if not check_csv_file():
        print("\n📋 Setup incomplete. Please add your dataset file and run again.")
        return
    
    # Step 3: Train model if needed
    if not check_model_file():
        print("\n🔄 Model not found. Training new model...")
        if not train_model():
            print("\n📋 Setup incomplete. Please check the error messages above.")
            return
    else:
        print("\n✅ Using existing trained model.")
        retrain = input("Would you like to retrain the model with current data? (y/n): ").lower().strip()
        if retrain == 'y':
            if not train_model():
                print("❌ Retraining failed. Using existing model.")
    
    # Step 4: Final check
    if check_model_file():
        print("\n✅ Setup complete! All components ready.")
        print("\n🚀 Launching application...")
        run_streamlit()
    else:
        print("\n❌ Setup failed. Model file not found after training.")

if __name__ == "__main__":
    main()