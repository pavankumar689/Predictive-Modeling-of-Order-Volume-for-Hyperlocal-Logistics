"""
Model Training Script for Food Delivery Demand Prediction
Run this script to train and save the model from your notebook code
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import time

warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    """Load and preprocess the data"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    print(f"Data shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Display unique values in categorical columns to verify
    print("\n=== Dataset Overview ===")
    categorical_cols = ['city', 'zone_id', 'zone_type', 'weather_condition']
    for col in categorical_cols:
        if col in df.columns:
            print(f"{col}: {sorted(df[col].unique())}")
    
    # Create a copy for processing
    df_processed = df.copy()
    
    # One-hot encoding - ensure same columns as in your original analysis
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    print(f"After encoding - columns: {len(df_processed.columns)}")
    print(f"Feature columns: {[col for col in df_processed.columns if col != 'num_orders']}")
    
    return df_processed

def train_models(df_processed):
    """Train and compare multiple models"""
    # Prepare features and target
    y = df_processed['num_orders']
    X = df_processed.drop('num_orders', axis=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        end_time = time.time()
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[name] = {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'training_time': end_time - start_time
        }
        
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"Training Time: {end_time - start_time:.2f}s")
    
    return results, X_train, X_test, y_train, y_test

def tune_best_model(results, X_train, X_test, y_train, y_test):
    """Hyperparameter tuning for the best model"""
    # Find the best model based on R² score
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    print(f"\nBest base model: {best_model_name}")
    
    if best_model_name == "LightGBM":
        print("\nPerforming hyperparameter tuning for LightGBM...")
        
        # Parameter grid
        param_grid = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100],
            'max_depth': [-1, 10, 20],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        lgbm_base = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
        
        random_search = RandomizedSearchCV(
            estimator=lgbm_base,
            param_distributions=param_grid,
            n_iter=20,
            cv=3,
            random_state=42,
            n_jobs=-1,
            scoring='neg_root_mean_squared_error'
        )
        
        start_time = time.time()
        random_search.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"Tuning completed in {end_time - start_time:.2f}s")
        print(f"Best parameters: {random_search.best_params_}")
        
        # Evaluate tuned model
        best_model = random_search.best_estimator_
        y_pred_tuned = best_model.predict(X_test)
        
        r2_tuned = r2_score(y_test, y_pred_tuned)
        rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
        
        print(f"Tuned model R² Score: {r2_tuned:.4f}")
        print(f"Tuned model RMSE: {rmse_tuned:.2f}")
        
        return best_model, X_train.columns.tolist()
    
    else:
        # Return the best base model if not LightGBM
        return results[best_model_name]['model'], X_train.columns.tolist()

def save_model(model, feature_columns, filename="hyperlocal_demand_predictor.pkl"):
    """Save the trained model and feature columns"""
    model_assets = {
        'model': model,
        'columns': feature_columns
    }
    
    joblib.dump(model_assets, filename)
    print(f"\nModel saved as '{filename}'")
    
    # Verification
    loaded_assets = joblib.load(filename)
    print("Model verification successful!")
    print(f"Feature columns count: {len(loaded_assets['columns'])}")

def main():
    """Main training pipeline"""
    print("=== Food Delivery Demand Prediction - Model Training ===\n")
    
    # File path - update this to your CSV file path
    csv_file_path = "multi_city_food_delivery_demand.csv"
    
    try:
        # Load and preprocess data
        df_processed = load_and_preprocess_data(csv_file_path)
        
        # Train models
        results, X_train, X_test, y_train, y_test = train_models(df_processed)
        
        # Tune best model
        best_model, feature_columns = tune_best_model(results, X_train, X_test, y_train, y_test)
        
        # Save the final model
        save_model(best_model, feature_columns)
        
        print("\n=== Training Pipeline Completed Successfully! ===")
        print("You can now run the Streamlit app with: streamlit run streamlit_app.py")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file_path}'")
        print("Please ensure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()