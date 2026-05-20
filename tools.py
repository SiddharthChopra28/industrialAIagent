import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from scipy.stats import pearsonr
from langchain_core.tools import tool

@tool
def extract_key_variables(cold_water_flow: float = None, inlet_temp: float = None, fouling_factor: float = None, ambient_temp: float = None) -> dict:
    """
    Extract key variables using XGBoost and SHAP.
    Hyperparameter tunes/updates the model with new data from data_averages.csv.
    Returns a dictionary with 'feature_correlations' containing the Pearson correlation coefficient of 
    the raw variable readings and the SHAP data for that variable, for each variable Cold Water Flow Rate, Inlet Temperature, Fouling Factor, Ambient Air Temperature.
    Optionally predicts Outlet Temperature if cold_water_flow, inlet_temp, fouling_factor, ambient_temp are provided.
    """
    csv_file = "data_averages.csv"
    model_file = "xgboost_model.json"
    
    if not os.path.exists(model_file):
        return {"error": "Base model not found. Please run create_base_model.py first."}
        
    if not os.path.exists(csv_file):
         return {"error": "CSV file data_averages.csv not found. Background socket needs to collect data before training."}

    df = pd.read_csv(csv_file)
    
    # Ensure Used_For_Training column exists for backward compatibility
    if 'Used_For_Training' not in df.columns:
        df['Used_For_Training'] = False
        
    # Cast to object/string to avoid dtype issues with mixed bools and strings
    df['Used_For_Training'] = df['Used_For_Training'].astype(str)
        
    # Find rows that haven't been used for training yet
    mask = df['Used_For_Training'].str.lower() != 'true'

    # Always include the latest batch for training
    if 'Timestamp_avg' in df.columns and len(df) > 0:
        latest_idx = df['Timestamp_avg'].idxmax()
        mask.loc[latest_idx] = True
    
    # Train if new data is present
    if mask.any():
        new_df = df[mask]
        X_new = new_df[['cold_water_flow_avg', 'inlet_temp_avg', 'fouling_factor_avg', 'ambient_temp_avg']]
        y_new = new_df['outlet_temp_avg']
        
        # Load and update model
        model = xgb.XGBRegressor()
        model.load_model(model_file)
        
        # Fit on new data incrementally by passing xgb_model
        model.fit(X_new, y_new, xgb_model=model_file)
        model.save_model(model_file)
        
        # Mark as used and save to avoid retraining on the same data next time
        df.loc[mask, 'Used_For_Training'] = 'True'
        df.to_csv(csv_file, index=False)
        
    # Reload model for SHAP & prediction
    model = xgb.XGBRegressor()
    model.load_model(model_file)
    
    # Reload CSV and extract entire features to compute SHAP values
    df = pd.read_csv(csv_file)
    X_all = df[['cold_water_flow_avg', 'inlet_temp_avg', 'fouling_factor_avg', 'ambient_temp_avg']]
    
    correlations = [0.0, 0.0, 0.0, 0.0]
    correlation_warning = None
    if len(X_all) > 1:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_all)
        
        features = ['cold_water_flow_avg', 'inlet_temp_avg', 'fouling_factor_avg', 'ambient_temp_avg']
        for i, feature in enumerate(features):
            raw_vals = X_all[feature].values
            shap_vals = shap_values[:, i]
            
            if np.std(raw_vals) > 0 and np.std(shap_vals) > 0:
                corr, _ = pearsonr(raw_vals, shap_vals)
            else:
                corr = 0.0
            correlations[i] = float(corr)
    else:
        correlation_warning = (
            "Not enough batch data to compute Pearson correlations reliably. "
            "Need at least 2 batches with variance."
        )
            
    predicted_x = None
    if all(v is not None for v in (cold_water_flow, inlet_temp, fouling_factor, ambient_temp)):
        input_data = pd.DataFrame({'cold_water_flow_avg': [cold_water_flow], 'inlet_temp_avg': [inlet_temp], 'fouling_factor_avg': [fouling_factor], 'ambient_temp_avg': [ambient_temp]})
        predicted_x = float(model.predict(input_data)[0])
        
    result = {
        "feature_correlations": correlations,
        "predicted_outlet_temp": predicted_x
    }
    if correlation_warning:
        result["warning"] = correlation_warning
    return result

@tool
def fetch_batch_by_timestamp(target_timestamp: float) -> dict:
    """
    Fetches the details of a batch from the CSV file based on its average timestamp.
    Returns the batch data whose 'Timestamp_avg' is closest to the provided target_timestamp.
    """
    csv_file = "data_averages.csv"
    if not os.path.exists(csv_file):
        return {"error": "CSV file data_averages.csv not found."}
        
    df = pd.read_csv(csv_file)
    if len(df) == 0:
        return {"error": "CSV file is empty."}
        
    if 'Timestamp_avg' not in df.columns:
        return {"error": "Timestamp_avg column not found in CSV."}
        
    # Find the row with the minimum absolute difference to the target timestamp
    df['time_diff'] = (df['Timestamp_avg'] - target_timestamp).abs()
    closest_batch = df.loc[df['time_diff'].idxmin()]
    
    # Drop the temporary time_diff column
    result = closest_batch.drop('time_diff').to_dict()
    
    return {"batch_details": result}
