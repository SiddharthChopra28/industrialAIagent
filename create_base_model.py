import numpy as np
import pandas as pd
import xgboost as xgb

def create_base_model():
    print("Generating simulated base dataset...")
    np.random.seed(42)
    N = 1000
    
    # Generate features in realistic ranges
    cold_water_flow = np.random.uniform(5.0, 20.0, N)
    inlet_temp = np.random.uniform(60.0, 120.0, N)
    fouling_factor = np.random.uniform(0.0, 30.0, N)
    ambient_temp = np.random.uniform(10.0, 35.0, N)
    
    # Generate target using the same relationship used in test_socket.py
    noise = np.random.normal(0, 1.5, N)
    outlet_temp = (
        0.8 * inlet_temp
        - 0.3 * cold_water_flow
        - 0.2 * fouling_factor
        + 0.5 * ambient_temp
        + noise
    )
    
    df = pd.DataFrame({
        'cold_water_flow_avg': cold_water_flow,
        'inlet_temp_avg': inlet_temp,
        'fouling_factor_avg': fouling_factor,
        'ambient_temp_avg': ambient_temp,
        'outlet_temp_avg': outlet_temp
    })
    
    print("Training base XGBoost model...")
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3, learning_rate=0.1)
    
    X_train = df[['cold_water_flow_avg', 'inlet_temp_avg', 'fouling_factor_avg', 'ambient_temp_avg']]
    y_train = df['outlet_temp_avg']
    
    model.fit(X_train, y_train)
    
    model_filename = "xgboost_model.json"
    model.save_model(model_filename)
    print(f"Base model successfully created and saved to {model_filename}")

if __name__ == "__main__":
    create_base_model()