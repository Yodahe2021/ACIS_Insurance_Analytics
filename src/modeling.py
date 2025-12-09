# src/modeling.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
import shap 

# --- Configuration ---
DATA_PATH = 'data/MachineLearningRating_v3.txt' 
DATA_SEP = '|' 

# CRITICAL: Using the EXACT lowercase names from the shared data header
numerical_features = ['Cylinders', 'cubiccapacity', 'kilowatts', 'CustomValueEstimate', 'SumInsured'] 
categorical_features = ['Province', 'Gender', 'make', 'VehicleType', 'bodytype', 'AlarmImmobiliser', 'TrackingDevice']

def create_preprocessor(): 
    """Defines the preprocessing steps (Scaling and Encoding)."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            # FIX: Removed 'handle_missing' to avoid TypeError
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) 
        ],
        remainder='drop'
    )

def prep_data_for_modeling(file_path, sep):
    """Loads and splits data for Classification and Regression tasks."""
    try:
        df = pd.read_csv(file_path, sep=sep, low_memory=False)
    except Exception as e:
        print(f"Error loading data for modeling: {e}")
        # Return 8 Nones
        return None, None, None, None, None, None, None, None 

    # --- Feature Engineering ---
    try:
        if 'RegistrationYear' in df.columns:
            df['Car_Age'] = pd.to_datetime('2015-08-01').year - df['RegistrationYear']
            if 'Car_Age' not in numerical_features:
                 numerical_features.append('Car_Age')
        
    except KeyError:
        print("Warning: 'RegistrationYear' not found. Skipping Car_Age feature creation.")

    # Targets and Claim Indicator
    df['Claim_Indicator'] = np.where(df['TotalClaims'] > 0, 1, 0)
    
    X_cols = numerical_features + categorical_features
    
    # Drop rows with NaNs in critical feature columns
    try:
        df_clean = df.dropna(subset=X_cols + ['TotalClaims', 'Claim_Indicator']).copy()
    except KeyError as e:
        print("\nCRITICAL KEY ERROR DURING DATA CLEANUP:")
        print(f"The following features were not found in the data file: {e}")
        print("Please ensure the feature lists in modeling.py match your data header exactly.")
        # Return 8 Nones
        return None, None, None, None, None, None, None, None 

    X = df_clean[X_cols]
    
    # 1. Claim Probability Model Data (All policies)
    y_prob = df_clean['Claim_Indicator']
    X_train_prob, X_test_prob, y_train_prob, y_test_prob = train_test_split(
        X, y_prob, test_size=0.2, random_state=42, stratify=y_prob
    )

    # 2. Claim Severity Model Data (Only policies with claims)
    df_claims = df_clean[df_clean['Claim_Indicator'] == 1].copy()
    X_sev = df_claims[X_cols]
    y_sev = df_claims['TotalClaims']
    X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
        X_sev, y_sev, test_size=0.2, random_state=42
    )
    
    # CRITICAL FIX: Return ALL 8 variables: X_train/test, y_train/test for BOTH models
    return X_train_prob, X_test_prob, y_train_prob, y_test_prob, X_train_sev, X_test_sev, y_train_sev, y_test_sev

# CRITICAL FIX: This function accepts 8 variables
def train_and_evaluate_models(X_train_prob, X_test_prob, y_train_prob, y_test_prob, X_train_sev, X_test_sev, y_train_sev, y_test_sev):
    
    preprocessor = create_preprocessor()

    print("--- TASK 4: Predictive Modeling ---")
    
    # --- 1. Claim Probability Model (XGBoost Classifier) ---
    xgb_prob_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1))
    ])
    # y_train_prob is correctly available
    xgb_prob_model.fit(X_train_prob, y_train_prob) 
    y_pred_proba = xgb_prob_model.predict_proba(X_test_prob)[:, 1]
    auc_roc = roc_auc_score(y_test_prob, y_pred_proba)
    print(f"\n1. Claim Probability Model AUC-ROC: {auc_roc:.4f}")

    # --- 2. Claim Severity Model (XGBoost Regressor) ---
    xgb_sev_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1))
    ])
    # FIX: y_train_sev is now correctly available
    xgb_sev_model.fit(X_train_sev, y_train_sev)
    y_pred_sev = xgb_sev_model.predict(X_test_sev)
    rmse = np.sqrt(mean_squared_error(y_test_sev, y_pred_sev))
    r2 = r2_score(y_test_sev, y_pred_sev)
    print(f"2. Claim Severity Model RMSE: {rmse:.2f} | R-squared: {r2:.4f}")

    return xgb_prob_model, xgb_sev_model, X_test_prob, y_pred_proba

def calculate_risk_based_premium(prob_model, sev_model, X_test, y_pred_proba):
    """Calculates the final Risk-Based Premium based on predicted loss."""
    
    predicted_severity = sev_model.predict(X_test)
    pure_premium = y_pred_proba * predicted_severity
    
    EXPENSE_LOADING = 400.00  
    PROFIT_MARGIN = 0.08      
    
    # Risk-Based Premium Formula 
    risk_based_premium = (pure_premium + EXPENSE_LOADING) / (1 - PROFIT_MARGIN)
    
    print("\n3. Risk-Based Premium Calculation (Sample):")
    results = pd.DataFrame({
        'Predicted_Prob': y_pred_proba,
        'Predicted_Severity': predicted_severity,
        'Pure_Premium': pure_premium,
        'Risk_Based_Premium': risk_based_premium
    })
    print(results[['Pure_Premium', 'Risk_Based_Premium']].head())

def run_shap_analysis(model, X_train):
    """Generates SHAP values for model interpretation."""
    
    preprocessor = model.named_steps['preprocessor']
    X_train_transformed = preprocessor.transform(X_train)
    
    try:
        # Use the updated numerical_features list (including Car_Age if created)
        ohe_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    except AttributeError:
        ohe_features = [f'OHE_cat_{i}' for i in range(X_train_transformed.shape[1] - len(numerical_features))]

    full_feature_names = numerical_features + ohe_features
    
    regressor = model.named_steps['regressor']
    
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_train_transformed)
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=full_feature_names).sort_values(ascending=False)

    print("\n4. Top 10 Most Influential Features (Claim Severity Model - SHAP):") 
    print(feature_importance.head(10))

if __name__ == '__main__':
    
    # Prepare Data for Modeling
    # CRITICAL FIX: Must unpack 8 variables here to match the return from prep_data_for_modeling
    X_train_prob, X_test_prob, y_train_prob, y_test_prob, X_train_sev, X_test_sev, y_train_sev, y_test_sev = prep_data_for_modeling(
        DATA_PATH, DATA_SEP
    )

    # Check if data loading failed (indicated by None in X_train_prob)
    if X_train_prob is not None:
        # Train, Evaluate, and Calculate Premium
        prob_model, sev_model, X_test_full, y_pred_proba = train_and_evaluate_models(
            X_train_prob, X_test_prob, y_train_prob, y_test_prob, X_train_sev, X_test_sev, y_train_sev, y_test_sev
        )
        
        calculate_risk_based_premium(prob_model, sev_model, X_test_full, y_pred_proba)
        
        # Run Interpretation 
        run_shap_analysis(sev_model, X_train_sev)