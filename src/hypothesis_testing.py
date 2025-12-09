# src/hypothesis_testing.py

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest

# --- Configuration ---
# FIX: Corrected path for running from the project root directory
DATA_PATH = 'data/MachineLearningRating_v3.txt' 
# Define the separator character (pipe)
DATA_SEP = '|' 
SIGNIFICANCE_LEVEL = 0.05

def load_data(file_path, sep=DATA_SEP):
    """Loads processed data and creates necessary metrics."""
    try:
        # FIX: Pass the separator (sep) and suppress DtypeWarning
        df = pd.read_csv(file_path, sep=sep, low_memory=False) 
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}. Please run data preparation first.")
        return None
    except pd.errors.ParserError as e:
        print(f"Parser Error: Could not read file with separator '{sep}'. Check data file structure. Error: {e}")
        return None
    
    # Check for critical columns needed for the tests
    required_cols = ['PolicyID', 'TotalClaims', 'TotalPremium', 'Gender', 'Province', 'PostalCode']
    if not all(col in df.columns for col in required_cols):
         print(f"Error: Loaded data is missing required columns for analysis: {list(set(required_cols) - set(df.columns))}")
         return None

    # Feature Engineering for Metrics
    df['Claim_Indicator'] = np.where(df['TotalClaims'] > 0, 1, 0)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

def run_hypothesis_tests(df):
    """Executes all required A/B hypothesis tests."""
    if df is None:
        return

    print("--- TASK 3: A/B Hypothesis Testing Results ---")
    
    # Subset data for Claim Severity (only policies that had a claim)
    df_claims = df[df['Claim_Indicator'] == 1].copy()

    # --- H₀: No significant risk difference between Women and Men ---
    
    # Define the two groups for comparison
    groups_to_compare = ['Female', 'Male'] 
    df_gender_filtered = df[df['Gender'].isin(groups_to_compare)].copy()

    # 1. Claim Frequency (Risk): Two-Sample Z-Test for Proportions
    gender_risk = df_gender_filtered.groupby('Gender').agg(
        total_policies=('PolicyID', 'count'),
        total_claims=('Claim_Indicator', 'sum')
    )

    if len(gender_risk) == 2:
        # FIX: Explicitly pass counts and nobs for the two groups
        try:
            count = gender_risk.loc[groups_to_compare, 'total_claims'].tolist()
            nobs = gender_risk.loc[groups_to_compare, 'total_policies'].tolist()
            
            stat, p_value_freq = proportions_ztest(count, nobs)
            
            print(f"\n[Gender] Claim Frequency (Z-Test comparing {groups_to_compare[0]} vs {groups_to_compare[1]}): P-value = {p_value_freq:.4f}")
            print(f"Decision: {'Reject H₀ (Significant)' if p_value_freq < SIGNIFICANCE_LEVEL else 'Fail to Reject H₀ (Not Significant)'}")
        except KeyError:
             print("\n[Gender] Claim Frequency: Error locating 'Female' or 'Male' groups, skipping test.")
    else:
        print("\n[Gender] Claim Frequency: Could not isolate exactly two groups ('Female', 'Male') for Z-Test.")

    # 2. Claim Severity (Risk): Two-Sample T-Test for Means
    claims_women = df_claims[df_claims['Gender'] == 'Female']['TotalClaims'].dropna()
    claims_men = df_claims[df_claims['Gender'] == 'Male']['TotalClaims'].dropna()
    
    if len(claims_women) > 30 and len(claims_men) > 30: # Need sufficient sample size
        stat, p_value_sev = ttest_ind(claims_women, claims_men, equal_var=False)
        
        print(f"[Gender] Claim Severity (T-Test): P-value = {p_value_sev:.4f}")
        print(f"Decision: {'Reject H₀ (Significant)' if p_value_sev < SIGNIFICANCE_LEVEL else 'Fail to Reject H₀ (Not Significant)'}")
    else:
        print("[Gender] Claim Severity: Insufficient data for T-Test.")
    
    # --- H₀: No risk/margin differences across provinces ---
    
    # Filter for provinces with a sufficient sample size (e.g., top 1000 policies)
    province_counts = df['Province'].value_counts()
    top_provinces = province_counts[province_counts >= 1000].index.tolist()
    
    if len(top_provinces) >= 2:
        # 3. Margin Difference (Profit): ANOVA for Means
        province_margins = [df[df['Province'] == p]['Margin'].dropna() for p in top_provinces]
        f_stat, p_value_margin_prov = f_oneway(*province_margins)
        
        print(f"\n[Province] Margin Difference (ANOVA on Top {len(top_provinces)}): P-value = {p_value_margin_prov:.4f}")
        print(f"Decision: {'Reject H₀ (Significant)' if p_value_margin_prov < SIGNIFICANCE_LEVEL else 'Fail to Reject H₀ (Not Significant)'}")
    
        # 4. Risk Differences (Claim Frequency): Chi-Squared Test for Independence
        contingency_table_filtered = pd.crosstab(df['Province'], df['Claim_Indicator']).loc[top_provinces]
        chi2, p_value_chi2_prov, dof, expected = chi2_contingency(contingency_table_filtered)
        
        print(f"[Province] Claim Frequency (Chi-Squared on Top {len(top_provinces)}): P-value = {p_value_chi2_prov:.4f}")
        print(f"Decision: {'Reject H₀ (Significant)' if p_value_chi2_prov < SIGNIFICANCE_LEVEL else 'Fail to Reject H₀ (Not Significant)'}")
    else:
        print("\n[Province] Tests: Insufficient number of major provinces for testing.")

    # --- H₀: No risk/margin differences between zip codes ---
    
    zip_counts = df['PostalCode'].value_counts()
    top_zips = zip_counts[zip_counts >= 500].index.tolist()
    
    if len(top_zips) >= 2:
        # 5. Margin Difference (Profit): ANOVA for Means
        zip_margins = [df[df['PostalCode'] == z]['Margin'].dropna() for z in top_zips]
        f_stat, p_value_margin_zip = f_oneway(*zip_margins)
        
        print(f"\n[ZipCode] Margin Difference (ANOVA on Top {len(top_zips)}): P-value = {p_value_margin_zip:.4f}")
        print(f"Decision: {'Reject H₀ (Significant)' if p_value_margin_zip < SIGNIFICANCE_LEVEL else 'Fail to Reject H₀ (Not Significant)'}")
    else:
         print("\n[ZipCode] Tests: Insufficient number of major zip codes for testing.")
        
    print("\n--- ACTION: Use the rejected hypotheses to justify premium adjustments. ---")


if __name__ == '__main__':
    data = load_data(DATA_PATH)
    if data is not None:
        run_hypothesis_tests(data)