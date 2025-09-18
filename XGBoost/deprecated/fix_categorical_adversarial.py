import pandas as pd
import numpy as np
import json
import sys

def fix_categorical_variables(adversarial_csv, feature_info_file, output_csv):
    """
    Fix categorical variables in adversarial examples to be valid discrete values
    """
    # Load feature information
    with open(feature_info_file, 'r') as f:
        feature_info = json.load(f)
    
    # Load adversarial examples
    adv_df = pd.read_csv(adversarial_csv)
    
    print(f"Original adversarial examples shape: {adv_df.shape}")
    
    # Get categorical columns and their valid values
    categorical_columns = list(feature_info.get('label_encoders', {}).keys())
    
    print(f"Categorical columns to fix: {categorical_columns}")
    
    for col in categorical_columns:
        if col in adv_df.columns:
            # Get valid encoded values (integers)
            valid_values = list(range(len(feature_info['label_encoders'][col])))
            
            print(f"\nFixing {col}:")
            print(f"  Valid values: {valid_values}")
            print(f"  Before - Min: {adv_df[col].min():.3f}, Max: {adv_df[col].max():.3f}")
            
            # Round to nearest valid integer
            adv_df[col] = np.round(adv_df[col]).astype(int)
            
            # Clip to valid range
            adv_df[col] = np.clip(adv_df[col], min(valid_values), max(valid_values))
            
            print(f"  After  - Min: {adv_df[col].min()}, Max: {adv_df[col].max()}")
            print(f"  Value counts: {dict(adv_df[col].value_counts().sort_index())}")
    
    # Save fixed adversarial examples
    adv_df.to_csv(output_csv, index=False)
    
    print(f"\nFixed adversarial examples saved to: {output_csv}")
    print(f"Shape: {adv_df.shape}")
    
    # Show sample of fixed data
    print("\nSample of fixed adversarial examples:")
    print(adv_df[['fraud_bool'] + categorical_columns].head())
    
    return adv_df

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python fix_categorical_adversarial.py <adversarial_csv> <feature_info_json> <output_csv>")
        sys.exit(1)
    
    adversarial_csv = sys.argv[1]
    feature_info_file = sys.argv[2]
    output_csv = sys.argv[3]
    
    fix_categorical_variables(adversarial_csv, feature_info_file, output_csv)