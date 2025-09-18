import pandas as pd
import json
import sys

def convert_libsvm_to_csv(libsvm_file, original_csv, output_csv, feature_info_file):
    """
    Convert adversarial examples from LIBSVM format back to CSV format
    """
    # Load feature information
    with open(feature_info_file, 'r') as f:
        feature_info = json.load(f)
    
    features = feature_info['features']
    
    # Load original CSV to get column structure
    original_df = pd.read_csv(original_csv)
    
    # Read LIBSVM file
    adversarial_data = []
    
    with open(libsvm_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                label = int(parts[0])
                
                # Initialize with zeros
                feature_values = [0.0] * len(features)
                
                # Parse feature:value pairs
                for feature_pair in parts[1:]:
                    if ':' in feature_pair:
                        feature_idx, value = feature_pair.split(':')
                        feature_idx = int(feature_idx) - 1  # Convert to 0-based indexing
                        if 0 <= feature_idx < len(features):
                            feature_values[feature_idx] = float(value)
                
                # Create row with fraud_bool and features
                row_data = [label] + feature_values
                adversarial_data.append(row_data)
    
    # Create DataFrame
    columns = ['fraud_bool'] + features
    adv_df = pd.DataFrame(adversarial_data, columns=columns)
    
    # Decode categorical variables if needed
    if 'label_encoders' in feature_info:
        for col, encoding_dict in feature_info['label_encoders'].items():
            if col in adv_df.columns:
                # Create reverse mapping
                reverse_mapping = {v: k for k, v in encoding_dict.items()}
                adv_df[col] = adv_df[col].map(reverse_mapping).fillna(adv_df[col])
    
    # Save to CSV
    adv_df.to_csv(output_csv, index=False)
    print(f"Adversarial examples saved to {output_csv}")
    print(f"Shape: {adv_df.shape}")
    print(f"Fraud labels: {adv_df['fraud_bool'].value_counts()}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python output_adversarial_csv.py <libsvm_file> <original_csv> <output_csv> <feature_info_json>")
        sys.exit(1)
    
    libsvm_file = sys.argv[1]
    original_csv = sys.argv[2] 
    output_csv = sys.argv[3]
    feature_info_file = sys.argv[4]
    
    convert_libsvm_to_csv(libsvm_file, original_csv, output_csv, feature_info_file)