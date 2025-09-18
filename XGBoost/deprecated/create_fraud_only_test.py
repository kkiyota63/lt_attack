import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

def preprocess_data(df):
    le_dict = {}
    
    categorical_columns = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
    
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = {i: label for i, label in enumerate(le.classes_)}
    
    return df, le_dict

def main():
    df = pd.read_csv('Base.csv')
    
    df_processed, le_dict = preprocess_data(df.copy())
    
    X = df_processed.drop('fraud_bool', axis=1)
    y = df_processed['fraud_bool']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    fraud_indices = y_test[y_test == 1].index
    fraud_X = X_test.loc[fraud_indices]
    fraud_y = y_test.loc[fraud_indices]
    
    print(f"Total fraud cases in test set: {len(fraud_X)}")
    
    fraud_test_data = fraud_X.head(500)
    fraud_test_labels = fraud_y.head(500)
    
    print(f"Selected fraud cases for attack: {len(fraud_test_data)}")
    
    with open('base_fraud_only_test.libsvm', 'w') as f:
        for i, (idx, row) in enumerate(fraud_test_data.iterrows()):
            libsvm_line = f"{int(fraud_test_labels.iloc[i])}"
            for j, val in enumerate(row):
                if float(val) != 0:
                    libsvm_line += f" {j+1}:{val:.6f}"
            f.write(libsvm_line + "\n")
    
    print("Fraud-only test data saved as 'base_fraud_only_test.libsvm'")
    print("Each line represents a fraud case (label=1) that we want to attack to make it appear benign (label=0)")

if __name__ == "__main__":
    main()