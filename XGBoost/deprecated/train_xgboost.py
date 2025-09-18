import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import json
import os

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
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['fraud_bool'].sum()}")
    print(f"Non-fraud cases: {len(df) - df['fraud_bool'].sum()}")
    
    df_processed, le_dict = preprocess_data(df.copy())
    
    X = df_processed.drop('fraud_bool', axis=1)
    y = df_processed['fraud_bool']
    
    print(f"Features: {list(X.columns)}")
    print(f"Number of features: {len(X.columns)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    model.get_booster().dump_model('base_fraud_model.json', dump_format='json')
    print("\nModel saved as 'base_fraud_model.json'")
    
    with open('feature_info.json', 'w') as f:
        json.dump({
            'features': list(X.columns),
            'num_features': len(X.columns),
            'num_classes': 2,
            'feature_start': 0,
            'label_encoders': le_dict
        }, f, indent=2)
    
    print("Feature information saved as 'feature_info.json'")
    
    test_data = X_test.head(100)
    test_data.to_csv('base_test.libsvm', sep=' ', header=False, index=False, 
                    float_format='%.6f')
    
    with open('base_test.libsvm', 'r') as f:
        lines = f.readlines()
    
    with open('base_test.libsvm', 'w') as f:
        for i, line in enumerate(lines):
            features = line.strip().split()
            libsvm_line = f"{int(y_test.iloc[i])}"
            for j, val in enumerate(features):
                if float(val) != 0:
                    libsvm_line += f" {j+1}:{val}"
            f.write(libsvm_line + "\n")
    
    print("Test data saved in LIBSVM format as 'base_test.libsvm'")

if __name__ == "__main__":
    main()