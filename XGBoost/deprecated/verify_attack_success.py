import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import json
import numpy as np

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
    print("=== 攻撃成功率の検証 ===\n")
    
    # Load feature info
    with open('feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    # Load adversarial examples
    adv_df = pd.read_csv('adversarial_fixed.csv')
    print(f"敵対的サンプル数: {len(adv_df)}")
    print(f"元ラベル（すべて詐欺=1）: {adv_df['fraud_bool'].unique()}")
    
    # Recreate and retrain the model (since JSON format is incompatible)
    print("モデルを再作成して検証します...")
    
    # Load original training data
    df_original = pd.read_csv('Base.csv')
    df_processed, le_dict = preprocess_data(df_original.copy())
    
    X_original = df_processed.drop('fraud_bool', axis=1)
    y_original = df_processed['fraud_bool']
    
    # Train the same model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_original, y_original)
    print("モデル再訓練完了")
    
    # Extract features (exclude fraud_bool)
    features = feature_info['features']
    X_adv = adv_df[features]
    y_true = adv_df['fraud_bool']  # All should be 1 (fraud)
    
    print(f"\n特徴量数: {len(features)}")
    print(f"特徴量: {features[:5]}... (最初の5個)")
    
    # Make predictions on adversarial examples
    y_pred = model.predict(X_adv)
    y_pred_proba = model.predict_proba(X_adv)
    
    print(f"\n=== 攻撃結果 ===")
    print(f"予測結果の分布:")
    unique, counts = np.unique(y_pred, return_counts=True)
    for label, count in zip(unique, counts):
        label_name = "非詐欺" if label == 0 else "詐欺"
        percentage = (count / len(y_pred)) * 100
        print(f"  {label_name} (label={label}): {count:,} サンプル ({percentage:.1f}%)")
    
    # Calculate attack success rate
    successful_attacks = np.sum(y_pred == 0)  # Originally fraud(1) predicted as non-fraud(0)
    attack_success_rate = (successful_attacks / len(y_pred)) * 100
    
    print(f"\n=== 攻撃成功率 ===")
    print(f"攻撃成功: {successful_attacks:,} / {len(y_pred):,}")
    print(f"成功率: {attack_success_rate:.2f}%")
    
    # Analyze prediction confidence
    prob_fraud = y_pred_proba[:, 1]  # Probability of being fraud
    prob_non_fraud = y_pred_proba[:, 0]  # Probability of being non-fraud
    
    print(f"\n=== 予測確信度 ===")
    print(f"詐欺確率の統計:")
    print(f"  平均: {prob_fraud.mean():.4f}")
    print(f"  中央値: {np.median(prob_fraud):.4f}")
    print(f"  最小値: {prob_fraud.min():.4f}")
    print(f"  最大値: {prob_fraud.max():.4f}")
    
    print(f"\n非詐欺確率の統計:")
    print(f"  平均: {prob_non_fraud.mean():.4f}")
    print(f"  中央値: {np.median(prob_non_fraud):.4f}")
    print(f"  最小値: {prob_non_fraud.min():.4f}")
    print(f"  最大値: {prob_non_fraud.max():.4f}")
    
    # Analyze successful attacks
    successful_mask = y_pred == 0
    if successful_attacks > 0:
        successful_fraud_prob = prob_fraud[successful_mask]
        print(f"\n=== 成功した攻撃の分析 ===")
        print(f"成功攻撃の詐欺確率:")
        print(f"  平均: {successful_fraud_prob.mean():.4f}")
        print(f"  中央値: {np.median(successful_fraud_prob):.4f}")
        print(f"  最大値: {successful_fraud_prob.max():.4f}")
        
        # Count how many have very low fraud probability
        very_confident = np.sum(successful_fraud_prob < 0.1)
        print(f"  詐欺確率 < 0.1 の攻撃: {very_confident} ({(very_confident/successful_attacks)*100:.1f}%)")
    
    # Sample some successful attacks
    if successful_attacks > 0:
        print(f"\n=== 成功攻撃のサンプル ===")
        successful_indices = np.where(successful_mask)[0][:5]
        for i, idx in enumerate(successful_indices):
            print(f"サンプル {i+1} (行{idx}):")
            print(f"  予測: {y_pred[idx]} (非詐欺)")
            print(f"  詐欺確率: {prob_fraud[idx]:.4f}")
            print(f"  非詐欺確率: {prob_non_fraud[idx]:.4f}")
    
    # Save results
    results_df = adv_df.copy()
    results_df['predicted_label'] = y_pred
    results_df['fraud_probability'] = prob_fraud
    results_df['non_fraud_probability'] = prob_non_fraud
    results_df['attack_success'] = (y_pred == 0)
    
    results_df.to_csv('attack_results.csv', index=False)
    print(f"\n結果を 'attack_results.csv' に保存しました。")

if __name__ == "__main__":
    main()