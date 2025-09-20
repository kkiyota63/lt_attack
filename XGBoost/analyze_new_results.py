#!/usr/bin/env python3
"""
新しいデータセットの攻撃結果分析
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

def analyze_new_attack_results():
    """新しい攻撃結果の分析"""
    print("=== 新しいデータセット攻撃結果分析 ===\n")
    
    # 必要ファイルの確認
    required_files = [
        'new_adversarial_examples.libsvm',
        'new_feature_info.json',
        'fraud_data_cleaned.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"必要ファイルが見つかりません: {missing_files}")
        return
    
    # 特徴量情報読み込み
    with open('new_feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    features = feature_info['features']
    scaler_mean = np.array(feature_info['scaler_mean'])
    scaler_scale = np.array(feature_info['scaler_scale'])
    
    # 元データ読み込み
    df = pd.read_csv('fraud_data_cleaned.csv')
    fraud_case = df[df['fraud_bool'] == 1].iloc[0]  # 唯一の詐欺ケース
    
    print("元の詐欺ケース:")
    for feature in features:
        print(f"  {feature}: {fraud_case[feature]:.6f}")
    
    # 元データをスケーリング
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    
    original_features = fraud_case[features].values
    original_scaled = scaler.transform(original_features.reshape(1, -1))[0]
    
    # 敵対的サンプル読み込み
    adversarial_data = []
    with open('new_adversarial_examples.libsvm', 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                label = int(parts[0])
                
                feature_values = [0.0] * len(features)
                for feature_pair in parts[1:]:
                    if ':' in feature_pair:
                        feature_idx, value = feature_pair.split(':')
                        feature_idx = int(feature_idx) - 1
                        if 0 <= feature_idx < len(features):
                            feature_values[feature_idx] = float(value)
                
                adversarial_data.append(feature_values)
    
    if not adversarial_data:
        print("敵対的サンプルが見つかりません")
        return
    
    adversarial_scaled = np.array(adversarial_data[0])  # 最初のサンプル
    
    # スケーリングされた空間での摂動計算
    scaled_perturbation = adversarial_scaled - original_scaled
    
    # 元のスケールに戻す
    adversarial_original_scale = scaler.inverse_transform(adversarial_scaled.reshape(1, -1))[0]
    original_perturbation = adversarial_original_scale - original_features
    
    # 摂動分析
    print(f"\n=== 摂動分析 ===")
    print(f"スケーリング済み空間での摂動:")
    print(f"  L1ノルム: {np.sum(np.abs(scaled_perturbation)):.6f}")
    print(f"  L2ノルム: {np.sqrt(np.sum(scaled_perturbation**2)):.6f}")
    print(f"  L∞ノルム: {np.max(np.abs(scaled_perturbation)):.6f}")
    
    print(f"\n元スケールでの摂動:")
    print(f"  L1ノルム: {np.sum(np.abs(original_perturbation)):.6f}")
    print(f"  L2ノルム: {np.sqrt(np.sum(original_perturbation**2)):.6f}")
    print(f"  L∞ノルム: {np.max(np.abs(original_perturbation)):.6f}")
    
    print(f"\n=== 特徴量別変更 ===")
    print("元の値 → 敵対的値 (摂動)")
    
    # 変更された特徴量を摂動の絶対値順でソート
    feature_changes = []
    for i, feature in enumerate(features):
        orig_val = original_features[i]
        adv_val = adversarial_original_scale[i]
        pert = original_perturbation[i]
        if abs(pert) > 1e-6:  # 変更があった場合
            feature_changes.append((feature, orig_val, adv_val, pert, abs(pert)))
    
    # 摂動の絶対値でソート
    feature_changes.sort(key=lambda x: x[4], reverse=True)
    
    print(f"変更された特徴量数: {len(feature_changes)} / {len(features)}")
    
    for feature, orig_val, adv_val, pert, abs_pert in feature_changes[:10]:  # 上位10個
        print(f"  {feature:30s}: {orig_val:10.3f} → {adv_val:10.3f} ({pert:+10.3f})")
    
    # 結果保存
    result_df = pd.DataFrame({
        'feature': features,
        'original': original_features,
        'adversarial': adversarial_original_scale,
        'perturbation': original_perturbation,
        'abs_perturbation': np.abs(original_perturbation)
    })
    
    result_df.to_csv('new_attack_analysis.csv', index=False)
    print(f"\n詳細結果を 'new_attack_analysis.csv' に保存しました")
    
    # 相対的変化率の計算
    print(f"\n=== 相対変化率（上位5特徴量） ===")
    relative_changes = []
    for i, feature in enumerate(features):
        orig_val = abs(original_features[i])
        pert = abs(original_perturbation[i])
        if orig_val > 1e-6:  # ゼロ除算を避ける
            rel_change = (pert / orig_val) * 100
            relative_changes.append((feature, rel_change, pert))
    
    relative_changes.sort(key=lambda x: x[1], reverse=True)
    
    for feature, rel_change, abs_pert in relative_changes[:5]:
        print(f"  {feature:30s}: {rel_change:6.1f}% (摂動: {abs_pert:.3f})")

if __name__ == "__main__":
    import os
    analyze_new_attack_results()