#!/usr/bin/env python3
"""
新しいデータセット（top100_cleaned_data.csv）を使った攻撃システム
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import json
import numpy as np
import os

class NewFraudAttackSystem:
    """新しいデータセット用の攻撃システム"""
    
    def __init__(self, data_path: str = 'fraud_data_cleaned.csv'):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_info = None
        
    def load_and_analyze_data(self):
        """データの読み込みと分析"""
        df = pd.read_csv(self.data_path)
        
        print(f"データセット情報:")
        print(f"- Shape: {df.shape}")
        print(f"- fraud_bool分布: {df['fraud_bool'].value_counts().to_dict()}")
        print(f"- 特徴量数: {df.shape[1] - 1}")
        print(f"- カラム: {list(df.columns)}")
        
        # fraud_bool=1のケースを確認
        fraud_cases = df[df['fraud_bool'] == 1]
        print(f"\n詐欺ケース:")
        print(fraud_cases)
        
        return df
    
    def train_model_with_scaling(self, save_files: bool = True):
        """スケーリング付きでモデルを訓練"""
        df = self.load_and_analyze_data()
        
        # 特徴量とターゲットの分離
        X = df.drop('fraud_bool', axis=1)
        y = df['fraud_bool']
        
        # データをスケーリング
        print("\nデータをスケーリング中...")
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        print("スケーリング前後の統計:")
        print("元データ:")
        print(X.describe().iloc[1:3])  # mean, std
        print("スケーリング後:")
        print(X_scaled_df.describe().iloc[1:3])  # mean, std
        
        # 訓練・テスト分割（stratifyできないので通常分割）
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df, y, test_size=0.2, random_state=42
        )
        
        # XGBoostモデル訓練
        print("\nXGBoostモデルを訓練中...")
        self.model = xgb.XGBClassifier(
            n_estimators=50,  # データが少ないので減らす
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train, y_train)
        
        # モデル評価
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nモデル性能:")
        print(f"Accuracy: {accuracy:.4f}")
        
        # 特徴量情報
        self.feature_info = {
            'features': list(X.columns),
            'num_features': len(X.columns),
            'num_classes': 2,
            'feature_start': 1,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist()
        }
        
        if save_files:
            self._save_model_files(X_scaled_df, y)
        
        return {
            'accuracy': accuracy,
            'feature_info': self.feature_info,
            'scaled_data': X_scaled_df,
            'target': y
        }
    
    def _save_model_files(self, X_scaled, y):
        """モデルと関連ファイルの保存"""
        print("モデルファイルを保存中...")
        
        # スケーリング済みモデルをJSON形式で保存
        self.model.get_booster().dump_model('new_fraud_model.json', dump_format='json')
        self._fix_json_format()
        
        # 特徴量情報保存
        with open('new_feature_info.json', 'w') as f:
            json.dump(self.feature_info, f, indent=2)
        
        # 詐欺ケースのテストデータ作成
        self._create_fraud_test_data(X_scaled, y)
        
        print("ファイル保存完了")
    
    def _fix_json_format(self):
        """JSONフォーマットの修正（特徴量名→インデックス）"""
        feature_to_index = {name: idx for idx, name in enumerate(self.feature_info['features'])}
        
        with open('new_fraud_model.json', 'r') as f:
            model_data = json.load(f)
        
        def fix_tree(node):
            if 'split' in node:
                feature_name = node['split']
                if feature_name in feature_to_index:
                    node['split'] = feature_to_index[feature_name]
            
            if 'children' in node:
                for child in node['children']:
                    fix_tree(child)
        
        for tree in model_data:
            fix_tree(tree)
        
        with open('new_fraud_model.json', 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def _create_fraud_test_data(self, X_scaled, y):
        """詐欺ケースのテストデータ作成"""
        # 詐欺ケースを特定
        fraud_indices = y[y == 1].index
        
        if len(fraud_indices) == 0:
            print("警告: 詐欺ケースが見つかりません")
            return
        
        # 詐欺ケースのデータを取得
        fraud_X = X_scaled.loc[fraud_indices]
        fraud_y = y.loc[fraud_indices]
        
        print(f"詐欺ケース数: {len(fraud_X)}")
        
        # LIBSVM形式で保存
        with open('new_fraud_test.libsvm', 'w') as f:
            for i, (idx, row) in enumerate(fraud_X.iterrows()):
                libsvm_line = f"{int(fraud_y.iloc[i])}"
                for j, val in enumerate(row):
                    if float(val) != 0:
                        libsvm_line += f" {j+1}:{val:.6f}"
                f.write(libsvm_line + "\n")
        
        print(f"LIBSVM形式でテストデータを保存: {len(fraud_X)} サンプル")
    
    def create_attack_config(self):
        """攻撃用設定ファイルの作成"""
        config = {
            "search_mode": "lt-attack",
            "norm_type": 2,
            "num_point": 1,  # 詐欺ケースが1件のため
            "num_threads": 1,
            "num_attack_per_point": 1,
            "enable_early_return": True,
            "model": "XGBoost/new_fraud_model.json",
            "inputs": "XGBoost/new_fraud_test.libsvm",
            "num_classes": 2,
            "num_features": self.feature_info['num_features'],
            "feature_start": 1,
            "adv_training_path": "XGBoost/new_adversarial_examples.libsvm"
        }
        
        config_path = '../configs/new_fraud_attack.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"攻撃設定ファイルを作成: {config_path}")
        return config_path

def main():
    print("=== 新しいデータセットでの攻撃システム ===\n")
    
    # システム初期化
    system = NewFraudAttackSystem()
    
    # モデル訓練
    print("Step 1: スケーリング付きモデル訓練...")
    results = system.train_model_with_scaling()
    
    # 攻撃設定作成
    print("\nStep 2: 攻撃設定作成...")
    config_path = system.create_attack_config()
    
    print(f"\n=== 準備完了 ===")
    print(f"次のコマンドで攻撃を実行:")
    print(f"cd .. && ./lt_attack {config_path}")
    
    print(f"\n攻撃後の分析:")
    print(f"cd XGBoost && python3 analyze_new_results.py")

if __name__ == "__main__":
    main()