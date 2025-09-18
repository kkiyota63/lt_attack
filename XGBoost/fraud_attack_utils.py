"""
統合された詐欺検知攻撃ユーティリティモジュール

すべての共通機能を統合し、コードの重複を排除
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import json
import numpy as np
import os
from typing import Tuple, Dict, Any

class FraudAttackSystem:
    """詐欺検知攻撃システムのメインクラス"""
    
    def __init__(self, data_path: str = 'Base.csv', categorical_columns: list = None, feature_constraints: dict = None):
        self.data_path = data_path
        # デフォルトのカテゴリ変数（実際に存在するもののみ使用）
        default_categorical = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
        self.categorical_columns = categorical_columns if categorical_columns is not None else default_categorical
        
        # デフォルトの特徴量制約
        default_constraints = {
            'customer_age': {'min': 18, 'max': 100},
            'income': {'min': 0, 'max': None},
            'credit_risk_score': {'min': 0, 'max': 850},
            'proposed_credit_limit': {'min': 0, 'max': None},
            'session_length_in_minutes': {'min': 0, 'max': None},
            'prev_address_months_count': {'min': 0, 'max': None},
            'current_address_months_count': {'min': 0, 'max': None},
            'bank_months_count': {'min': 0, 'max': None}
        }
        self.feature_constraints = feature_constraints if feature_constraints is not None else default_constraints
        
        self.feature_info = None
        self.model = None
        self.label_encoders = {}
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """データの前処理（カテゴリ変数のエンコーディング）"""
        df_processed = df.copy()
        le_dict = {}
        
        # 実際に存在するカテゴリ変数のみ処理
        existing_categorical = [col for col in self.categorical_columns if col in df_processed.columns]
        
        print(f"Processing categorical columns: {existing_categorical}")
        if len(existing_categorical) < len(self.categorical_columns):
            missing_cols = [col for col in self.categorical_columns if col not in df_processed.columns]
            print(f"Warning: Categorical columns not found in data: {missing_cols}")
        
        for col in existing_categorical:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = {i: label for i, label in enumerate(le.classes_)}
            self.label_encoders[col] = le
        
        return df_processed, le_dict
    
    def train_model(self, save_files: bool = True) -> Dict[str, Any]:
        """XGBoostモデルの訓練"""
        print("Loading and preprocessing data...")
        df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Fraud cases: {df['fraud_bool'].sum()}")
        print(f"Non-fraud cases: {len(df) - df['fraud_bool'].sum()}")
        
        df_processed, le_dict = self.preprocess_data(df)
        
        X = df_processed.drop('fraud_bool', axis=1)
        y = df_processed['fraud_bool']
        
        print(f"Features: {list(X.columns)}")
        print(f"Number of features: {len(X.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Model evaluation
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature info
        self.feature_info = {
            'features': list(X.columns),
            'num_features': len(X.columns),
            'num_classes': 2,
            'feature_start': 1,
            'label_encoders': le_dict
        }
        
        if save_files:
            self._save_model_files(X_test, y_test)
        
        return {
            'accuracy': accuracy,
            'feature_info': self.feature_info,
            'test_data': (X_test, y_test)
        }
    
    def _save_model_files(self, X_test: pd.DataFrame, y_test: pd.Series):
        """モデルと関連ファイルの保存"""
        print("Saving model files...")
        
        # Save model in JSON format for attack
        self.model.get_booster().dump_model('base_fraud_model.json', dump_format='json')
        self._fix_json_format()
        
        # Save feature info
        with open('feature_info.json', 'w') as f:
            json.dump(self.feature_info, f, indent=2)
        
        # Create fraud-only test data
        self._create_fraud_test_data(X_test, y_test)
        
        print("Model files saved successfully")
    
    def _fix_json_format(self):
        """JSONフォーマットの修正（特徴量名→インデックス）"""
        if not self.feature_info:
            return
            
        feature_to_index = {name: idx for idx, name in enumerate(self.feature_info['features'])}
        
        with open('base_fraud_model.json', 'r') as f:
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
        
        with open('base_fraud_model.json', 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def _create_fraud_test_data(self, X_test: pd.DataFrame, y_test: pd.Series):
        """詐欺ケースのみのテストデータ作成"""
        fraud_indices = y_test[y_test == 1].index
        fraud_X = X_test.loc[fraud_indices]
        fraud_y = y_test.loc[fraud_indices]
        
        print(f"Total fraud cases in test set: {len(fraud_X)}")
        
        fraud_test_data = fraud_X.head(500)
        fraud_test_labels = fraud_y.head(500)
        
        # Save in LIBSVM format
        with open('base_fraud_only_test.libsvm', 'w') as f:
            for i, (idx, row) in enumerate(fraud_test_data.iterrows()):
                libsvm_line = f"{int(fraud_test_labels.iloc[i])}"
                for j, val in enumerate(row):
                    if float(val) != 0:
                        libsvm_line += f" {j+1}:{val:.6f}"
                f.write(libsvm_line + "\n")
        
        print(f"Created fraud-only test data: {len(fraud_test_data)} samples")
    
    def convert_adversarial_to_csv(self, libsvm_file: str, output_csv: str, fix_categorical: bool = True) -> pd.DataFrame:
        """敵対的サンプルをLIBSVMからCSVに変換"""
        print(f"Converting {libsvm_file} to CSV format...")
        
        if not self.feature_info:
            with open('feature_info.json', 'r') as f:
                self.feature_info = json.load(f)
        
        features = self.feature_info['features']
        adversarial_data = []
        
        with open(libsvm_file, 'r') as f:
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
                    
                    row_data = [label] + feature_values
                    adversarial_data.append(row_data)
        
        columns = ['fraud_bool'] + features
        adv_df = pd.DataFrame(adversarial_data, columns=columns)
        
        if fix_categorical:
            adv_df = self._fix_categorical_variables(adv_df)
        
        # Apply feature constraints
        adv_df = self._apply_feature_constraints(adv_df)
        
        adv_df.to_csv(output_csv, index=False)
        print(f"Adversarial examples saved to {output_csv}")
        print(f"Shape: {adv_df.shape}")
        
        return adv_df
    
    def _fix_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """カテゴリ変数の修正"""
        df_fixed = df.copy()
        
        if not self.feature_info:
            return df_fixed
        
        print("Fixing categorical variables...")
        
        # 実際に存在するカテゴリ変数のみ処理
        existing_categorical = [col for col in self.categorical_columns if col in df_fixed.columns]
        
        for col in existing_categorical:
            if col in self.feature_info.get('label_encoders', {}):
                valid_values = list(range(len(self.feature_info['label_encoders'][col])))
                
                print(f"  Fixing {col}: valid values {valid_values}")
                
                # Round to nearest valid integer
                df_fixed[col] = np.round(df_fixed[col]).astype(int)
                
                # Clip to valid range
                df_fixed[col] = np.clip(df_fixed[col], min(valid_values), max(valid_values))
        
        return df_fixed
    
    def _apply_feature_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量制約の適用"""
        df_constrained = df.copy()
        
        if not self.feature_constraints:
            return df_constrained
        
        print("Applying feature constraints...")
        
        for feature, constraints in self.feature_constraints.items():
            if feature in df_constrained.columns:
                original_values = df_constrained[feature].copy()
                
                # Apply min constraint
                if constraints.get('min') is not None:
                    min_val = constraints['min']
                    violated_min = df_constrained[feature] < min_val
                    if violated_min.any():
                        df_constrained.loc[violated_min, feature] = min_val
                        print(f"  {feature}: {violated_min.sum()} values clipped to min={min_val}")
                
                # Apply max constraint
                if constraints.get('max') is not None:
                    max_val = constraints['max']
                    violated_max = df_constrained[feature] > max_val
                    if violated_max.any():
                        df_constrained.loc[violated_max, feature] = max_val
                        print(f"  {feature}: {violated_max.sum()} values clipped to max={max_val}")
                
                # Show statistics if changes were made
                if not original_values.equals(df_constrained[feature]):
                    print(f"  {feature}: range {original_values.min():.2f}-{original_values.max():.2f} → {df_constrained[feature].min():.2f}-{df_constrained[feature].max():.2f}")
        
        return df_constrained
    
    def verify_attack_success(self, adversarial_csv: str) -> Dict[str, Any]:
        """攻撃成功率の検証"""
        print("=== 攻撃成功率の検証 ===\n")
        
        adv_df = pd.read_csv(adversarial_csv)
        print(f"敵対的サンプル数: {len(adv_df)}")
        print(f"元ラベル（すべて詐欺=1）: {adv_df['fraud_bool'].unique()}")
        
        if not self.model:
            print("モデルを再作成して検証します...")
            self.train_model(save_files=False)
        
        if not self.feature_info:
            with open('feature_info.json', 'r') as f:
                self.feature_info = json.load(f)
        
        features = self.feature_info['features']
        X_adv = adv_df[features]
        y_true = adv_df['fraud_bool']
        
        # Make predictions
        y_pred = self.model.predict(X_adv)
        y_pred_proba = self.model.predict_proba(X_adv)
        
        # Calculate results
        successful_attacks = np.sum(y_pred == 0)
        attack_success_rate = (successful_attacks / len(y_pred)) * 100
        
        prob_fraud = y_pred_proba[:, 1]
        prob_non_fraud = y_pred_proba[:, 0]
        
        results = {
            'total_samples': len(y_pred),
            'successful_attacks': successful_attacks,
            'success_rate': attack_success_rate,
            'fraud_prob_stats': {
                'mean': prob_fraud.mean(),
                'median': np.median(prob_fraud),
                'min': prob_fraud.min(),
                'max': prob_fraud.max()
            },
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Print results
        print(f"攻撃成功: {successful_attacks:,} / {len(y_pred):,}")
        print(f"成功率: {attack_success_rate:.2f}%")
        print(f"平均詐欺確率: {prob_fraud.mean():.4f}")
        print(f"詐欺確率 < 0.1 の攻撃: {np.sum(prob_fraud < 0.1)} ({(np.sum(prob_fraud < 0.1)/len(prob_fraud))*100:.1f}%)")
        
        # Save detailed results
        results_df = adv_df.copy()
        results_df['predicted_label'] = y_pred
        results_df['fraud_probability'] = prob_fraud
        results_df['non_fraud_probability'] = prob_non_fraud
        results_df['attack_success'] = (y_pred == 0)
        
        results_df.to_csv('attack_results.csv', index=False)
        print(f"\n結果を 'attack_results.csv' に保存しました。")
        
        return results
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """完全なパイプラインの実行"""
        print("=== 詐欺検知攻撃システム - 完全パイプライン ===\n")
        
        # Step 1: Train model
        print("Step 1: Training XGBoost model...")
        train_results = self.train_model()
        
        print(f"\nStep 2: 攻撃実行")
        print("実行コマンド:")
        print("cd .. && ./lt_attack configs/base_fraud_to_benign_attack.json")
        print("\nStep 3: 結果変換 (攻撃完了後に実行):")
        print("system.convert_and_verify_results()")
        
        return train_results
    
    def convert_and_verify_results(self, libsvm_file: str = 'adversarial_examples.libsvm') -> Dict[str, Any]:
        """攻撃結果の変換と検証"""
        print("=== 攻撃結果の変換と検証 ===\n")
        
        # Convert to CSV
        adv_df = self.convert_adversarial_to_csv(libsvm_file, 'adversarial_sample.csv')
        
        # Verify attack success
        results = self.verify_attack_success('adversarial_sample.csv')
        
        return results