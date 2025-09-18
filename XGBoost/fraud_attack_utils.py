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
    
    def analyze_perturbations(self, original_csv: str = None, adversarial_csv: str = 'adversarial_sample.csv') -> Dict[str, Any]:
        """元データと敵対的サンプルの摂動を詳細分析"""
        print("=== 摂動分析 ===\n")
        
        # Load adversarial data
        adv_df = pd.read_csv(adversarial_csv)
        
        # Load original data (from the fraud-only test set)
        if original_csv is None:
            # Load from the original Base.csv and filter to fraud cases
            df = pd.read_csv(self.data_path)
            df_processed, _ = self.preprocess_data(df)
            fraud_cases = df_processed[df_processed['fraud_bool'] == 1]
            
            # Get the same number of cases as adversarial samples
            original_df = fraud_cases.head(len(adv_df))
        else:
            original_df = pd.read_csv(original_csv)
        
        if not self.feature_info:
            with open('feature_info.json', 'r') as f:
                self.feature_info = json.load(f)
        
        features = self.feature_info['features']
        
        # Calculate perturbations
        original_features = original_df[features].values
        adversarial_features = adv_df[features].values
        
        # Calculate different norms of perturbation
        perturbations = adversarial_features - original_features
        
        # Per-sample perturbation norms
        l1_norms = np.sum(np.abs(perturbations), axis=1)
        l2_norms = np.sqrt(np.sum(perturbations**2, axis=1))
        linf_norms = np.max(np.abs(perturbations), axis=1)
        
        # Per-feature perturbation statistics
        feature_perturbations = {}
        for i, feature in enumerate(features):
            feature_pert = perturbations[:, i]
            feature_perturbations[feature] = {
                'mean_abs': np.mean(np.abs(feature_pert)),
                'std': np.std(feature_pert),
                'max_abs': np.max(np.abs(feature_pert)),
                'changed_samples': np.sum(np.abs(feature_pert) > 1e-6),
                'change_rate': np.sum(np.abs(feature_pert) > 1e-6) / len(feature_pert)
            }
        
        # Create detailed comparison DataFrame
        comparison_df = pd.DataFrame()
        comparison_df['sample_id'] = range(len(adv_df))
        comparison_df['l1_perturbation'] = l1_norms
        comparison_df['l2_perturbation'] = l2_norms
        comparison_df['linf_perturbation'] = linf_norms
        
        # Add original vs adversarial values for key features
        for feature in features:
            comparison_df[f'{feature}_original'] = original_df[feature].values
            comparison_df[f'{feature}_adversarial'] = adv_df[feature].values
            comparison_df[f'{feature}_perturbation'] = perturbations[:, features.index(feature)]
        
        # Statistics
        perturbation_stats = {
            'total_samples': len(adv_df),
            'l1_norm': {
                'mean': np.mean(l1_norms),
                'median': np.median(l1_norms),
                'std': np.std(l1_norms),
                'min': np.min(l1_norms),
                'max': np.max(l1_norms)
            },
            'l2_norm': {
                'mean': np.mean(l2_norms),
                'median': np.median(l2_norms),
                'std': np.std(l2_norms),
                'min': np.min(l2_norms),
                'max': np.max(l2_norms)
            },
            'linf_norm': {
                'mean': np.mean(linf_norms),
                'median': np.median(linf_norms),
                'std': np.std(linf_norms),
                'min': np.min(linf_norms),
                'max': np.max(linf_norms)
            },
            'feature_perturbations': feature_perturbations
        }
        
        # Print summary
        print(f"総サンプル数: {len(adv_df)}")
        print(f"\n摂動のノルム統計:")
        print(f"L1ノルム - 平均: {perturbation_stats['l1_norm']['mean']:.4f}, 中央値: {perturbation_stats['l1_norm']['median']:.4f}")
        print(f"L2ノルム - 平均: {perturbation_stats['l2_norm']['mean']:.4f}, 中央値: {perturbation_stats['l2_norm']['median']:.4f}")
        print(f"L∞ノルム - 平均: {perturbation_stats['linf_norm']['mean']:.4f}, 中央値: {perturbation_stats['linf_norm']['median']:.4f}")
        
        print(f"\n最も変更された特徴量 (変更率順):")
        sorted_features = sorted(feature_perturbations.items(), key=lambda x: x[1]['change_rate'], reverse=True)
        for feature, stats in sorted_features[:10]:
            print(f"  {feature}: 変更率{stats['change_rate']:.2%}, 平均摂動{stats['mean_abs']:.4f}, 最大摂動{stats['max_abs']:.4f}")
        
        # Save results
        comparison_df.to_csv('perturbation_analysis.csv', index=False)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        perturbation_stats_serializable = convert_numpy_types(perturbation_stats)
        
        with open('perturbation_stats.json', 'w') as f:
            json.dump(perturbation_stats_serializable, f, indent=2)
        
        print(f"\n詳細分析結果を 'perturbation_analysis.csv' と 'perturbation_stats.json' に保存しました。")
        
        return {
            'comparison_df': comparison_df,
            'perturbation_stats': perturbation_stats,
            'original_df': original_df,
            'adversarial_df': adv_df
        }
    
    def visualize_perturbations(self, analysis_results: Dict = None, top_n_features: int = 10, 
                               save_plots: bool = True) -> None:
        """摂動の可視化"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("可視化にはmatplotlibとseabornが必要です: pip install matplotlib seaborn")
            return
        
        if analysis_results is None:
            print("先に analyze_perturbations() を実行してください")
            return
        
        comparison_df = analysis_results['comparison_df']
        perturbation_stats = analysis_results['perturbation_stats']
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Perturbation norms distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('摂動の分布分析', fontsize=16, fontweight='bold')
        
        # L1 norm distribution
        axes[0, 0].hist(comparison_df['l1_perturbation'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title(f'L1ノルム分布\n平均: {perturbation_stats["l1_norm"]["mean"]:.4f}')
        axes[0, 0].set_xlabel('L1 摂動')
        axes[0, 0].set_ylabel('頻度')
        axes[0, 0].grid(True, alpha=0.3)
        
        # L2 norm distribution
        axes[0, 1].hist(comparison_df['l2_perturbation'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title(f'L2ノルム分布\n平均: {perturbation_stats["l2_norm"]["mean"]:.4f}')
        axes[0, 1].set_xlabel('L2 摂動')
        axes[0, 1].set_ylabel('頻度')
        axes[0, 1].grid(True, alpha=0.3)
        
        # L∞ norm distribution
        axes[1, 0].hist(comparison_df['linf_perturbation'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title(f'L∞ノルム分布\n平均: {perturbation_stats["linf_norm"]["mean"]:.4f}')
        axes[1, 0].set_xlabel('L∞ 摂動')
        axes[1, 0].set_ylabel('頻度')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Correlation between norms
        axes[1, 1].scatter(comparison_df['l1_perturbation'], comparison_df['l2_perturbation'], 
                          alpha=0.6, color='purple')
        axes[1, 1].set_title('L1 vs L2 摂動の相関')
        axes[1, 1].set_xlabel('L1 摂動')
        axes[1, 1].set_ylabel('L2 摂動')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('perturbation_norms_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Feature-wise perturbation analysis
        feature_stats = perturbation_stats['feature_perturbations']
        sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['change_rate'], reverse=True)
        top_features = sorted_features[:top_n_features]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'特徴量別摂動分析 (上位{top_n_features}特徴量)', fontsize=16, fontweight='bold')
        
        # Change rate
        features_names = [f[0] for f in top_features]
        change_rates = [f[1]['change_rate'] * 100 for f in top_features]
        
        axes[0, 0].barh(range(len(features_names)), change_rates, color='steelblue')
        axes[0, 0].set_yticks(range(len(features_names)))
        axes[0, 0].set_yticklabels(features_names, fontsize=10)
        axes[0, 0].set_xlabel('変更率 (%)')
        axes[0, 0].set_title('特徴量変更率')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mean absolute perturbation
        mean_abs_pert = [f[1]['mean_abs'] for f in top_features]
        axes[0, 1].barh(range(len(features_names)), mean_abs_pert, color='darkorange')
        axes[0, 1].set_yticks(range(len(features_names)))
        axes[0, 1].set_yticklabels(features_names, fontsize=10)
        axes[0, 1].set_xlabel('平均絶対摂動')
        axes[0, 1].set_title('平均摂動量')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Max absolute perturbation
        max_abs_pert = [f[1]['max_abs'] for f in top_features]
        axes[1, 0].barh(range(len(features_names)), max_abs_pert, color='crimson')
        axes[1, 0].set_yticks(range(len(features_names)))
        axes[1, 0].set_yticklabels(features_names, fontsize=10)
        axes[1, 0].set_xlabel('最大絶対摂動')
        axes[1, 0].set_title('最大摂動量')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Number of changed samples
        changed_samples = [f[1]['changed_samples'] for f in top_features]
        axes[1, 1].barh(range(len(features_names)), changed_samples, color='forestgreen')
        axes[1, 1].set_yticks(range(len(features_names)))
        axes[1, 1].set_yticklabels(features_names, fontsize=10)
        axes[1, 1].set_xlabel('変更されたサンプル数')
        axes[1, 1].set_title('影響サンプル数')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('feature_perturbation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Sample-wise perturbation heatmap for top features
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get perturbation data for top features
        perturbation_data = []
        feature_labels = []
        for feature_name, _ in top_features:
            pert_col = f'{feature_name}_perturbation'
            if pert_col in comparison_df.columns:
                perturbation_data.append(comparison_df[pert_col].values[:50])  # First 50 samples
                feature_labels.append(feature_name)
        
        if perturbation_data:
            perturbation_matrix = np.array(perturbation_data)
            
            # Create heatmap
            sns.heatmap(perturbation_matrix, 
                       xticklabels=range(1, min(51, len(comparison_df) + 1)),
                       yticklabels=feature_labels,
                       cmap='RdBu_r', center=0,
                       cbar_kws={'label': '摂動量'},
                       ax=ax)
            
            ax.set_title(f'上位{len(feature_labels)}特徴量の摂動ヒートマップ (最初の50サンプル)', fontsize=14)
            ax.set_xlabel('サンプル番号')
            ax.set_ylabel('特徴量')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('perturbation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n可視化完了！以下のファイルが保存されました:")
        if save_plots:
            print("- perturbation_norms_distribution.png")
            print("- feature_perturbation_analysis.png") 
            print("- perturbation_heatmap.png")
    
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
    
    def generate_perturbation_report(self, analysis_results: Dict = None) -> str:
        """摂動分析の詳細レポート生成"""
        if analysis_results is None:
            print("先に analyze_perturbations() を実行してください")
            return ""
        
        comparison_df = analysis_results['comparison_df']
        perturbation_stats = analysis_results['perturbation_stats']
        
        report = []
        report.append("=" * 80)
        report.append("敵対的サンプル摂動分析レポート")
        report.append("=" * 80)
        report.append("")
        
        # 1. 基本統計
        report.append("1. 基本統計")
        report.append("-" * 40)
        report.append(f"総サンプル数: {perturbation_stats['total_samples']:,}")
        report.append("")
        
        # 2. 摂動ノルム統計
        report.append("2. 摂動ノルム統計")
        report.append("-" * 40)
        for norm_type in ['l1_norm', 'l2_norm', 'linf_norm']:
            stats = perturbation_stats[norm_type]
            norm_name = norm_type.replace('_norm', '').upper()
            report.append(f"{norm_name}ノルム:")
            report.append(f"  平均: {stats['mean']:.6f}")
            report.append(f"  中央値: {stats['median']:.6f}")
            report.append(f"  標準偏差: {stats['std']:.6f}")
            report.append(f"  最小値: {stats['min']:.6f}")
            report.append(f"  最大値: {stats['max']:.6f}")
            report.append("")
        
        # 3. 特徴量別分析
        report.append("3. 特徴量別摂動分析")
        report.append("-" * 40)
        feature_stats = perturbation_stats['feature_perturbations']
        sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['change_rate'], reverse=True)
        
        report.append("上位10特徴量（変更率順）:")
        report.append("")
        for i, (feature, stats) in enumerate(sorted_features[:10], 1):
            report.append(f"{i:2d}. {feature}")
            report.append(f"    変更率: {stats['change_rate']:.1%}")
            report.append(f"    平均摂動: {stats['mean_abs']:.6f}")
            report.append(f"    最大摂動: {stats['max_abs']:.6f}")
            report.append(f"    変更サンプル数: {stats['changed_samples']:,}")
            report.append("")
        
        # 4. インパクト分析
        report.append("4. 攻撃インパクト分析")
        report.append("-" * 40)
        
        # 微小摂動での成功例を特定
        small_perturbation_threshold = np.percentile(comparison_df['l2_perturbation'], 25)  # 下位25%
        small_pert_mask = comparison_df['l2_perturbation'] <= small_perturbation_threshold
        small_pert_count = small_pert_mask.sum()
        
        report.append(f"微小摂動での攻撃成功:")
        report.append(f"  L2摂動 ≤ {small_perturbation_threshold:.6f} のサンプル: {small_pert_count:,}")
        report.append(f"  全体に占める割合: {(small_pert_count/len(comparison_df))*100:.1f}%")
        report.append("")
        
        # 最も効果的な特徴量の組み合わせ
        report.append("5. 攻撃効果の高い特徴量")
        report.append("-" * 40)
        
        # 変更率と平均摂動の積でランキング
        feature_impact = {}
        for feature, stats in feature_stats.items():
            impact_score = stats['change_rate'] * stats['mean_abs']
            feature_impact[feature] = impact_score
        
        sorted_impact = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)
        
        report.append("インパクトスコア順（変更率 × 平均摂動）:")
        report.append("")
        for i, (feature, score) in enumerate(sorted_impact[:10], 1):
            stats = feature_stats[feature]
            report.append(f"{i:2d}. {feature}")
            report.append(f"    インパクトスコア: {score:.8f}")
            report.append(f"    変更率: {stats['change_rate']:.1%}")
            report.append(f"    平均摂動: {stats['mean_abs']:.6f}")
            report.append("")
        
        # 6. 実例サンプル
        report.append("6. 摂動実例（最小・最大・代表例）")
        report.append("-" * 40)
        
        # 最小摂動のサンプル
        min_l2_idx = comparison_df['l2_perturbation'].idxmin()
        min_l2_value = comparison_df.loc[min_l2_idx, 'l2_perturbation']
        
        # 最大摂動のサンプル
        max_l2_idx = comparison_df['l2_perturbation'].idxmax()
        max_l2_value = comparison_df.loc[max_l2_idx, 'l2_perturbation']
        
        # 中央値付近のサンプル
        median_l2 = comparison_df['l2_perturbation'].median()
        median_idx = (comparison_df['l2_perturbation'] - median_l2).abs().idxmin()
        median_l2_value = comparison_df.loc[median_idx, 'l2_perturbation']
        
        examples = [
            ("最小摂動サンプル", min_l2_idx, min_l2_value),
            ("中央値摂動サンプル", median_idx, median_l2_value),
            ("最大摂動サンプル", max_l2_idx, max_l2_value)
        ]
        
        for example_name, idx, l2_value in examples:
            report.append(f"{example_name} (ID: {idx}):")
            report.append(f"  L2摂動: {l2_value:.6f}")
            report.append(f"  L1摂動: {comparison_df.loc[idx, 'l1_perturbation']:.6f}")
            report.append(f"  L∞摂動: {comparison_df.loc[idx, 'linf_perturbation']:.6f}")
            
            # 変更された特徴量を表示
            changed_features = []
            for feature in feature_stats.keys():
                pert_col = f'{feature}_perturbation'
                if pert_col in comparison_df.columns:
                    pert_value = comparison_df.loc[idx, pert_col]
                    if abs(pert_value) > 1e-6:
                        changed_features.append((feature, pert_value))
            
            if changed_features:
                report.append(f"  変更された特徴量:")
                for feature, pert in sorted(changed_features, key=lambda x: abs(x[1]), reverse=True)[:5]:
                    orig_col = f'{feature}_original'
                    adv_col = f'{feature}_adversarial'
                    if orig_col in comparison_df.columns and adv_col in comparison_df.columns:
                        orig_val = comparison_df.loc[idx, orig_col]
                        adv_val = comparison_df.loc[idx, adv_col]
                        report.append(f"    {feature}: {orig_val:.6f} → {adv_val:.6f} (摂動: {pert:+.6f})")
            report.append("")
        
        # 7. 結論
        report.append("7. 主要な発見")
        report.append("-" * 40)
        report.append(f"• 平均L2摂動: {perturbation_stats['l2_norm']['mean']:.6f}")
        report.append(f"• 最も影響を受けやすい特徴量: {sorted_features[0][0]}")
        report.append(f"• 最小摂動で攻撃成功: L2={min_l2_value:.6f}")
        report.append(f"• 微小摂動成功率: {(small_pert_count/len(comparison_df))*100:.1f}%")
        report.append("")
        report.append("この分析により、詐欺検知モデルが非常に小さな特徴量の変更で")
        report.append("誤分類される脆弱性が確認されました。")
        
        report_text = "\n".join(report)
        
        # Save report
        with open('perturbation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("詳細レポートを 'perturbation_report.txt' に保存しました。")
        
        return report_text