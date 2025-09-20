#!/usr/bin/env python3
"""
統合詐欺検知攻撃システム (Integrated Fraud Detection Attack System)

このシステムは以下の機能を提供します：
1. データの前処理と正規化
2. XGBoostモデルの訓練
3. LT-Attack攻撃の実行
4. 摂動分析と可視化
5. 詳細レポートの生成
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import json
import numpy as np
import os
import subprocess
from typing import Dict, Tuple, Any, Optional

class FraudAttackSystem:
    """統合詐欺検知攻撃システム"""
    
    def __init__(self, data_path: str, normalize_features: bool = True):
        """
        システムの初期化
        
        Args:
            data_path: CSVデータファイルのパス
            normalize_features: 特徴量を正規化するかどうか
        """
        self.data_path = data_path
        self.normalize_features = normalize_features
        self.model = None
        self.scaler = StandardScaler() if normalize_features else None
        self.feature_info = None
        self.results = {}
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """データの読み込みと準備"""
        print("=== データ読み込みと準備 ===")
        
        # データ読み込み
        df = pd.read_csv(self.data_path)
        print(f"データ形状: {df.shape}")
        
        # has_missingカラムが存在する場合は削除
        if 'has_missing' in df.columns:
            df = df.drop('has_missing', axis=1)
            print("has_missingカラムを削除しました")
        
        # fraud_bool分布の確認
        fraud_counts = df['fraud_bool'].value_counts()
        print(f"fraud_bool分布: {fraud_counts.to_dict()}")
        
        # 特徴量とターゲットの分離
        X = df.drop('fraud_bool', axis=1)
        y = df['fraud_bool']
        
        # 詐欺ケースの表示
        fraud_cases = df[df['fraud_bool'] == 1]
        print(f"詐欺ケース数: {len(fraud_cases)}")
        if len(fraud_cases) > 0:
            print("詐欺ケースの例:")
            print(fraud_cases.head())
        
        return df, X, y
    
    def train_model(self) -> Dict[str, Any]:
        """XGBoostモデルの訓練"""
        print("\n=== モデル訓練 ===")
        
        df, X, y = self.load_and_prepare_data()
        
        # 特徴量の正規化
        if self.normalize_features:
            print("特徴量を正規化中...")
            X_processed = pd.DataFrame(
                self.scaler.fit_transform(X), 
                columns=X.columns,
                index=X.index
            )
            print("正規化完了 (平均=0, 標準偏差=1)")
        else:
            X_processed = X.copy()
            print("正規化をスキップ")
        
        # 訓練・テスト分割
        if len(y.unique()) > 1:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError:
                # 層化抽出できない場合は通常分割
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=0.2, random_state=42
                )
        else:
            # 単一クラスの場合
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42
            )
        
        # XGBoostモデル訓練
        print("XGBoostモデルを訓練中...")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train, y_train)
        
        # モデル評価
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"テスト精度: {accuracy:.4f}")
        
        # 特徴量情報の保存
        self.feature_info = {
            'features': list(X.columns),
            'num_features': len(X.columns),
            'num_classes': len(y.unique()),
            'feature_start': 1,
            'normalization': self.normalize_features
        }
        
        if self.normalize_features:
            self.feature_info.update({
                'scaler_mean': self.scaler.mean_.tolist(),
                'scaler_scale': self.scaler.scale_.tolist()
            })
        
        # ファイル保存
        self._save_model_files(X_processed, y)
        
        self.results['training'] = {
            'accuracy': accuracy,
            'feature_info': self.feature_info,
            'fraud_cases': len(df[df['fraud_bool'] == 1])
        }
        
        return self.results['training']
    
    def _save_model_files(self, X: pd.DataFrame, y: pd.Series):
        """モデルと関連ファイルの保存"""
        print("モデルファイルを保存中...")
        
        # モデルをJSON形式で保存
        self.model.get_booster().dump_model('attack_model.json', dump_format='json')
        self._fix_json_format()
        
        # 特徴量情報保存
        with open('feature_info.json', 'w') as f:
            json.dump(self.feature_info, f, indent=2)
        
        # 詐欺ケースのテストデータ作成
        self._create_fraud_test_data(X, y)
        
        print("ファイル保存完了")
    
    def _fix_json_format(self):
        """JSONフォーマットの修正（特徴量名→インデックス）"""
        feature_to_index = {name: idx for idx, name in enumerate(self.feature_info['features'])}
        
        with open('attack_model.json', 'r') as f:
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
        
        with open('attack_model.json', 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def _create_fraud_test_data(self, X: pd.DataFrame, y: pd.Series):
        """詐欺ケースのテストデータ作成"""
        fraud_indices = y[y == 1].index
        
        if len(fraud_indices) == 0:
            print("警告: 詐欺ケースが見つかりません")
            return
        
        fraud_X = X.loc[fraud_indices]
        fraud_y = y.loc[fraud_indices]
        
        # 最大500件まで
        max_samples = min(500, len(fraud_X))
        fraud_X = fraud_X.head(max_samples)
        fraud_y = fraud_y.head(max_samples)
        
        # LIBSVM形式で保存
        with open('fraud_test.libsvm', 'w') as f:
            for i, (idx, row) in enumerate(fraud_X.iterrows()):
                libsvm_line = f"{int(fraud_y.iloc[i])}"
                for j, val in enumerate(row):
                    if float(val) != 0:
                        libsvm_line += f" {j+1}:{val:.6f}"
                f.write(libsvm_line + "\n")
        
        print(f"LIBSVM形式でテストデータを保存: {len(fraud_X)} サンプル")
    
    def create_attack_config(self, num_threads: int = 10) -> str:
        """攻撃用設定ファイルの作成"""
        fraud_count = self.results['training']['fraud_cases']
        
        config = {
            "search_mode": "lt-attack",
            "norm_type": 2,
            "num_point": min(500, fraud_count),
            "num_threads": num_threads,
            "num_attack_per_point": num_threads,
            "enable_early_return": True,
            "model": "XGBoost/attack_model.json",
            "inputs": "XGBoost/fraud_test.libsvm",
            "num_classes": self.feature_info['num_classes'],
            "num_features": self.feature_info['num_features'],
            "feature_start": 1,
            "adv_training_path": "XGBoost/adversarial_examples.libsvm"
        }
        
        config_path = '../configs/attack_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"攻撃設定ファイルを作成: {config_path}")
        return config_path
    
    def run_attack(self, config_path: str) -> bool:
        """LT-Attack攻撃の実行"""
        print(f"\n=== LT-Attack攻撃実行 ===")
        print(f"設定ファイル: {config_path}")
        
        try:
            # カレントディレクトリを親ディレクトリに変更
            original_dir = os.getcwd()
            parent_dir = os.path.dirname(os.getcwd())
            os.chdir(parent_dir)
            
            # 攻撃実行
            cmd = ['./lt_attack', config_path]
            print(f"実行コマンド: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # カレントディレクトリを戻す
            os.chdir(original_dir)
            
            if result.returncode == 0:
                print("攻撃が正常に完了しました")
                print("出力:")
                print(result.stdout[-500:])  # 最後の500文字
                return True
            else:
                print(f"攻撃が失敗しました (戻り値: {result.returncode})")
                print("エラー:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            os.chdir(original_dir)
            print("攻撃がタイムアウトしました")
            return False
        except Exception as e:
            os.chdir(original_dir)
            print(f"攻撃実行中にエラーが発生しました: {e}")
            return False
    
    def analyze_attack_results(self) -> Dict[str, Any]:
        """攻撃結果の分析"""
        print(f"\n=== 攻撃結果分析 ===")
        
        # 必要ファイルの確認
        required_files = ['adversarial_examples.libsvm', 'feature_info.json']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"必要ファイルが見つかりません: {missing_files}")
            return {}
        
        # 元データと敵対的サンプルを読み込み
        original_data, adversarial_data = self._load_attack_data()
        
        if len(adversarial_data) == 0:
            print("敵対的サンプルが見つかりません")
            return {}
        
        # 摂動分析
        analysis_results = self._analyze_perturbations(original_data, adversarial_data)
        
        # 結果保存
        self._save_analysis_results(analysis_results)
        
        self.results['analysis'] = analysis_results
        return analysis_results
    
    def _load_attack_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """攻撃データの読み込み"""
        # 元データ読み込み
        df = pd.read_csv(self.data_path)
        if 'has_missing' in df.columns:
            df = df.drop('has_missing', axis=1)
        
        fraud_cases = df[df['fraud_bool'] == 1]
        X_original = fraud_cases.drop('fraud_bool', axis=1)
        
        if self.normalize_features:
            X_original = pd.DataFrame(
                self.scaler.transform(X_original),
                columns=X_original.columns
            )
        
        # 敵対的サンプル読み込み
        adversarial_data = []
        with open('adversarial_examples.libsvm', 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    feature_values = [0.0] * self.feature_info['num_features']
                    
                    for feature_pair in parts[1:]:
                        if ':' in feature_pair:
                            feature_idx, value = feature_pair.split(':')
                            feature_idx = int(feature_idx) - 1
                            if 0 <= feature_idx < len(feature_values):
                                feature_values[feature_idx] = float(value)
                    
                    adversarial_data.append(feature_values)
        
        return X_original.values, np.array(adversarial_data)
    
    def _analyze_perturbations(self, original_data: np.ndarray, adversarial_data: np.ndarray) -> Dict[str, Any]:
        """摂動の詳細分析"""
        print("摂動を分析中...")
        
        # 摂動計算
        num_samples = min(len(original_data), len(adversarial_data))
        original = original_data[:num_samples]
        adversarial = adversarial_data[:num_samples]
        
        perturbations = adversarial - original
        
        # ノルム計算
        l1_norms = np.sum(np.abs(perturbations), axis=1)
        l2_norms = np.sqrt(np.sum(perturbations**2, axis=1))
        linf_norms = np.max(np.abs(perturbations), axis=1)
        
        # 特徴量別統計
        features = self.feature_info['features']
        feature_stats = {}
        
        for i, feature in enumerate(features):
            feature_pert = perturbations[:, i]
            feature_stats[feature] = {
                'mean_abs': float(np.mean(np.abs(feature_pert))),
                'std': float(np.std(feature_pert)),
                'max_abs': float(np.max(np.abs(feature_pert))),
                'changed_samples': int(np.sum(np.abs(feature_pert) > 1e-6)),
                'change_rate': float(np.sum(np.abs(feature_pert) > 1e-6) / len(feature_pert))
            }
        
        analysis_results = {
            'total_samples': num_samples,
            'l1_norm': {
                'mean': float(np.mean(l1_norms)),
                'median': float(np.median(l1_norms)),
                'std': float(np.std(l1_norms)),
                'min': float(np.min(l1_norms)),
                'max': float(np.max(l1_norms))
            },
            'l2_norm': {
                'mean': float(np.mean(l2_norms)),
                'median': float(np.median(l2_norms)),
                'std': float(np.std(l2_norms)),
                'min': float(np.min(l2_norms)),
                'max': float(np.max(l2_norms))
            },
            'linf_norm': {
                'mean': float(np.mean(linf_norms)),
                'median': float(np.median(linf_norms)),
                'std': float(np.std(linf_norms)),
                'min': float(np.min(linf_norms)),
                'max': float(np.max(linf_norms))
            },
            'feature_perturbations': feature_stats,
            'perturbations': perturbations,
            'original_data': original,
            'adversarial_data': adversarial
        }
        
        print(f"分析完了: {num_samples}サンプル")
        print(f"平均L2摂動: {analysis_results['l2_norm']['mean']:.6f}")
        
        return analysis_results
    
    def _save_analysis_results(self, analysis_results: Dict[str, Any]):
        """分析結果の保存"""
        # 詳細データをCSVで保存
        features = self.feature_info['features']
        num_samples = analysis_results['total_samples']
        
        # 比較データフレーム作成
        comparison_data = []
        for i in range(num_samples):
            row = {'sample_id': i}
            
            # ノルム情報
            perturbations = analysis_results['perturbations'][i]
            row['l1_perturbation'] = np.sum(np.abs(perturbations))
            row['l2_perturbation'] = np.sqrt(np.sum(perturbations**2))
            row['linf_perturbation'] = np.max(np.abs(perturbations))
            
            # 特徴量別データ
            for j, feature in enumerate(features):
                row[f'{feature}_original'] = analysis_results['original_data'][i, j]
                row[f'{feature}_adversarial'] = analysis_results['adversarial_data'][i, j]
                row[f'{feature}_perturbation'] = perturbations[j]
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('attack_analysis.csv', index=False)
        
        # 統計情報をJSON保存（numpy配列を除く）
        stats_data = analysis_results.copy()
        stats_data.pop('perturbations', None)
        stats_data.pop('original_data', None)
        stats_data.pop('adversarial_data', None)
        
        with open('attack_stats.json', 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        print("分析結果を保存しました:")
        print("- attack_analysis.csv: 詳細比較データ")
        print("- attack_stats.json: 統計情報")
    
    def generate_report(self) -> str:
        """総合レポートの生成"""
        print(f"\n=== レポート生成 ===")
        
        if 'analysis' not in self.results:
            print("分析結果がありません。先に analyze_attack_results() を実行してください。")
            return ""
        
        analysis = self.results['analysis']
        training = self.results['training']
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("詐欺検知モデル攻撃分析レポート")
        report_lines.append("="*80)
        report_lines.append("")
        
        # 基本情報
        report_lines.append("1. 基本情報")
        report_lines.append("-"*40)
        report_lines.append(f"データセット: {self.data_path}")
        report_lines.append(f"正規化: {'適用' if self.normalize_features else '未適用'}")
        report_lines.append(f"特徴量数: {training['feature_info']['num_features']}")
        report_lines.append(f"詐欺ケース数: {training['fraud_cases']}")
        report_lines.append(f"モデル精度: {training['accuracy']:.4f}")
        report_lines.append(f"攻撃サンプル数: {analysis['total_samples']}")
        report_lines.append("")
        
        # 摂動統計
        report_lines.append("2. 摂動統計")
        report_lines.append("-"*40)
        for norm_type in ['l1_norm', 'l2_norm', 'linf_norm']:
            stats = analysis[norm_type]
            norm_name = norm_type.replace('_norm', '').upper()
            report_lines.append(f"{norm_name}ノルム:")
            report_lines.append(f"  平均: {stats['mean']:.6f}")
            report_lines.append(f"  中央値: {stats['median']:.6f}")
            report_lines.append(f"  最小値: {stats['min']:.6f}")
            report_lines.append(f"  最大値: {stats['max']:.6f}")
        report_lines.append("")
        
        # 主要特徴量
        report_lines.append("3. 最も影響を受けた特徴量")
        report_lines.append("-"*40)
        
        feature_stats = analysis['feature_perturbations']
        sorted_features = sorted(feature_stats.items(), 
                               key=lambda x: x[1]['change_rate'], reverse=True)
        
        for i, (feature, stats) in enumerate(sorted_features[:10], 1):
            report_lines.append(f"{i:2d}. {feature}")
            report_lines.append(f"    変更率: {stats['change_rate']:.1%}")
            report_lines.append(f"    平均摂動: {stats['mean_abs']:.6f}")
            report_lines.append(f"    最大摂動: {stats['max_abs']:.6f}")
        report_lines.append("")
        
        # 結論
        report_lines.append("4. 主要な発見")
        report_lines.append("-"*40)
        
        if self.normalize_features:
            is_small = analysis['l2_norm']['mean'] < 1.0
            perturbation_scale = "微小" if is_small else "大きな"
        else:
            perturbation_scale = "大きな"
        
        report_lines.append(f"• 平均L2摂動: {analysis['l2_norm']['mean']:.6f}")
        report_lines.append(f"• 摂動規模: {perturbation_scale}摂動")
        report_lines.append(f"• 最も脆弱な特徴量: {sorted_features[0][0]}")
        
        if self.normalize_features and is_small:
            report_lines.append("• 正規化により微小摂動での攻撃が成功")
            report_lines.append("• モデルの頑健性に課題があることが判明")
        else:
            report_lines.append("• 攻撃には比較的大きな特徴量変更が必要")
            report_lines.append("• モデルは一定の頑健性を持つ")
        
        report_text = "\n".join(report_lines)
        
        # レポート保存
        with open('attack_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("レポートを 'attack_report.txt' に保存しました")
        return report_text
    
    def run_full_pipeline(self, num_threads: int = 10) -> Dict[str, Any]:
        """完全なパイプラインの実行"""
        print("="*80)
        print("詐欺検知攻撃システム - 完全パイプライン実行")
        print("="*80)
        
        try:
            # 1. モデル訓練
            training_results = self.train_model()
            
            # 2. 攻撃設定作成
            print(f"\n=== 攻撃設定作成 ===")
            config_path = self.create_attack_config(num_threads)
            
            # 3. 攻撃実行
            attack_success = self.run_attack(config_path)
            
            if not attack_success:
                print("攻撃が失敗したため、パイプラインを中断します")
                return self.results
            
            # 4. 結果分析
            analysis_results = self.analyze_attack_results()
            
            # 5. レポート生成
            if analysis_results:
                report = self.generate_report()
                print(f"\n=== パイプライン完了 ===")
                print("生成されたファイル:")
                print("- attack_model.json: 訓練済みモデル")
                print("- attack_analysis.csv: 詳細分析データ")
                print("- attack_stats.json: 統計情報")
                print("- attack_report.txt: 総合レポート")
            
            return self.results
            
        except Exception as e:
            print(f"パイプライン実行中にエラーが発生しました: {e}")
            return self.results

def main():
    """メイン関数 - コマンドライン実行用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='詐欺検知攻撃システム')
    parser.add_argument('data_path', help='CSVデータファイルのパス')
    parser.add_argument('--no-normalize', action='store_true', 
                       help='特徴量の正規化を無効にする')
    parser.add_argument('--threads', type=int, default=10,
                       help='攻撃に使用するスレッド数 (default: 10)')
    
    args = parser.parse_args()
    
    # システム実行
    system = FraudAttackSystem(
        data_path=args.data_path,
        normalize_features=not args.no_normalize
    )
    
    results = system.run_full_pipeline(num_threads=args.threads)
    
    if 'analysis' in results:
        print(f"\n攻撃分析完了！")
        print(f"平均L2摂動: {results['analysis']['l2_norm']['mean']:.6f}")
    else:
        print(f"\n攻撃が完了しませんでした")

if __name__ == "__main__":
    main()