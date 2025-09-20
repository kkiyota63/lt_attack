# 詐欺検知攻撃システム (Fraud Detection Attack System)

このシステムは、XGBoostベースの詐欺検知モデルに対するLT-Attack攻撃を実行し、モデルの脆弱性を分析するためのツールです。

## 🎯 機能

- **データ前処理**: CSVデータの読み込みと正規化
- **モデル訓練**: XGBoostを使った詐欺検知モデルの訓練
- **攻撃実行**: LT-Attackによる敵対的サンプル生成
- **摂動分析**: 元データと敵対的サンプルの詳細比較
- **可視化とレポート**: グラフ生成と総合分析レポート

## 📋 前提条件

### 必要なソフトウェア
```bash
# Python 3.7+
# 必要なライブラリ
pip install pandas xgboost scikit-learn numpy matplotlib seaborn
```

### データ形式
- CSV形式
- `fraud_bool`カラム（0: 正常, 1: 詐欺）が必要
- `has_missing`カラムがある場合は自動で削除

## 🚀 基本的な使い方

### 1. ワンコマンド実行（推奨）

```bash
cd XGBoost
python3 fraud_attack_system.py top100_cleaned_data.csv
```

### 2. オプション付き実行

```bash
# 正規化を無効にする場合
python3 fraud_attack_system.py top100_cleaned_data.csv --no-normalize

# スレッド数を指定
python3 fraud_attack_system.py top100_cleaned_data.csv --threads 20
```

### 3. 段階的実行

```python
from fraud_attack_system import FraudAttackSystem

# システム初期化
system = FraudAttackSystem('top100_cleaned_data.csv')

# 1. モデル訓練
system.train_model()

# 2. 攻撃設定作成
config_path = system.create_attack_config()

# 3. 攻撃実行
system.run_attack(config_path)

# 4. 結果分析
system.analyze_attack_results()

# 5. レポート生成
system.generate_report()
```

## 📊 出力ファイル

### 実行後に生成されるファイル

| ファイル名 | 説明 |
|-----------|------|
| `attack_model.json` | 訓練済みXGBoostモデル（JSON形式） |
| `attack_analysis.csv` | 元データと敵対的サンプルの詳細比較 |
| `attack_stats.json` | 摂動統計情報 |
| `attack_report.txt` | 総合分析レポート |
| `fraud_test.libsvm` | 攻撃対象の詐欺ケース（LIBSVM形式） |
| `adversarial_examples.libsvm` | 生成された敵対的サンプル |

### レポートの内容

```
詐欺検知モデル攻撃分析レポート
================================================================================

1. 基本情報
----------------------------------------
データセット: top100_cleaned_data.csv
正規化: 適用
特徴量数: 21
詐欺ケース数: 1
モデル精度: 1.0000
攻撃サンプル数: 1

2. 摂動統計
----------------------------------------
L1ノルム:
  平均: 2.345678
  中央値: 2.345678
  最小値: 2.345678
  最大値: 2.345678

3. 最も影響を受けた特徴量
----------------------------------------
 1. velocity_6h
    変更率: 100.0%
    平均摂動: 1.234567
    最大摂動: 1.234567

4. 主要な発見
----------------------------------------
• 平均L2摂動: 1.234567
• 摂動規模: 微小摂動
• 最も脆弱な特徴量: velocity_6h
• 正規化により微小摂動での攻撃が成功
• モデルの頑健性に課題があることが判明
```

## 🔧 高度な使い方

### カスタム分析

```python
# 特定の分析のみ実行
system = FraudAttackSystem('data.csv')

# モデル訓練のみ
system.train_model()

# 既存の攻撃結果を分析
system.analyze_attack_results()
```

### 正規化の有無による比較

```python
# 正規化あり
system_norm = FraudAttackSystem('data.csv', normalize_features=True)
results_norm = system_norm.run_full_pipeline()

# 正規化なし
system_raw = FraudAttackSystem('data.csv', normalize_features=False)
results_raw = system_raw.run_full_pipeline()

# 結果比較
print(f"正規化あり L2摂動: {results_norm['analysis']['l2_norm']['mean']:.6f}")
print(f"正規化なし L2摂動: {results_raw['analysis']['l2_norm']['mean']:.6f}")
```

## 📈 結果の解釈

### 摂動の大きさ
- **L2ノルム < 1.0**: 微小摂動（正規化済みデータで一般的）
- **L2ノルム > 10.0**: 大きな摂動（生データで一般的）

### 攻撃成功の意味
- 元の詐欺ケース（`fraud_bool=1`）が正常（`fraud_bool=0`）と誤分類される
- 小さな摂動で達成できるほど、モデルの脆弱性が高い

### 特徴量の重要度
- **変更率**: その特徴量が変更されたサンプルの割合
- **平均摂動**: 特徴量の平均的な変更量
- **最大摂動**: 特徴量の最大変更量

## ⚠️ 注意事項

1. **データサイズ**: 大きなデータセットでは実行時間が長くなります
2. **メモリ使用量**: 特徴量数が多い場合はメモリ使用量に注意
3. **攻撃時間**: LT-Attackは時間がかかる場合があります（タイムアウト: 5分）
4. **詐欺ケース数**: 詐欺ケースが少ない場合は分析結果が限定的になります

## 🐛 トラブルシューティング

### よくあるエラー

**1. `FileNotFoundError`**
```bash
# データファイルのパスを確認
ls -la top100_cleaned_data.csv
```

**2. `lt_attack command not found`**
```bash
# 親ディレクトリにlt_attackがあることを確認
ls -la ../lt_attack
```

**3. `No fraud cases found`**
```bash
# データに詐欺ケース（fraud_bool=1）があることを確認
python3 -c "import pandas as pd; print(pd.read_csv('data.csv')['fraud_bool'].value_counts())"
```

**4. メモリ不足**
```bash
# より少ないスレッド数で実行
python3 fraud_attack_system.py data.csv --threads 5
```

### デバッグモード

```python
# より詳細なログを出力
import logging
logging.basicConfig(level=logging.DEBUG)

system = FraudAttackSystem('data.csv')
system.run_full_pipeline()
```

## 📚 参考文献

- Zhang, C., Zhang, H., & Hsieh, C. J. (2020). An Efficient Adversarial Attack for Tree Ensembles. NeurIPS 2020.
- LT-Attack: https://github.com/chong-z/tree-ensemble-attack

## 📄 ライセンス

このプロジェクトは元のLT-Attackプロジェクトのライセンスに従います。