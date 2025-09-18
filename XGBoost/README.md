# 詐欺検知回避攻撃システム (リファクタリング版)

XGBoostベースの詐欺検知モデルに対する敵対的攻撃システム。統合されたAPIで簡単に使用できます。

## 概要

詐欺検知システムを騙すために、詐欺取引（label=1）を非詐欺取引（label=0）として誤分類させる敵対的サンプルを生成します。LT-Attack（Leaf Tuple Attack）アルゴリズムを使用して、最小限の変更で効果的な攻撃を実現します。

## 対応プラットフォーム

- ✅ **Windows 10/11** (MSYS2/MinGW-w64)
- ✅ **Ubuntu/Linux** (18.04以上)
- ✅ **macOS** (Intel & Apple Silicon)

## システム要件

- Python 3.7+
- C++ コンパイラ (g++)
- Boost C++ ライブラリ
- Make ユーティリティ

## 🚀 簡単インストール（推奨）

### 自動インストール
```bash
# 1. 依存関係の自動インストール
python install.py

# 2. システムのビルド
python build.py

# 3. 動作確認
python install.py check
```

### プラットフォーム別インストール

<details>
<summary><b>🪟 Windows (MSYS2)</b></summary>

```bash
# 1. MSYS2のインストール
# https://www.msys2.org/ からダウンロード・インストール

# 2. MSYS2 MINGW64ターミナルで実行
pacman -Syu
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-boost mingw-w64-x86_64-python mingw-w64-x86_64-python-pip make

# 3. Pythonパッケージ
pip install xgboost scikit-learn pandas numpy

# 4. ビルド
make -f Makefile.windows
```
</details>

<details>
<summary><b>🐧 Ubuntu/Linux</b></summary>

```bash
# 1. 依存関係のインストール
sudo apt-get update
sudo apt-get install build-essential libboost-all-dev python3 python3-pip python3-dev

# 2. Pythonパッケージ
pip3 install xgboost scikit-learn pandas numpy

# 3. ビルド
make -f Makefile.windows  # クロスプラットフォーム対応版
```
</details>

<details>
<summary><b>🍎 macOS</b></summary>

```bash
# 1. Homebrewのインストール (未インストールの場合)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 依存関係のインストール
brew install boost python3

# 3. Pythonパッケージ
pip3 install xgboost scikit-learn pandas numpy

# 4. ビルド
make  # または make -f Makefile.windows
```
</details>

## 使用方法

### 🚀 **簡単実行（推奨）**

**完全パイプライン実行**
```bash
cd XGBoost
python main.py pipeline
```

**個別ステップ実行**
```bash
cd XGBoost

# 1. モデル訓練
python main.py train

# 2. 敵対的攻撃
python main.py attack

# 3. 結果変換・検証
python main.py convert
```

### 📋 **利用可能なコマンド**

```bash
python main.py help      # ヘルプ表示
python main.py train     # モデル訓練
python main.py attack    # 敵対的攻撃実行
python main.py convert   # 結果変換・検証
python main.py verify    # 攻撃検証のみ
python main.py pipeline  # 完全パイプライン
```

### 🔧 **プログラマティック使用**

```python
from fraud_attack_utils import FraudAttackSystem

# システム初期化
system = FraudAttackSystem('Base.csv')

# モデル訓練
results = system.train_model()

# 攻撃後の結果変換・検証
results = system.convert_and_verify_results()
```

## 出力ファイル

### 最終結果
- `adversarial_sample.csv` - 修正済み敵対的サンプル（CSV形式）

### 中間ファイル
- `adversarial_examples.libsvm` - 敵対的サンプル（LIBSVM形式）
- `adversarial_raw.csv` - 未修正敵対的サンプル（CSV形式）

## 結果の解釈

攻撃が成功すると、以下のような出力が表示されます：

```
===== Attack result for example 1/500 Norm(2)=0.000000 =====
All Best Norms: Norm(-1)=0.000000 Norm(1)=0.000000 Norm(2)=0.000000
```

- **Norm(2)=0.000000**: L2ノルムが0に近いほど、元データからの変更が小さい
- **Initial point label:0**: 攻撃により詐欺(1)→非詐欺(0)への誤分類に成功

## データセット

### Base.csv構造
- `fraud_bool`: 詐欺フラグ（0=非詐欺, 1=詐欺）
- 31個の特徴量（収入、年齢、デバイス情報など）
- カテゴリ変数: `payment_type`, `employment_status`, `housing_status`, `source`, `device_os`

### 特徴量の種類
1. **数値特徴量**: 収入、年齢、金額など
2. **カテゴリ特徴量**: 支払い方法、雇用状況、住居状況など
3. **バイナリ特徴量**: フラグ系の特徴量

## トラブルシューティング

### コンパイルエラー
```bash
# Boostが見つからない場合
brew install boost
export BOOST_ROOT=/opt/homebrew/opt/boost

# Makefileの修正が必要な場合
make clean
make
```

### Python エラー
```bash
# 必要なライブラリが不足している場合
pip install xgboost==1.6.2 scikit-learn pandas numpy

# XGBoostのバージョンが古い場合
pip install --upgrade xgboost
```

### 攻撃の失敗
- モデルの性能が低い場合、攻撃対象となる正しく分類されたサンプルが少なくなります
- `num_threads`や`num_attack_per_point`を調整して攻撃強度を変更できます

## アルゴリズム詳細

### LT-Attack（Leaf Tuple Attack）
1. 決定木アンサンブルの各サンプルをLeaf Tupleとして表現
2. Hamming距離1の近傍探索で最適な敵対的サンプルを発見
3. 複数のノルム（L1, L2, L∞）で変更量を最小化

### カテゴリ変数の処理
1. LabelEncoderで数値に変換
2. 攻撃実行（連続値として処理）
3. 最近傍の有効なカテゴリ値に丸め

## セキュリティ考慮事項

**⚠️ 警告: このシステムは研究・教育目的のみに使用してください**

- 実際の詐欺検知システムへの攻撃は違法行為です
- 防御手法の研究や、ロバストなモデル開発にのみ使用してください
- 商用利用前には必ず法的検討を行ってください

## 参考文献

Chong Zhang, Huan Zhang, Cho-Jui Hsieh, "An Efficient Adversarial Attack for Tree Ensembles", NeurIPS 2020

