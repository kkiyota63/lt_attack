# クロスプラットフォーム セットアップガイド

このガイドでは、Windows、Ubuntu/Linux、macOSでの詐欺検知攻撃システムのセットアップ方法を説明します。

## 🌟 推奨: 自動セットアップ

すべてのプラットフォームで以下のコマンドが使用できます：

```bash
# 1. 依存関係の自動インストール
python install.py

# 2. インストール確認
python install.py check

# 3. システムビルド
python build.py

# 4. システム実行
cd XGBoost
python main.py pipeline
```

## 🪟 Windows (MSYS2/MinGW-w64)

### 必要なソフトウェア
- [MSYS2](https://www.msys2.org/) (推奨)
- または [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)

### MSYS2での手順

1. **MSYS2のインストール**
   ```bash
   # https://www.msys2.org/ からダウンロード・インストール
   # インストール後、MSYS2 MINGW64ターミナルを開く
   ```

2. **システムの更新**
   ```bash
   pacman -Syu
   # ターミナルが閉じた場合は再度開いて
   pacman -Su
   ```

3. **開発ツールのインストール**
   ```bash
   pacman -S mingw-w64-x86_64-gcc \
             mingw-w64-x86_64-boost \
             mingw-w64-x86_64-python \
             mingw-w64-x86_64-python-pip \
             make \
             git
   ```

4. **Pythonパッケージのインストール**
   ```bash
   pip install xgboost scikit-learn pandas numpy
   ```

5. **ビルドとテスト**
   ```bash
   # プロジェクトディレクトリで
   python build.py info    # 環境確認
   python build.py         # ビルド
   cd XGBoost
   python main.py help     # テスト
   ```

### トラブルシューティング (Windows)

- **パッケージが見つからない**: MSYS2のパッケージデータベースを更新 `pacman -Sy`
- **コンパイルエラー**: 環境変数PATHにMSYS2のbinディレクトリが含まれているか確認
- **Pythonモジュールエラー**: MSYS2のPythonを使用しているか確認 `which python`

## 🐧 Ubuntu/Linux

### サポートされるディストリビューション
- Ubuntu 18.04以上
- Debian 10以上
- CentOS 8以上
- Fedora 30以上

### Ubuntu/Debianでの手順

1. **システムの更新**
   ```bash
   sudo apt-get update
   sudo apt-get upgrade
   ```

2. **開発ツールのインストール**
   ```bash
   sudo apt-get install build-essential \
                        libboost-all-dev \
                        python3 \
                        python3-pip \
                        python3-dev \
                        git
   ```

3. **Pythonパッケージのインストール**
   ```bash
   pip3 install --user xgboost scikit-learn pandas numpy
   # または
   python3 -m pip install --user xgboost scikit-learn pandas numpy
   ```

4. **ビルドとテスト**
   ```bash
   python3 build.py info   # 環境確認
   python3 build.py        # ビルド
   cd XGBoost
   python3 main.py help    # テスト
   ```

### CentOS/Fedoraでの手順

1. **開発ツールのインストール (CentOS)**
   ```bash
   sudo dnf groupinstall "Development Tools"
   sudo dnf install boost-devel python3 python3-pip python3-devel
   ```

2. **開発ツールのインストール (Fedora)**
   ```bash
   sudo dnf install gcc-c++ boost-devel python3 python3-pip python3-devel make git
   ```

### トラブルシューティング (Linux)

- **Boostが見つからない**: `locate boost` でBoostの場所を確認
- **権限エラー**: `sudo`を使用するか`--user`フラグでPythonパッケージをインストール
- **古いコンパイラ**: GCC 7.0以上が必要。`gcc --version`で確認

## 🍎 macOS

### 必要なソフトウェア
- Xcode Command Line Tools
- [Homebrew](https://brew.sh/) (推奨)

### Homebrewでの手順

1. **Xcode Command Line Toolsのインストール**
   ```bash
   xcode-select --install
   ```

2. **Homebrewのインストール (未インストールの場合)**
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **依存関係のインストール**
   ```bash
   brew install boost python3 git
   ```

4. **Pythonパッケージのインストール**
   ```bash
   pip3 install xgboost scikit-learn pandas numpy
   ```

5. **ビルドとテスト**
   ```bash
   python3 build.py info   # 環境確認
   python3 build.py        # ビルド
   cd XGBoost
   python3 main.py help    # テスト
   ```

### Apple Silicon (M1/M2) Mac特有の注意点

- Homebrewのパスが`/opt/homebrew`になります
- Rosetta 2は不要です（ネイティブサポート）
- Intel Macとは異なるライブラリパスが使用されます

### トラブルシューティング (macOS)

- **brew command not found**: Homebrewのパスを確認 `echo $PATH`
- **ライブラリが見つからない**: `brew --prefix boost`でBoostの場所を確認
- **権限エラー**: `sudo`を使わずにHomebrewでインストール

## 🔧 共通トラブルシューティング

### ビルドエラー

1. **依存関係の確認**
   ```bash
   python install.py check
   ```

2. **環境情報の表示**
   ```bash
   python build.py info
   ```

3. **クリーンビルド**
   ```bash
   python build.py clean
   python build.py
   ```

### Pythonモジュールエラー

1. **仮想環境の使用**
   ```bash
   python -m venv fraud_attack_env
   source fraud_attack_env/bin/activate  # Linux/macOS
   # または
   fraud_attack_env\Scripts\activate     # Windows
   
   pip install xgboost scikit-learn pandas numpy
   ```

2. **パッケージの再インストール**
   ```bash
   pip uninstall xgboost scikit-learn pandas numpy
   pip install xgboost scikit-learn pandas numpy
   ```

### パフォーマンス最適化

1. **並列コンパイル**
   ```bash
   # Linuxの場合
   make -j$(nproc) -f Makefile.windows
   
   # macOSの場合  
   make -j$(sysctl -n hw.ncpu) -f Makefile.windows
   ```

2. **リリースビルド**
   ```bash
   python build.py       # 最適化済みビルド (デフォルト)
   python build.py debug # デバッグ用ビルド
   ```

## 📝 設定ファイルの説明

- `Makefile.windows` - クロスプラットフォーム対応Makefile
- `build.py` - 自動ビルドスクリプト
- `install.py` - 依存関係インストールスクリプト
- `main.py` - メインエントリポイント

## ⚡ 高速セットアップ (上級者向け)

全プラットフォームで以下のワンライナーが使用できます：

```bash
# 完全自動セットアップ
python install.py && python build.py && cd XGBoost && python main.py pipeline
```

このコマンドで依存関係のインストール、ビルド、実行まで自動実行されます。