#!/usr/bin/env python3
"""
クロスプラットフォーム インストールスクリプト

Windows、Ubuntu/Linux、macOSで依存関係を自動インストールします。

使用方法:
    python install.py          # 依存関係のインストール
    python install.py check    # インストール状況の確認
    python install.py help     # ヘルプ表示
"""

import subprocess
import sys
import platform
import shutil
import os

def run_command(cmd, description, check=True):
    """コマンドの実行"""
    print(f"実行中: {description}")
    print(f"コマンド: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        
        if result.stdout:
            print(f"出力: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"エラー: {e}")
        if e.stderr:
            print(f"エラー詳細: {e.stderr.strip()}")
        return False

def install_windows():
    """Windows (MSYS2) での依存関係インストール"""
    print("=== Windows (MSYS2) 環境での依存関係インストール ===")
    
    if not shutil.which("pacman"):
        print("エラー: MSYS2 が見つかりません")
        print("以下からMSYS2をインストールしてください:")
        print("https://www.msys2.org/")
        print("\nインストール後、MSYS2 MINGW64ターミナルで実行してください")
        return False
    
    packages = [
        "mingw-w64-x86_64-gcc",
        "mingw-w64-x86_64-boost", 
        "mingw-w64-x86_64-python",
        "mingw-w64-x86_64-python-pip",
        "make"
    ]
    
    print("MSYS2パッケージの更新...")
    if not run_command("pacman -Syu --noconfirm", "システム更新", check=False):
        print("警告: システム更新に失敗しましたが続行します")
    
    print("必要パッケージのインストール...")
    for package in packages:
        run_command(f"pacman -S --noconfirm {package}", f"{package} インストール", check=False)
    
    return install_python_packages()

def install_ubuntu():
    """Ubuntu/Debian での依存関係インストール"""
    print("=== Ubuntu/Linux 環境での依存関係インストール ===")
    
    packages = [
        "build-essential",
        "libboost-all-dev",
        "python3",
        "python3-pip",
        "python3-dev"
    ]
    
    print("パッケージリストの更新...")
    if not run_command("sudo apt-get update", "apt update"):
        print("エラー: パッケージリストの更新に失敗")
        return False
    
    print("必要パッケージのインストール...")
    cmd = ["sudo", "apt-get", "install", "-y"] + packages
    if not run_command(cmd, "依存パッケージインストール"):
        print("エラー: パッケージのインストールに失敗")
        return False
    
    return install_python_packages()

def install_macos():
    """macOS での依存関係インストール"""
    print("=== macOS 環境での依存関係インストール ===")
    
    if not shutil.which("brew"):
        print("エラー: Homebrew が見つかりません")
        print("以下のコマンドでHomebrewをインストールしてください:")
        print('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
        return False
    
    packages = ["boost", "python3"]
    
    print("Homebrewの更新...")
    run_command("brew update", "Homebrew update", check=False)
    
    print("必要パッケージのインストール...")
    for package in packages:
        run_command(f"brew install {package}", f"{package} インストール", check=False)
    
    return install_python_packages()

def install_python_packages():
    """Python パッケージのインストール"""
    print("\n=== Python パッケージのインストール ===")
    
    packages = [
        "xgboost>=1.6.0",
        "scikit-learn>=1.0.0", 
        "pandas>=1.3.0",
        "numpy>=1.20.0"
    ]
    
    # pip upgrade
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], "pip upgrade", check=False)
    
    for package in packages:
        if not run_command([sys.executable, "-m", "pip", "install", package], f"{package} インストール"):
            print(f"警告: {package} のインストールに失敗")
    
    return True

def check_installation():
    """インストール状況の確認"""
    print("=== インストール状況の確認 ===")
    
    system = platform.system()
    print(f"OS: {system} {platform.release()}")
    print(f"アーキテクチャ: {platform.machine()}")
    print(f"Python: {sys.version}")
    
    # Essential tools
    tools = {
        "g++": "C++ コンパイラ",
        "make": "Makeユーティリティ", 
        "python3": "Python 3",
        "pip": "Python パッケージマネージャ"
    }
    
    if system == "Windows":
        tools["pacman"] = "MSYS2 パッケージマネージャ"
    elif system == "Darwin":
        tools["brew"] = "Homebrew"
    
    print("\n=== 必須ツール ===")
    for tool, description in tools.items():
        path = shutil.which(tool)
        status = "✓" if path else "✗"
        print(f"{status} {description} ({tool}): {path or '見つかりません'}")
    
    # Boost libraries
    print("\n=== Boost C++ ライブラリ ===")
    boost_paths = []
    
    if system == "Windows":
        boost_paths = ["/mingw64/include/boost", "C:/msys64/mingw64/include/boost"]
    elif system == "Darwin":
        boost_paths = ["/opt/homebrew/include/boost", "/usr/local/include/boost"]
    else:  # Linux
        boost_paths = ["/usr/include/boost", "/usr/local/include/boost"]
    
    boost_found = False
    for path in boost_paths:
        if os.path.exists(path):
            print(f"✓ Boost ヘッダー: {path}")
            boost_found = True
            break
    
    if not boost_found:
        print("✗ Boost ヘッダーが見つかりません")
    
    # Python packages
    print("\n=== Python パッケージ ===")
    python_packages = ["xgboost", "sklearn", "pandas", "numpy"]
    
    for package in python_packages:
        try:
            __import__(package)
            print(f"✓ {package}: インストール済み")
        except ImportError:
            print(f"✗ {package}: 見つかりません")
    
    # Build test
    print("\n=== ビルドテスト ===")
    if shutil.which("g++") and boost_found:
        # Simple boost test
        test_code = '''
#include <boost/version.hpp>
#include <iostream>
int main() {
    std::cout << "Boost version: " << BOOST_VERSION << std::endl;
    return 0;
}
'''
        try:
            with open("boost_test.cpp", "w") as f:
                f.write(test_code)
            
            compile_cmd = ["g++", "-o", "boost_test", "boost_test.cpp"]
            if system == "Linux":
                compile_cmd.extend(["-lboost_system"])
            
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Boost コンパイルテスト: 成功")
                # Run test
                test_result = subprocess.run(["./boost_test"], capture_output=True, text=True)
                if test_result.returncode == 0:
                    print(f"✓ Boost 実行テスト: {test_result.stdout.strip()}")
            else:
                print(f"✗ Boost コンパイルテスト: 失敗")
                print(f"エラー: {result.stderr}")
            
            # Cleanup
            for file in ["boost_test.cpp", "boost_test", "boost_test.exe"]:
                if os.path.exists(file):
                    os.remove(file)
                    
        except Exception as e:
            print(f"✗ ビルドテスト中にエラー: {e}")
    else:
        print("✗ ビルドテスト: g++またはBoostが見つかりません")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_installation()
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == "help":
        print(__doc__)
        return
    
    system = platform.system()
    
    print("=== 詐欺検知攻撃システム 依存関係インストール ===")
    print(f"検出されたOS: {system}")
    
    if system == "Windows":
        success = install_windows()
    elif system == "Linux":
        success = install_ubuntu()
    elif system == "Darwin":
        success = install_macos()
    else:
        print(f"未対応のOS: {system}")
        success = False
    
    if success:
        print("\n✓ 依存関係のインストールが完了しました")
        print("\n次のステップ:")
        print("1. ビルド: python build.py")
        print("2. 実行: cd XGBoost && python main.py pipeline")
    else:
        print("\n✗ 依存関係のインストールに失敗しました")
        print("エラーを確認して手動でインストールしてください")
        sys.exit(1)

if __name__ == "__main__":
    main()