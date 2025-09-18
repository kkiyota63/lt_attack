#!/usr/bin/env python3
"""
クロスプラットフォーム ビルドスクリプト

Windows、Linux、macOSで自動的に適切なコンパイル設定を使用します。

使用方法:
    python build.py          # 通常ビルド
    python build.py debug    # デバッグビルド
    python build.py clean    # クリーンアップ
    python build.py info     # ビルド情報表示
"""

import subprocess
import sys
import platform
import os
import shutil

def detect_platform():
    """プラットフォームとビルド設定の検出"""
    system = platform.system()
    
    if system == "Windows":
        return {
            "os": "Windows",
            "makefile": "Makefile.windows",
            "exe_suffix": ".exe",
            "make_cmd": "mingw32-make" if shutil.which("mingw32-make") else "make"
        }
    elif system == "Darwin":
        return {
            "os": "macOS",
            "makefile": "Makefile",
            "exe_suffix": "",
            "make_cmd": "make"
        }
    else:  # Linux and others
        return {
            "os": "Linux",
            "makefile": "Makefile",
            "exe_suffix": "",
            "make_cmd": "make"
        }

def check_dependencies(config):
    """依存関係のチェック"""
    print(f"=== {config['os']} 環境でのビルド ===")
    
    # Make/Compiler check
    if not shutil.which(config["make_cmd"]):
        print(f"エラー: {config['make_cmd']} が見つかりません")
        if config["os"] == "Windows":
            print("MSYS2/MinGW-w64 環境をインストールしてください")
            print("https://www.msys2.org/")
        return False
    
    # Compiler check
    if not shutil.which("g++"):
        print("エラー: g++ コンパイラが見つかりません")
        if config["os"] == "Windows":
            print("MSYS2で以下をインストール: pacman -S mingw-w64-x86_64-gcc")
        elif config["os"] == "macOS":
            print("Xcode Command Line Tools をインストール: xcode-select --install")
        else:
            print("GCC をインストール: sudo apt-get install build-essential")
        return False
    
    print(f"✓ コンパイラ: {shutil.which('g++')}")
    print(f"✓ Make: {shutil.which(config['make_cmd'])}")
    
    return True

def run_build(target="all"):
    """ビルドの実行"""
    config = detect_platform()
    
    if not check_dependencies(config):
        return False
    
    print(f"\n=== ビルド開始: {target} ===")
    
    # Makefile selection
    makefile_args = ["-f", config["makefile"]] if config["makefile"] != "Makefile" else []
    
    try:
        cmd = [config["make_cmd"]] + makefile_args + [target]
        print(f"実行コマンド: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True)
        
        print(f"✓ ビルド成功: {target}")
        
        # Show built executables
        if target in ["all", "lt_attack"]:
            exe_name = f"lt_attack{config['exe_suffix']}"
            if os.path.exists(exe_name):
                print(f"✓ 実行ファイル生成: {exe_name}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ ビルドエラー: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ {config['make_cmd']} が見つかりません")
        return False

def show_info():
    """ビルド情報の表示"""
    config = detect_platform()
    
    print("=== ビルド情報 ===")
    print(f"プラットフォーム: {config['os']}")
    print(f"アーキテクチャ: {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"Make: {config['make_cmd']}")
    print(f"Makefile: {config['makefile']}")
    print(f"実行ファイル拡張子: {config['exe_suffix'] or '(なし)'}")
    
    # Check dependencies
    print("\n=== 依存関係チェック ===")
    
    tools = ["g++", "make", "mingw32-make", "python", "pip"]
    for tool in tools:
        path = shutil.which(tool)
        status = "✓" if path else "✗"
        print(f"{status} {tool}: {path or '見つかりません'}")
    
    # Boost check
    if config["os"] == "Windows":
        boost_paths = ["/mingw64/include/boost", "C:/msys64/mingw64/include/boost"]
    elif config["os"] == "Darwin":
        boost_paths = ["/opt/homebrew/include/boost", "/usr/local/include/boost"]
    else:
        boost_paths = ["/usr/include/boost", "/usr/local/include/boost"]
    
    boost_found = any(os.path.exists(path) for path in boost_paths)
    status = "✓" if boost_found else "✗"
    print(f"{status} Boost C++: {'見つかりました' if boost_found else '見つかりません'}")

def main():
    if len(sys.argv) < 2:
        target = "all"
    else:
        target = sys.argv[1]
    
    if target == "info":
        show_info()
    elif target == "help":
        print(__doc__)
    elif target in ["all", "debug", "test", "clean"]:
        success = run_build(target)
        if not success:
            sys.exit(1)
    else:
        print(f"未知のターゲット: {target}")
        print("利用可能: all, debug, test, clean, info, help")
        sys.exit(1)

if __name__ == "__main__":
    main()