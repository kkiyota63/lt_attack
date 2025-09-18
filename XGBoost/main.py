#!/usr/bin/env python3
"""
詐欺検知攻撃システム - メインエントリポイント

使用方法:
    python main.py train                    # モデル訓練
    python main.py attack                   # 攻撃実行
    python main.py convert                  # 結果変換
    python main.py verify                   # 攻撃検証
    python main.py pipeline                 # 完全パイプライン
    python main.py help                     # ヘルプ表示
"""

import sys
import os
import subprocess
from fraud_attack_utils import FraudAttackSystem

def print_help():
    """ヘルプメッセージの表示"""
    print(__doc__)
    print("\nコマンド詳細:")
    print("  train     - XGBoostモデルの訓練とファイル生成")
    print("  attack    - 敵対的攻撃の実行 (C++バイナリを呼び出し)")
    print("  convert   - LIBSVM結果をCSVに変換")
    print("  verify    - 攻撃成功率の検証")
    print("  pipeline  - 訓練から検証までの完全実行")
    print("  help      - このヘルプメッセージを表示")

def run_attack():
    """敵対的攻撃の実行"""
    import platform
    print("=== 敵対的攻撃の実行 ===")
    
    # Cross-platform binary detection
    if platform.system() == "Windows":
        attack_binary = os.path.join("..", "lt_attack.exe")
        shell_cmd = ["lt_attack.exe", "configs/base_fraud_to_benign_attack.json"]
    else:
        attack_binary = os.path.join("..", "lt_attack")
        shell_cmd = ["./lt_attack", "configs/base_fraud_to_benign_attack.json"]
    
    config_file = os.path.join("..", "configs", "base_fraud_to_benign_attack.json")
    
    if not os.path.exists(attack_binary):
        print("エラー: lt_attack バイナリが見つかりません")
        if platform.system() == "Windows":
            print("まず 'make -f Makefile.windows' コマンドでコンパイルしてください")
        else:
            print("まず 'make' コマンドでコンパイルしてください")
        return False
    
    if not os.path.exists(config_file):
        print("エラー: 設定ファイルが見つかりません")
        print(f"ファイル: {config_file}")
        return False
    
    print(f"実行コマンド: {' '.join(shell_cmd)}")
    
    try:
        # Change to parent directory and run attack
        original_dir = os.getcwd()
        parent_dir = os.path.dirname(original_dir)
        os.chdir(parent_dir)
        
        result = subprocess.run(
            shell_cmd,
            capture_output=True,
            text=True,
            shell=(platform.system() == "Windows")
        )
        
        os.chdir(original_dir)
        
        if result.returncode == 0:
            print("攻撃が正常に完了しました")
            print(result.stdout)
            return True
        else:
            print(f"攻撃が失敗しました (終了コード: {result.returncode})")
            print(f"エラー: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"攻撃実行中にエラーが発生しました: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print_help()
        return
    
    command = sys.argv[1].lower()
    system = FraudAttackSystem()
    
    try:
        if command == "help":
            print_help()
            
        elif command == "train":
            print("=== モデル訓練 ===")
            results = system.train_model()
            print(f"\n訓練完了 - 精度: {results['accuracy']:.4f}")
            
        elif command == "attack":
            success = run_attack()
            if success:
                print("\n次は結果を変換してください:")
                print("python main.py convert")
            
        elif command == "convert":
            print("=== 結果変換 ===")
            if not os.path.exists('adversarial_examples.libsvm'):
                print("エラー: adversarial_examples.libsvm が見つかりません")
                print("まず攻撃を実行してください: python main.py attack")
                return
            
            results = system.convert_and_verify_results()
            print(f"\n変換・検証完了 - 成功率: {results['success_rate']:.2f}%")
            
        elif command == "verify":
            print("=== 攻撃検証 ===")
            if not os.path.exists('adversarial_fixed.csv'):
                print("エラー: adversarial_fixed.csv が見つかりません")
                print("まず変換を実行してください: python main.py convert")
                return
            
            results = system.verify_attack_success('adversarial_fixed.csv')
            print(f"\n検証完了 - 成功率: {results['success_rate']:.2f}%")
            
        elif command == "pipeline":
            print("=== 完全パイプライン ===")
            
            # Step 1: Train
            print("\n[1/4] モデル訓練中...")
            train_results = system.train_model()
            print(f"✓ 訓練完了 - 精度: {train_results['accuracy']:.4f}")
            
            # Step 2: Attack
            print("\n[2/4] 敵対的攻撃実行中...")
            attack_success = run_attack()
            if not attack_success:
                print("✗ 攻撃に失敗しました")
                return
            print("✓ 攻撃完了")
            
            # Step 3: Convert and verify
            print("\n[3/4] 結果変換・検証中...")
            verify_results = system.convert_and_verify_results()
            print(f"✓ 検証完了 - 成功率: {verify_results['success_rate']:.2f}%")
            
            print("\n[4/4] パイプライン完了!")
            print(f"最終結果:")
            print(f"  - モデル精度: {train_results['accuracy']:.4f}")
            print(f"  - 攻撃成功率: {verify_results['success_rate']:.2f}%")
            print(f"  - 攻撃サンプル数: {verify_results['successful_attacks']:,}/{verify_results['total_samples']:,}")
            
        else:
            print(f"未知のコマンド: {command}")
            print_help()
            
    except KeyboardInterrupt:
        print("\n処理が中断されました")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()