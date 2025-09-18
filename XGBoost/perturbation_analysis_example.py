#!/usr/bin/env python3
"""
摂動分析の使用例

このスクリプトは、敵対的サンプルの摂動分析を実行する方法を示します。
"""

from fraud_attack_utils import FraudAttackSystem

def main():
    print("=== 敵対的サンプル摂動分析の実行例 ===\n")
    
    # システムの初期化
    system = FraudAttackSystem(data_path='Base.csv')
    
    # 1. 摂動分析の実行
    print("1. 摂動分析を実行...")
    analysis_results = system.analyze_perturbations(
        adversarial_csv='adversarial_sample.csv'  # 攻撃後に生成されるファイル
    )
    
    # 2. 可視化の生成
    print("\n2. 可視化を生成...")
    system.visualize_perturbations(analysis_results, top_n_features=15)
    
    # 3. 詳細レポートの生成
    print("\n3. 詳細レポートを生成...")
    report = system.generate_perturbation_report(analysis_results)
    
    print("\n=== 分析完了 ===")
    print("以下のファイルが生成されました:")
    print("- perturbation_analysis.csv: 詳細な摂動データ")
    print("- perturbation_stats.json: 統計情報")
    print("- perturbation_report.txt: 分析レポート")
    print("- perturbation_norms_distribution.png: ノルム分布")
    print("- feature_perturbation_analysis.png: 特徴量別分析")
    print("- perturbation_heatmap.png: 摂動ヒートマップ")
    
    # 4. 簡単な結果サマリー
    print("\n=== 結果サマリー ===")
    stats = analysis_results['perturbation_stats']
    print(f"平均L2摂動: {stats['l2_norm']['mean']:.6f}")
    print(f"最小L2摂動: {stats['l2_norm']['min']:.6f}")
    print(f"最大L2摂動: {stats['l2_norm']['max']:.6f}")
    
    # 最も変更された特徴量
    feature_stats = stats['feature_perturbations']
    sorted_features = sorted(feature_stats.items(), 
                           key=lambda x: x[1]['change_rate'], reverse=True)
    
    print(f"\n最も変更された特徴量:")
    for i, (feature, feature_stat) in enumerate(sorted_features[:5], 1):
        print(f"  {i}. {feature}: 変更率{feature_stat['change_rate']:.1%}")

if __name__ == "__main__":
    # 注意: 実際に実行する前に以下の条件を満たしてください：
    # 1. Base.csv ファイルが存在する
    # 2. 攻撃が実行済みで adversarial_sample.csv が存在する
    # 3. feature_info.json が存在する
    
    try:
        main()
    except FileNotFoundError as e:
        print(f"ファイルが見つかりません: {e}")
        print("\n使用前の準備:")
        print("1. Base.csvを配置")
        print("2. FraudAttackSystem.run_full_pipeline()でモデル訓練")
        print("3. LT-Attack攻撃の実行")
        print("4. convert_and_verify_results()で結果変換")
        print("5. この摂動分析スクリプトの実行")
    except Exception as e:
        print(f"エラーが発生しました: {e}")