# 🚀 クイックスタートガイド

## 最も簡単な使い方

```bash
cd XGBoost
./run_attack.sh top100_cleaned_data.csv
```

たったこれだけで完全な攻撃分析が実行されます！

## 実行される処理

1. **データ確認** - CSVファイルと詐欺ケースの存在をチェック
2. **モデル訓練** - XGBoostで詐欺検知モデルを訓練（正規化あり）
3. **攻撃実行** - LT-Attackで敵対的サンプルを生成
4. **結果分析** - 摂動を詳細分析
5. **レポート生成** - 総合分析レポートを出力

## 出力されるファイル

- `attack_report.txt` - **メインのレポート**（これを最初に確認）
- `attack_analysis.csv` - 詳細な比較データ
- `attack_stats.json` - 統計情報

## その他のオプション

```bash
# 正規化なしで実行（大きな摂動になる）
./run_attack.sh top100_cleaned_data.csv --no-normalize

# スレッド数を指定（高速化）
./run_attack.sh top100_cleaned_data.csv --threads 20

# Pythonから直接実行
python3 fraud_attack_system.py top100_cleaned_data.csv
```

## 結果の見方

レポートの「主要な発見」セクションをチェック：

```
• 平均L2摂動: 0.123456  ← 小さいほど危険（微小摂動で攻撃成功）
• 摂動規模: 微小摂動     ← 「微小摂動」なら脆弱性あり
• 最も脆弱な特徴量: income ← この特徴量が攻撃対象になりやすい
```

## トラブル時は

```bash
# ヘルプ表示
./run_attack.sh --help

# エラー時のデバッグ
python3 fraud_attack_system.py top100_cleaned_data.csv --threads 5
```

詳細は `README.md` をご覧ください。