#!/bin/bash

# 詐欺検知攻撃システム ワンコマンド実行スクリプト
# Usage: ./run_attack.sh <data_file> [options]

set -e  # エラーで停止

# 色付きメッセージ関数
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# 使用方法の表示
show_usage() {
    echo "詐欺検知攻撃システム - ワンコマンド実行"
    echo ""
    echo "使用方法:"
    echo "  ./run_attack.sh <data_file> [options]"
    echo ""
    echo "オプション:"
    echo "  --no-normalize    特徴量の正規化を無効にする"
    echo "  --threads N       攻撃に使用するスレッド数 (default: 10)"
    echo "  --help           このヘルプを表示"
    echo ""
    echo "例:"
    echo "  ./run_attack.sh top100_cleaned_data.csv"
    echo "  ./run_attack.sh data.csv --no-normalize --threads 20"
}

# 引数チェック
if [ $# -eq 0 ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

DATA_FILE="$1"
shift

# データファイルの存在確認
if [ ! -f "$DATA_FILE" ]; then
    print_error "データファイルが見つかりません: $DATA_FILE"
    exit 1
fi

print_info "詐欺検知攻撃システムを開始します"
print_info "データファイル: $DATA_FILE"

# Python依存関係チェック
print_info "Python依存関係をチェック中..."
python3 -c "
import pandas, xgboost, sklearn, numpy
print('✓ 必要なライブラリが利用可能です')
" 2>/dev/null || {
    print_error "必要なPythonライブラリがインストールされていません"
    print_info "以下のコマンドでインストールしてください:"
    echo "  pip install pandas xgboost scikit-learn numpy matplotlib seaborn"
    exit 1
}

# lt_attackバイナリの確認
if [ ! -f "../lt_attack" ]; then
    print_error "lt_attackバイナリが見つかりません"
    print_info "親ディレクトリにlt_attackがあることを確認してください"
    exit 1
fi

# 詐欺ケースの存在確認
print_info "データ内の詐欺ケースをチェック中..."
FRAUD_COUNT=$(python3 -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$DATA_FILE')
    fraud_count = df['fraud_bool'].sum() if 'fraud_bool' in df.columns else 0
    print(fraud_count)
except Exception as e:
    print(0)
")

if [ "$FRAUD_COUNT" -eq 0 ]; then
    print_error "データに詐欺ケース（fraud_bool=1）が見つかりません"
    exit 1
fi

print_success "詐欺ケース $FRAUD_COUNT 件を確認"

# 実行コマンド構築
CMD="python3 fraud_attack_system.py $DATA_FILE"

# オプション処理
while [ $# -gt 0 ]; do
    case $1 in
        --no-normalize)
            CMD="$CMD --no-normalize"
            print_info "正規化を無効にします"
            ;;
        --threads)
            if [ -n "$2" ] && [ "$2" -eq "$2" ] 2>/dev/null; then
                CMD="$CMD --threads $2"
                print_info "スレッド数: $2"
                shift
            else
                print_error "--threads オプションには数値を指定してください"
                exit 1
            fi
            ;;
        *)
            print_warning "不明なオプション: $1"
            ;;
    esac
    shift
done

print_info "実行コマンド: $CMD"
echo ""

# メイン実行
START_TIME=$(date +%s)

print_info "攻撃システムを実行中..."
if eval "$CMD"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    print_success "攻撃システムが正常に完了しました"
    print_info "実行時間: ${DURATION}秒"
    
    echo ""
    print_info "生成されたファイル:"
    ls -la attack_*.csv attack_*.json attack_*.txt 2>/dev/null | while read line; do
        echo "  $line"
    done
    
    echo ""
    print_info "主要な結果:"
    if [ -f "attack_report.txt" ]; then
        # レポートから主要な統計を抽出
        grep -E "(平均L2摂動|摂動規模|最も脆弱な特徴量)" attack_report.txt | head -3
    fi
    
    echo ""
    print_success "詳細は attack_report.txt をご確認ください"
    
else
    print_error "攻撃システムの実行に失敗しました"
    print_info "トラブルシューティング:"
    echo "  1. データファイルの形式を確認してください"
    echo "  2. メモリ不足の場合は --threads を減らしてください"
    echo "  3. 詳細なエラーログを確認してください"
    exit 1
fi