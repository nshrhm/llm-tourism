#!/bin/bash

################################################################################
# LLM地域観光紹介性能比較実験 - 実行スクリプト
#
# このスクリプトは以下を自動実行します：
# 1. 環境チェック
# 2. 依存関係インストール
# 3. 実験実行
# 4. データ変換
# 5. 完了通知
#
# 使用方法:
#   bash scripts/run_experiment.sh
#
################################################################################

set -e  # エラー時に終了

# カラー出力設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ログ関数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ヘッダー表示
echo "================================================================================"
echo "  LLM地域観光紹介性能比較実験 - 自動実行スクリプト"
echo "================================================================================"
echo ""

# プロジェクトルートディレクトリに移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

log_info "プロジェクトルート: $PROJECT_ROOT"
echo ""

################################################################################
# 1. 環境チェック
################################################################################

log_info "【ステップ1】環境チェック"
echo "--------------------------------------------------------------------------------"

# Python バージョンチェック
if ! command -v python3 &> /dev/null; then
    log_error "Python3がインストールされていません"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
log_success "Python3 検出: $PYTHON_VERSION"

# Python 3.10以上かチェック
REQUIRED_VERSION="3.10"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    log_error "Python 3.10以上が必要です（現在: $PYTHON_VERSION）"
    exit 1
fi
log_success "Python バージョン要件を満たしています"

# .envファイルのチェック
if [ ! -f ".env" ]; then
    log_warning ".envファイルが見つかりません"
    log_info ".env.exampleをコピーして.envファイルを作成してください"
    log_info "  cp .env.example .env"
    log_info "その後、.envファイルを編集してAPIキーを設定してください"
    exit 1
fi
log_success ".envファイル検出"

# APIキー設定チェック
if grep -q "your_api_key_here" .env; then
    log_error "APIキーが設定されていません"
    log_info ".envファイルを編集してOPENROUTER_API_KEYを設定してください"
    exit 1
fi
log_success "APIキー設定確認"

echo ""

################################################################################
# 2. 仮想環境のチェックと依存関係インストール
################################################################################

log_info "【ステップ2】依存関係のチェック"
echo "--------------------------------------------------------------------------------"

# 仮想環境の確認
if [ -z "$VIRTUAL_ENV" ]; then
    log_warning "仮想環境が有効化されていません"

    # venvディレクトリの存在チェック
    if [ ! -d "venv" ]; then
        log_info "仮想環境を作成中..."
        python3 -m venv venv
        log_success "仮想環境を作成しました"
    fi

    log_info "仮想環境を有効化してください:"
    log_info "  source venv/bin/activate  # Linux/Mac"
    log_info "  venv\\Scripts\\activate     # Windows"
    exit 1
else
    log_success "仮想環境が有効化されています: $VIRTUAL_ENV"
fi

# 依存関係のインストール確認
log_info "依存関係をインストール中..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
log_success "依存関係のインストール完了"

echo ""

################################################################################
# 3. 設定ファイルのチェック
################################################################################

log_info "【ステップ3】設定ファイルのチェック"
echo "--------------------------------------------------------------------------------"

if [ ! -f "config/experiment_config.yaml" ]; then
    log_error "設定ファイルが見つかりません: config/experiment_config.yaml"
    exit 1
fi
log_success "設定ファイル検出"

echo ""

################################################################################
# 4. 実験実行の確認
################################################################################

log_info "【ステップ4】実験実行の確認"
echo "--------------------------------------------------------------------------------"

echo "以下の実験を実行します:"
echo "  - モデル数: 4"
echo "  - ペルソナ数: 4"
echo "  - 旅行タイプ数: 4"
echo "  - 総実験数: 64パターン"
echo ""
echo "注意: APIコールが発生します（費用が発生する可能性があります）"
echo ""

read -p "実験を開始しますか？ (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy](es)?$ ]]; then
    log_warning "実験をキャンセルしました"
    exit 0
fi

################################################################################
# 5. 実験実行
################################################################################

log_info "【ステップ5】実験実行"
echo "--------------------------------------------------------------------------------"

START_TIME=$(date +%s)

# 実験実行
log_info "実験を開始します..."
python3 -m src.experiment_runner

EXPERIMENT_STATUS=$?

if [ $EXPERIMENT_STATUS -ne 0 ]; then
    log_error "実験実行中にエラーが発生しました"
    exit 1
fi

log_success "実験実行完了"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

log_info "実験実行時間: ${DURATION_MIN}分${DURATION_SEC}秒"

echo ""

################################################################################
# 6. データ変換
################################################################################

log_info "【ステップ6】データ変換"
echo "--------------------------------------------------------------------------------"

log_info "JSON→CSV変換を開始します..."
python3 -m src.data_converter

CONVERSION_STATUS=$?

if [ $CONVERSION_STATUS -ne 0 ]; then
    log_error "データ変換中にエラーが発生しました"
    exit 1
fi

log_success "データ変換完了"

echo ""

################################################################################
# 7. 結果の確認
################################################################################

log_info "【ステップ7】結果の確認"
echo "--------------------------------------------------------------------------------"

# 生データのカウント
RAW_COUNT=$(find data/raw -name "*.json" 2>/dev/null | wc -l)
log_info "生データ（JSON）: ${RAW_COUNT}ファイル"

# 処理済みデータの確認
if [ -f "data/processed/experiment_results.csv" ]; then
    CSV_LINES=$(wc -l < data/processed/experiment_results.csv)
    CSV_RECORDS=$((CSV_LINES - 1))  # ヘッダー行を除く
    log_info "処理済みデータ（CSV）: ${CSV_RECORDS}レコード"
    log_success "CSVファイル: data/processed/experiment_results.csv"
else
    log_warning "CSVファイルが見つかりません"
fi

# 統計サマリーの確認
if [ -f "data/processed/statistics_summary.txt" ]; then
    log_success "統計サマリー: data/processed/statistics_summary.txt"
fi

echo ""

################################################################################
# 8. 完了通知
################################################################################

echo "================================================================================"
echo "  実験完了"
echo "================================================================================"
echo ""
log_success "すべての処理が正常に完了しました"
echo ""
echo "次のステップ:"
echo "  1. データの確認"
echo "     - 生データ: data/raw/"
echo "     - CSV: data/processed/experiment_results.csv"
echo "     - 統計: data/processed/statistics_summary.txt"
echo ""
echo "  2. データ分析"
echo "     - Excelで data/processed/experiment_results.csv を開く"
echo "     - Python/Rでさらなる分析を実行"
echo ""
echo "  3. 結果の可視化と論文執筆"
echo "     - docs/results.md を参考に結果をまとめる"
echo ""
echo "================================================================================"
