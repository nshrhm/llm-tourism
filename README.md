# LLM地域観光紹介性能比較実験システム

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenRouter](https://img.shields.io/badge/API-OpenRouter-green.svg)](https://openrouter.ai/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

> **開かれた学術研究**: 本プロジェクトは、LLMの観光情報生成能力を体系的に評価する、再現可能な実験システムです。すべてのコード、データ、ドキュメントをCC BY 4.0ライセンスで公開しています。

## 📋 目次

- [概要](#概要)
- [クイックスタート](#クイックスタート)
- [検証済み動作環境](#検証済み動作環境)
- [実験デザイン](#実験デザイン)
- [セットアップ手順](#セットアップ手順)
- [実行方法](#実行方法)
- [データ構造](#データ構造)
- [サンプル出力](#サンプル出力)
- [データ分析](#データ分析)
- [トラブルシューティング](#トラブルシューティング)
- [貢献方法](#貢献方法)
- [ライセンス](#ライセンス)

## 概要

本研究は、大規模言語モデル（Large Language Model, LLM）の地域観光紹介における性能を体系的に比較評価するための実験システムです。異なるLLM、ペルソナ、旅行タイプの組み合わせによる**64パターン**の実験を通じて、各モデルの特性を定量的・定性的に分析します。

### 研究目的

1. **性能比較**: 異なるLLMの観光情報生成能力の定量的評価
2. **ペルソナ影響**: ペルソナ設定が出力内容に与える影響の分析
3. **ターゲット適応**: ターゲット層（旅行タイプ）による応答特性の評価
4. **再現性**: 再現可能な実験環境の構築と共有

### 主な特徴

- ✅ **完全自動化**: 64パターンの実験を自動実行
- ✅ **堅牢な設計**: エラーハンドリング、リトライ、中断・再開機能
- ✅ **データ分析支援**: JSON/CSV出力、テキスト統計、類似度分析、感情分析
- ✅ **学術標準準拠**: ISO 8601形式、CC BY 4.0ライセンス、完全なドキュメント
- ✅ **検証済み**: 単体テスト、統合テスト、実環境での動作確認済み

## 🚀 クイックスタート

```bash
# 1. リポジトリのクローン
git clone https://github.com/nshrhm/llm-tourism.git
cd llm-tourism

# 2. 仮想環境作成と依存関係インストール
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# 3. 環境変数設定
cp .env.example .env
# .envファイルを編集してOPENROUTER_API_KEYを設定

# 4. 検証テスト（2パターンのみ）
python3 scripts/auto_test.py

# 5. 本実験実行（64パターン）
bash scripts/run_experiment.sh
```

## ✅ 検証済み動作環境

本システムは以下の環境で動作確認済みです：

### システム環境
- **OS**: Linux (WSL2), macOS, Windows
- **Python**: 3.10, 3.11, 3.12 ✓
- **メモリ**: 4GB以上推奨
- **ストレージ**: 500MB以上

### 検証結果（2025-11-21実施）

| 項目 | 結果 | 詳細 |
|------|------|------|
| 単体テスト | ✅ 11/11合格 | APIクライアント、ヘルパー関数 |
| 統合テスト | ✅ 成功 | 4パターンの実験実行 |
| API接続 | ✅ 成功 | OpenRouter認証・レスポンス確認 |
| データ生成 | ✅ 成功 | JSON 4件、CSV 1件生成 |
| 成功率 | ✅ 100% | 4/4パターン成功 |
| 平均応答時間 | 6.5秒 | モデル: GPT-5.1 |
| 平均トークン数 | 592トークン | 範囲: 523-660 |
| 平均文字数 | 635文字 | 範囲: 563-709 |

## 実験デザイン

### 実験パラメータ（4×4×4=64パターン）

#### 1. LLM（4種類）
- `openai/gpt-5.1-chat` - OpenAI GPT-5.1
- `anthropic/claude-sonnet-4.5` - Anthropic Claude Sonnet 4.5
- `google/gemini-2.5-flash` - Google Gemini 2.5 Flash
- `x-ai/grok-4-fast` - xAI Grok-4 Fast

#### 2. ペルソナ（4種類）
- **旅行代理店スタッフ**: プロフェッショナルな旅行提案
- **旅行系YouTuber**: エンターテイメント性の高い紹介
- **地方自治体職員**: 地域活性化の視点
- **SNSインフルエンサー**: トレンドを意識した紹介

#### 3. 旅行タイプ（4種類）
- **家族旅行**: 子供連れの家族向け
- **カップル旅行**: ロマンチックな雰囲気重視
- **外国人観光客**: 日本初訪問者向け
- **シニア旅行**: 60歳以上のシニア層向け

### プロンプトテンプレート

```
あなたは{ペルソナ}です。下関の観光名所を紹介してください。
紹介相手は、{旅行タイプ}を計画しています。1000文字程度で要約してください。
```

### 対象地域

**山口県下関市**を研究対象地域として選定

**選定理由**:
- 歴史的観光地と自然景観の両方を有する
- 多様な観光資源（関門海峡、温泉、歴史遺産、ふぐグルメ）
- 国内外の観光客層の多様性
- 中規模都市として汎用性の高い知見が得られる

## 技術仕様

### システム要件
- **Python**: 3.10以上
- **API**: OpenRouter経由でLLMにアクセス
- **メモリ**: 4GB以上推奨
- **ストレージ**: 実験データ用に500MB以上

### 主要依存ライブラリ
- `requests`: API通信
- `pandas`: データ処理と分析
- `pyyaml`: 設定ファイル管理
- `python-dotenv`: 環境変数管理
- `tqdm`: プログレスバー表示
- `pytest`: 単体テスト

### テキスト分析ライブラリ（オプション）
- `sentence-transformers`: テキスト類似度分析
- `transformers`: 感情分析（日本語BERT）
- `scikit-learn`: 統計分析・クラスタリング

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/nshrhm/llm-tourism.git
cd llm-tourism
```

### 2. 仮想環境の作成と有効化

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows
```

### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 4. 環境変数の設定

```bash
cp .env.example .env
# .envファイルを編集してOpenRouter APIキーを設定
```

**OpenRouter APIキーの取得方法**：
1. [OpenRouter](https://openrouter.ai/)にアクセス
2. アカウント作成・ログイン
3. [API Keys](https://openrouter.ai/keys)ページでキーを生成
4. `.env`ファイルの`OPENROUTER_API_KEY`に設定

```bash
# .envファイルの例
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxx
```

### 5. 設定ファイルの確認

[`config/experiment_config.yaml`](config/experiment_config.yaml) で実験パラメータを確認・調整できます。

## 実行方法

### オプション1: 自動実行スクリプト（推奨）

```bash
bash scripts/run_experiment.sh
```

このスクリプトは以下を自動実行します：
1. 環境チェック
2. 実験実行（64パターン）
3. データ変換（JSON→CSV）
4. 統計サマリー生成
5. 完了レポート表示

### オプション2: Pythonモジュールとして実行

```bash
# 実験の実行
python -m src.experiment_runner

# データの変換（JSON→CSV）
python -m src.data_converter
```

### オプション3: 検証テスト（小規模実験）

```bash
# 2パターンのみで動作確認
python3 scripts/auto_test.py
```

### 個別コンポーネントのテスト

```python
# プロンプト生成のテスト
from src.prompt_generator import PromptGenerator
generator = PromptGenerator()
prompts = generator.generate_all_prompts()
print(f"生成されたプロンプト数: {len(prompts)}")

# APIクライアントのテスト
from src.api_client import OpenRouterClient
client = OpenRouterClient()
response = client.generate(
    model="openai/gpt-5.1-chat",
    prompt="下関の観光名所を紹介してください",
    max_tokens=100
)
print(response['response'])
```

## データ構造

### 生データ（JSON形式）

保存先: [`data/raw/`](data/raw/)

```json
{
  "session_id": "session_20250121_143022",
  "experiment_id": "exp_000",
  "timestamp": "2025-11-21T14:30:22.123456+09:00",
  "model": {
    "name": "openai/gpt-5.1-chat",
    "display_name": "OpenAI GPT-5.1"
  },
  "persona": {
    "id": "travel_agent",
    "name": "旅行代理店スタッフ"
  },
  "travel_type": {
    "id": "family",
    "name": "家族旅行"
  },
  "prompt": "あなたは旅行代理店スタッフです...",
  "response": "下関は山口県の西端に位置し...",
  "performance": {
    "tokens_used": 660,
    "latency_ms": 6792
  },
  "metadata": { ... },
  "api_parameters": { ... }
}
```

### 処理済みデータ（CSV形式）

保存先: [`data/processed/experiment_results.csv`](data/processed/)

**カラム構成**:

| カラム名 | データ型 | 説明 |
|----------|----------|------|
| `session_id` | string | セッション識別子 |
| `experiment_id` | string | 実験識別子 |
| `timestamp` | datetime | 実行日時（ISO 8601） |
| `model` | string | モデル名 |
| `model_display_name` | string | モデル表示名 |
| `persona_id` | string | ペルソナID |
| `persona_name` | string | ペルソナ名 |
| `travel_type_id` | string | 旅行タイプID |
| `travel_type_name` | string | 旅行タイプ名 |
| `prompt` | text | 入力プロンプト |
| `response` | text | 生成テキスト |
| `tokens_used` | integer | 使用トークン数 |
| `latency_ms` | integer | 応答時間（ミリ秒） |
| `response_char_count` | integer | 文字数 |
| `response_word_count` | integer | 単語数（推定） |
| `response_line_count` | integer | 行数 |
| `prompt_char_count` | integer | プロンプト文字数 |

## 📊 サンプル出力

### 検証テストの実行例

```
================================================================================
  LLM地域観光紹介性能比較実験 - 自動検証テスト
================================================================================

【ステップ1】プロンプト生成器の初期化
✓ 64個のプロンプトを生成
✓ テスト用に4パターンを選択

【ステップ2】APIクライアントの初期化
✓ APIクライアント初期化成功

【ステップ3】実験実行（自動モード）
実験実行中: 100%|██████████| 4/4 [00:32<00:00,  8.06s/実験, 成功=4, 失敗=0]

【実験結果】
  - 成功: 4件
  - 失敗: 0件
  - 実行時間: 32.3秒

【ステップ4】データ変換
✓ データ変換完了
  - レコード数: 4
  - 出力ファイル: data/processed/experiment_results.csv

✓ すべてのコンポーネントが正常に動作しています！
```

### 生成された観光案内の例

**条件**: 旅行代理店スタッフ × 外国人観光客（日本初訪問）

```
山口県の最西端に位置する下関は、日本の歴史と海の魅力が凝縮された港町で、
初めて日本を訪れる外国人旅行者にも訪れやすい場所です。市の象徴ともいえる
関門海峡は、本州と九州を隔てる重要な水路で、潮の流れが速く迫力があります。
海峡沿いには散策路が整備され、行き交う船や対岸の門司港レトロ地区を眺めな
がら散歩が楽しめます。徒歩で海峡を渡れる関門トンネル人道もユニークな体験
として人気です...

（文字数: 601、トークン: 545、応答時間: 5.6秒）
```

## ディレクトリ構造

```
llm-tourism/
├── README.md                    # 本ファイル
├── LICENSE                      # CC BY 4.0ライセンス
├── CITATION.cff                 # 引用情報（BibTeX等）
├── CONTRIBUTING.md              # 貢献ガイドライン
├── CHANGELOG.md                 # 変更履歴
├── .gitignore                   # Git除外設定
├── .env.example                 # 環境変数テンプレート
├── requirements.txt             # Python依存関係
├── pytest.ini                   # テスト設定
├── config/
│   └── experiment_config.yaml   # 実験設定
├── src/
│   ├── __init__.py
│   ├── api_client.py            # OpenRouter APIクライアント
│   ├── prompt_generator.py      # プロンプト生成器
│   ├── experiment_runner.py     # 実験実行エンジン
│   └── data_converter.py        # データ変換ツール
├── data/
│   ├── raw/                     # JSON生データ
│   ├── processed/               # CSV処理済みデータ
│   └── archive/                 # アーカイブデータ
├── docs/
│   ├── methodology.md           # 実験方法論（学術論文スタイル）
│   ├── results.md               # 結果報告テンプレート
│   ├── manual.md                # データ分析マニュアル
│   └── diagrams/
│       ├── system_flow.mermaid  # システムフロー図
│       └── architecture.drawio  # アーキテクチャ図
├── scripts/
│   ├── run_experiment.sh        # 自動実行スクリプト
│   ├── auto_test.py             # 検証テストスクリプト
│   └── debug_api.py             # API接続デバッグ
└── tests/
    └── test_api_client.py       # 単体テスト
```

## データ分析

### 基本的な分析

```python
import pandas as pd

# データ読み込み
df = pd.read_csv('data/processed/experiment_results.csv')

# モデル別の基礎統計
model_stats = df.groupby('model').agg({
    'response_char_count': ['mean', 'std', 'min', 'max'],
    'tokens_used': ['mean', 'std'],
    'latency_ms': ['mean', 'std']
}).round(2)

print(model_stats)
```

### テキスト類似度分析

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# モデルのロード
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# 埋め込みベクトル生成
embeddings = model.encode(df['response'].tolist())

# コサイン類似度計算
similarity_matrix = cosine_similarity(embeddings)
```

### 感情分析

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 日本語BERTモデルのロード
model_name = "daigo/bert-base-japanese-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 感情スコア計算
# 詳細はdocs/manual.mdを参照
```

詳細な分析手法は[`docs/manual.md`](docs/manual.md)を参照してください。

## 倫理的配慮

- ✅ APIキーは絶対にコミットしない（`.gitignore`で保護）
- ✅ 実験データの取り扱いは研究倫理規定に準拠
- ✅ 生成されたテキストの著作権はLLM提供者の利用規約に従う
- ✅ 個人情報を含むプロンプトの使用禁止
- ✅ すべてのデータとコードをCC BY 4.0で公開（オープンサイエンス）

## 再現性の確保

本研究では以下の情報を記録し、完全な再現性を確保しています：

1. **実験パラメータ**: YAML設定ファイルで明示的に管理
2. **実行日時**: ISO 8601形式、タイムゾーン付き
3. **モデルバージョン**: APIレスポンスから取得
4. **システム環境**: Python version、ライブラリバージョン（`requirements.txt`）
5. **乱数シード**: 該当する場合は固定値を使用

## トラブルシューティング

### APIキーエラー

```
Error: OpenRouter API key not found
```

**解決方法**:
1. `.env`ファイルが存在するか確認
2. `OPENROUTER_API_KEY`が正しく設定されているか確認
3. APIキーが`sk-or-v1-`で始まっているか確認

### API認証エラー（401 Unauthorized）

```
Error: 401 Client Error: Unauthorized
```

**解決方法**:
1. APIキーが有効か確認（[OpenRouter Keys](https://openrouter.ai/keys)）
2. アカウントにクレジットがあるか確認
3. 新しいAPIキーを生成して`.env`を更新

### レート制限エラー（429 Too Many Requests）

**解決方法**:
`config/experiment_config.yaml`で`request_delay`を増やしてください：

```yaml
api:
  request_delay: 5  # 2秒 → 5秒に変更
```

### データ保存エラー

**解決方法**:
1. `data/raw/`と`data/processed/`ディレクトリの書き込み権限を確認
2. ディスク容量を確認（500MB以上推奨）

### 依存関係のインストールエラー

**解決方法**:
```bash
# pipをアップグレード
pip install --upgrade pip

# 個別にインストール
pip install requests pandas pyyaml python-dotenv tqdm

# PyTorchのインストール（感情分析を使用する場合）
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 詳細なデバッグ

```bash
# API接続テスト
python3 scripts/debug_api.py

# 単体テスト実行
pytest tests/ -v
```

## 貢献方法

本研究への貢献を歓迎します！詳細は[CONTRIBUTING.md](CONTRIBUTING.md)を参照してください。

### 貢献の例

1. **Issuesでバグ報告・機能提案**
2. **Pull Requestで改善提案**
3. **実験結果の共有**
4. **ドキュメントの改善**
5. **新しい分析手法の追加**

## 引用方法

本研究を引用する場合は、以下の形式を推奨します：

```bibtex
@software{llm_tourism_2025,
  author = {白濵 成希},
  title = {LLM地域観光紹介性能比較実験システム},
  year = {2025},
  url = {https://github.com/nshrhm/llm-tourism},
  license = {CC-BY-4.0}
}
```

詳細は[CITATION.cff](CITATION.cff)を参照してください。

## ライセンス

本プロジェクトは **Creative Commons Attribution 4.0 International (CC BY 4.0)** ライセンスの下で公開されています。

- ✅ 実験システム（コード）: CC BY 4.0
- ✅ 実験データ: CC BY 4.0
- ✅ ドキュメント: CC BY 4.0

詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 連絡先

研究に関する問い合わせ：
- **GitHub Issues**: [リポジトリのIssues](https://github.com/nshrhm/llm-tourism/issues)
- **Email**: [nshirahama＠ieee.org]

## 参考文献

- OpenRouter API Documentation: https://openrouter.ai/docs
- 観光情報学研究: [`docs/methodology.md`](docs/methodology.md)に記載
- テキストマイニング手法: [`docs/manual.md`](docs/manual.md)に記載

## 更新履歴

詳細は[CHANGELOG.md](CHANGELOG.md)を参照してください。

- **2025-01-21**: プロジェクト初版リリース
  - Phase 1-5完了: 基盤構築、コアシステム、データ処理、ドキュメント、品質保証
  - 検証テスト実施（4パターン、成功率100%）
  - 単体テスト11件合格

---

**オープンサイエンス**: すべてのコード、データ、ドキュメントをCC BY 4.0ライセンスで公開し、学術コミュニティへの貢献を目指しています。
