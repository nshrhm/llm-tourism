# Changelog

このファイルは、プロジェクトの重要な変更をすべて記録します。

フォーマットは [Keep a Changelog](https://keepachangelog.com/ja/1.0.0/) に基づき、
バージョニングは [Semantic Versioning](https://semver.org/lang/ja/) に準拠します。

## [Unreleased]

### 今後の予定
- 実験結果の可視化ダッシュボード
- より詳細な統計分析機能
- 複数地域への対応
- 多言語プロンプトサポート

## [1.0.0] - 2025-01-21

### 概要
LLM地域観光紹介性能比較実験システムの初版リリース。
64パターン（4モデル × 4ペルソナ × 4旅行タイプ）の実験を自動実行し、
定量的・定性的分析を可能にする学術研究向けシステムです。

### Added - 新機能

#### Phase 1: プロジェクト基盤構築
- プロジェクトディレクトリ構造の作成
- `.gitignore` - 環境変数とデータファイルの保護
- `.env.example` - 環境変数テンプレート
- `requirements.txt` - 依存関係管理（テキスト分析ライブラリ含む）
- `README.md` - 学術研究向け包括的ドキュメント
- `LICENSE` - CC BY 4.0 ライセンス全文

#### Phase 2-3: コアシステム実装
- `config/experiment_config.yaml` - 実験設定（64パターン定義）
- `src/api_client.py` - OpenRouter API クライアント
  - 指数バックオフによるリトライロジック
  - レート制限対応
  - タイムアウト処理
  - 詳細なエラーハンドリング
- `src/prompt_generator.py` - 64パターンプロンプト生成器
  - YAMLからの設定読み込み
  - テンプレートベース生成
  - メタデータ付与
- `src/experiment_runner.py` - 実験実行エンジン
  - チェックポイント機能（中断・再開対応）
  - プログレスバー表示（tqdm）
  - フィルタリング実験実行
  - JSON形式での結果保存
- `src/data_converter.py` - データ変換ツール
  - JSON → CSV 変換
  - テキスト統計計算（文字数、単語数、行数）
  - UTF-8 BOM 付き出力（Excel互換性）
- `scripts/run_experiment.sh` - 自動実験実行スクリプト
- `tests/test_api_client.py` - APIクライアント単体テスト（11テスト）

#### Phase 4-5: ドキュメンテーションと品質保証
- `docs/methodology.md` - 学術的方法論ドキュメント
  - 実験設計（3要因完全実施要因計画）
  - データ収集手順
  - 倫理的配慮
  - 再現性確保
- `docs/results.md` - 結果報告テンプレート
  - 統計分析フレームワーク
  - 可視化ガイドライン
- `docs/manual.md` - データ分析マニュアル
  - 基本分析コード例
  - 類似度分析（Sentence Transformers）
  - 感情分析（日本語BERT）
  - 可視化例
- `docs/diagrams/system_flow.mermaid` - システムフロー図

#### 検証・テストツール
- `scripts/auto_test.py` - 自動検証スクリプト
  - 4パターン小規模実験
  - エンドツーエンド検証
  - 結果サマリー表示
- `scripts/debug_api.py` - API接続デバッグツール

#### GitHub公開準備ドキュメント
- `CITATION.cff` - 学術引用フォーマットファイル
  - Citation File Format (CFF) v1.2.0
  - BibTeX互換
- `CONTRIBUTING.md` - 貢献ガイドライン
  - 行動規範
  - 開発環境セットアップ
  - コーディング規約（PEP 8準拠）
  - PR/Issueテンプレート
- `CLAUDE.md` - AI支援開発コンテキスト
  - プロジェクト概要とアーキテクチャ
  - 開発履歴（Phase 1-5）
  - コーディング規約
  - トラブルシューティングガイド
- `CHANGELOG.md` - このファイル

### Fixed - 修正

#### HTTP ヘッダーエンコーディングエラー (2025-01-21)
**問題**: `'latin-1' codec can't encode characters in position 3-12`
- **原因**: X-Titleヘッダーの日本語テキスト
- **修正**: `src/api_client.py:106` - ヘッダーを英語に変更
  ```python
  # Before: "X-Title": "LLM地域観光性能比較実験"
  # After: "X-Title": "LLM Tourism Performance Comparison Study"
  ```

#### API認証エラー (2025-01-21)
**問題**: `{"error":{"message":"User not found.","code":401}}`
- **原因**: 無効なOpenRouter APIキー
- **対処**: `scripts/debug_api.py` でデバッグ、APIキー更新を案内
- **検証**: 200 OK レスポンス確認

#### 英語出力問題 (2025-11-21)
**問題**: Google Gemini 2.5 Flash で「外国人観光客」旅行タイプの場合に英語出力が発生
- **検出パターン**: 2/64パターン（3.1%）
  - Google Gemini × 旅行代理店スタッフ × 外国人観光客（英語割合98.7%）
  - Google Gemini × 地方自治体の地域振興担当者 × 外国人観光客（英語割合98.6%）
- **原因**: モデルが「外国人観光客」から外国人向けの英語応答が必要と推論
- **修正**: `config/experiment_config.yaml:99-103` - プロンプトテンプレート変更
  ```yaml
  # Before:
  prompt_template: |
    あなたは{persona}です。下関の観光名所を紹介してください。
    紹介相手は、{travel_type}を計画しています。1000文字程度で要約してください。

  # After (Modification C):
  prompt_template: |
    あなたは{persona}です。以下の指示に従って、必ず日本語で回答してください。

    下関の観光名所を紹介してください。紹介相手は、{travel_type}を計画しています。
    1000文字程度で日本語で要約してください。
  ```
- **検証ツール**: `scripts/detect_english_output.py` - 英語割合検出スクリプト作成
- **テスト**: `scripts/test_japanese_output.py` - 2パターンで検証（英語割合0.0%）
- **再実験**: `scripts/run_all_experiments.py` - 全64パターン再実行
- **結果**: 0/64パターンで英語出力（0%、修正前3.1%）
- **データ管理**: `data/archive/backup_20251121_145858/` に旧データバックアップ
- **知見**: プロンプト冒頭と末尾の両方に明示的な言語指定を含めることで、モデルの推論を上書き可能

### Tested - テスト結果

#### 単体テスト
- **実行日**: 2025-01-21
- **結果**: 11/11 テスト合格 ✓
- **カバレッジ**: APIクライアント全機能

#### 統合テスト（自動検証）
- **実行日**: 2025-01-21
- **パターン**: 4パターン（GPT-5.1 × 家族連れ × 2旅行タイプ × 2回）
- **結果**: 4/4 成功（100%成功率）
- **パフォーマンス**:
  - 平均応答時間: 6.5秒
  - 平均トークン数: 592トークン
  - 平均文字数: 635文字

#### 環境検証
- **Python**: 3.12.11 ✓
- **設定ファイル**: 64パターン生成確認 ✓
- **API接続**: OpenRouter認証成功 ✓

### Changed - 変更

#### データ管理
- `.gitignore` にアーカイブディレクトリ追加（`data/archive/*`）
- テストデータのアーカイブ化ワークフロー確立
- 本実験前のクリーンアップ手順確立

### Security - セキュリティ

#### 検証済み項目
- ✓ `.env` ファイルが `.gitignore` で保護されている
- ✓ `data/raw/*`, `data/processed/*`, `data/archive/*` が除外されている
- ✓ APIキーやシークレットがコミットされていない
- ✓ `.env.example` にプレースホルダーのみ含まれる
- ✓ README.md や docs/ に実際のAPIキーが含まれていない

#### セキュリティ対策
- 環境変数による機密情報管理
- `.gitignore` による自動保護
- `.env.example` によるテンプレート提供

## バージョニングポリシー

本プロジェクトは [Semantic Versioning 2.0.0](https://semver.org/lang/ja/) に準拠します：

- **MAJOR version (X.0.0)**: 互換性のない API 変更
- **MINOR version (0.X.0)**: 後方互換性のある機能追加
- **PATCH version (0.0.X)**: 後方互換性のあるバグ修正

## 開発マイルストーン

### Phase 1: プロジェクト基盤構築 ✓
- 2025-01-21 完了
- ディレクトリ構造、基本ドキュメント、依存関係管理

### Phase 2-3: コアシステム実装 ✓
- 2025-01-21 完了
- API統合、プロンプト生成、実験実行、データ変換

### Phase 4-5: ドキュメンテーションと品質保証 ✓
- 2025-01-21 完了
- 学術文書、分析マニュアル、単体テスト、統合テスト

### GitHub公開準備 ✓
- 2025-01-21 完了
- README更新、引用フォーマット、貢献ガイド、セキュリティチェック

## リンク

- [プロジェクトリポジトリ](https://github.com/nshrhm/llm-tourism)
- [Issue トラッカー](https://github.com/nshrhm/llm-tourism/issues)
- [貢献ガイドライン](CONTRIBUTING.md)
- [ライセンス](LICENSE)

---

**注意**: このCHANGELOGは、プロジェクトの透明性と再現性を確保するために、
すべての重要な変更を記録しています。学術研究の一環として、
変更履歴の正確性を保つことを重視しています。
