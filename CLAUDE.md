# Claude Code開発コンテキスト

このファイルは、Claude CodeやAI開発アシスタントがこのプロジェクトを理解し、効果的にサポートするためのコンテキスト情報です。

## プロジェクト概要

**プロジェクト名**: LLM地域観光紹介性能比較実験システム

**目的**: 大規模言語モデル（LLM）の地域観光情報生成能力を体系的に評価し、学術研究として公開する

**重要な特徴**:
- 完全に再現可能な実験システム
- オープンサイエンス（CC BY 4.0ライセンス）
- 学術標準準拠（ISO 8601、研究倫理規定）
- テキスト類似度分析・感情分析対応

## アーキテクチャ

### コアコンポーネント

1. **PromptGenerator** (`src/prompt_generator.py`)
   - 64パターン（4モデル × 4ペルソナ × 4旅行タイプ）のプロンプト生成
   - YAML設定ファイルベース
   - メタデータ管理

2. **OpenRouterClient** (`src/api_client.py`)
   - OpenRouter API経由でLLMにアクセス
   - リトライロジック（指数バックオフ）
   - エラーハンドリング
   - レート制限対応

3. **ExperimentRunner** (`src/experiment_runner.py`)
   - 64パターンの実験自動実行
   - 中断・再開機能（チェックポイント）
   - プログレス表示
   - JSON形式でデータ保存

4. **DataConverter** (`src/data_converter.py`)
   - JSON→CSV変換
   - テキスト統計計算
   - Excel互換出力（UTF-8 BOM）

### データフロー

```
設定(YAML) → PromptGenerator → 64パターン生成
                                    ↓
                          ExperimentRunner → OpenRouterClient → LLM API
                                    ↓
                            JSON保存(data/raw/)
                                    ↓
                            DataConverter
                                    ↓
                            CSV出力(data/processed/)
```

## 開発履歴

### 2025-01-21: プロジェクト初版作成

**Phase 1: 基盤構築**
- ディレクトリ構造作成
- `.gitignore`, `.env.example`, `requirements.txt`
- README.md初版

**Phase 2: コアシステム実装**
- `config/experiment_config.yaml`: 実験設定
- `src/api_client.py`: OpenRouter APIクライアント
- `src/prompt_generator.py`: プロンプト生成器
- `src/experiment_runner.py`: 実験実行エンジン

**Phase 3: データ処理**
- `src/data_converter.py`: JSON→CSV変換
- `scripts/run_experiment.sh`: 自動実行スクリプト

**Phase 4: ドキュメント整備**
- `docs/methodology.md`: 実験方法論（学術論文スタイル）
- `docs/results.md`: 結果報告テンプレート
- `docs/manual.md`: データ分析マニュアル
- `docs/diagrams/system_flow.mermaid`: システムフロー図

**Phase 5: 品質保証**
- `tests/test_api_client.py`: 単体テスト（11件合格）
- `LICENSE`: CC BY 4.0全文
- 検証テスト実施（4パターン、成功率100%）

**GitHub公開準備**
- README.md更新（検証結果反映）
- CITATION.cff作成
- CONTRIBUTING.md作成
- セキュリティチェック完了

## コーディング規約

### 重要な原則

1. **日本語コメント**: すべてのコメント・docstringは日本語
2. **型ヒント**: すべての関数に型ヒント必須
3. **PEP 8準拠**: black, flake8でチェック
4. **再現性**: すべてのパラメータを記録
5. **学術標準**: ISO 8601形式、研究倫理規定準拠

### コード例

```python
from typing import Dict, List, Any
from datetime import datetime

def generate_prompts(
    config: Dict[str, Any],
    filter_model: str = None
) -> List[Dict[str, Any]]:
    """
    プロンプトを生成

    Args:
        config: 実験設定
        filter_model: フィルタするモデル名（オプション）

    Returns:
        プロンプト情報のリスト
    """
    # 実装
    pass
```

## 検証済み動作環境

- **Python**: 3.12.11 (3.10以上)
- **OS**: Linux (WSL2)
- **テスト**: 11/11合格
- **統合テスト**: 4パターン成功率100%
- **API**: OpenRouter正常動作確認済み

## よくある作業パターン

### 新機能の追加

1. `src/`に新しいモジュールを作成
2. 型ヒントとdocstringを必ず記載
3. `tests/`にテストを追加
4. README.mdとCHANGELOG.mdを更新

### バグ修正

1. `tests/`に再現テストを追加
2. 修正を実装
3. すべてのテストが通ることを確認
4. CHANGELOG.mdに記載

### データ分析機能の追加

1. `src/data_converter.py`または新規モジュールに実装
2. `docs/manual.md`に使用例を追加
3. サンプルコードを提供

## 重要な制約

### セキュリティ

- **絶対に禁止**: `.env`ファイルのコミット
- **絶対に禁止**: APIキーのハードコーディング
- **絶対に禁止**: `data/`配下のコミット

### 学術研究として

- **データの正確性**: すべてのデータを正確に記録
- **再現性**: パラメータ、日時、環境をすべて記録
- **倫理規定**: 個人情報を含むプロンプトの使用禁止
- **オープンサイエンス**: すべてをCC BY 4.0で公開

## トラブルシューティング

### よくある問題と解決方法

**問題**: `401 Unauthorized`
- **原因**: APIキーが無効
- **解決**: [OpenRouter Keys](https://openrouter.ai/keys)で新しいキーを生成

**問題**: HTTPヘッダーエンコードエラー
- **原因**: 日本語がヘッダーに含まれている
- **解決**: HTTPヘッダーは英語のみ使用（`src/api_client.py:106`参照）

**問題**: データが混在
- **原因**: 古いデータが残っている
- **解決**: `data/archive/`にアーカイブしてから実験実行

## 今後の拡張予定

### 分析機能の追加

- [ ] テキスト類似度分析の実装
- [ ] 感情分析の実装
- [ ] トピックモデリング
- [ ] 可視化ダッシュボード

### システム改善

- [ ] 並列実験実行
- [ ] リアルタイム結果表示
- [ ] Web UI
- [ ] 他のLLMプロバイダー対応

### ドキュメント

- [ ] 英語版README
- [ ] 学術論文執筆
- [ ] チュートリアル動画

## 参考情報

### 外部ドキュメント

- [OpenRouter API Documentation](https://openrouter.ai/docs)
- [PEP 8 Style Guide](https://pep8.org/)
- [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/)

### 内部ドキュメント

- [docs/methodology.md](docs/methodology.md): 実験方法論
- [docs/manual.md](docs/manual.md): データ分析マニュアル
- [CONTRIBUTING.md](CONTRIBUTING.md): 貢献ガイドライン

## メンテナンス情報

### 定期的に確認すべき項目

- [ ] 依存ライブラリの更新（月次）
- [ ] セキュリティアップデート（随時）
- [ ] LLMモデルの更新（四半期）
- [ ] ドキュメントの更新（随時）

### 既知の制限事項

1. **単一地域のみ**: 現在は下関のみ対象
2. **日本語のみ**: 多言語対応は未実装
3. **OpenRouterのみ**: 他のAPIプロバイダー未対応
4. **同期実行**: 並列実行は未実装

## 開発時の注意事項

### Claude Codeを使用する場合

- **コンテキスト**: このファイルを参照してプロジェクトを理解
- **コーディング規約**: 日本語コメント、型ヒント必須
- **テスト**: 変更後は必ず`pytest tests/ -v`を実行
- **セキュリティ**: APIキーを含むファイルは絶対に参照しない

### Git操作

```bash
# 初回コミット前の確認
git status
# .envやdata/が含まれていないことを確認

# 推奨コミットメッセージ形式
git commit -m "feat: 新機能の説明"
git commit -m "fix: バグ修正の説明"
git commit -m "docs: ドキュメント更新"
git commit -m "test: テスト追加"
```

---

**最終更新**: 2025-01-21
**プロジェクトバージョン**: 1.0.0
**ライセンス**: CC BY 4.0
