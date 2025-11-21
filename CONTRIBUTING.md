# 貢献ガイドライン

LLM地域観光紹介性能比較実験プロジェクトへの貢献を歓迎します！

本プロジェクトは**オープンサイエンス**の原則に基づき、学術コミュニティへの貢献を目指しています。あなたの知識、経験、アイデアを共有してください。

## 目次

- [行動規範](#行動規範)
- [貢献の種類](#貢献の種類)
- [開発環境のセットアップ](#開発環境のセットアップ)
- [Issue の報告](#issue-の報告)
- [Pull Request の提出](#pull-request-の提出)
- [コーディング規約](#コーディング規約)
- [テストの実行](#テストの実行)
- [ドキュメントの更新](#ドキュメントの更新)
- [コミュニティ](#コミュニティ)

## 行動規範

本プロジェクトは学術研究プロジェクトとして、以下の原則を重視します：

- **敬意と包容**: すべての貢献者を尊重し、建設的なフィードバックを提供
- **オープンな議論**: 意見の相違を建設的に議論し、合意形成を目指す
- **学術的誠実性**: 研究倫理を遵守し、データの正確性を保つ
- **再現性の重視**: すべての変更が再現可能であることを確認

## 貢献の種類

### 1. バグ報告

バグを発見した場合は、[Issue](https://github.com/nshrhm/llm-tourism/issues)を作成してください。

**バグ報告に含めるべき情報**:
- バグの詳細な説明
- 再現手順
- 期待される動作と実際の動作
- 環境情報（OS、Pythonバージョン等）
- エラーメッセージ（ある場合）

### 2. 機能提案

新機能のアイデアがある場合も、Issueで提案してください。

**機能提案に含めるべき情報**:
- 機能の目的と利点
- 使用例
- 実装のアイデア（あれば）
- 学術研究への影響

### 3. コードの改善

Pull Requestを通じてコードの改善を提案できます。

**改善の例**:
- パフォーマンスの最適化
- エラーハンドリングの改善
- コードの可読性向上
- 新しい分析手法の追加

### 4. ドキュメントの改善

- READMEの明確化
- コメントの追加・改善
- 使用例の追加
- 翻訳（英語版の作成等）

### 5. 実験結果の共有

- 新しい地域での実験結果
- 異なるモデルでの実験
- 分析手法の応用例

## 開発環境のセットアップ

### 1. リポジトリのフォーク

```bash
# GitHubでリポジトリをフォーク後
git clone https://github.com/YOUR_USERNAME/llm-tourism.git
cd llm-tourism
```

### 2. 開発環境の構築

```bash
# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 開発用依存関係のインストール
pip install -r requirements.txt
pip install black flake8 mypy  # コード品質ツール
```

### 3. 環境変数の設定

```bash
cp .env.example .env
# .envを編集してAPIキーを設定
```

### 4. テストの実行

```bash
# 単体テスト
pytest tests/ -v

# カバレッジ付き
pytest tests/ --cov=src --cov-report=html
```

## Issue の報告

### バグ報告テンプレート

```markdown
## バグの説明
[バグの簡潔な説明]

## 再現手順
1. [手順1]
2. [手順2]
3. [手順3]

## 期待される動作
[期待される動作の説明]

## 実際の動作
[実際に起こったことの説明]

## 環境情報
- OS: [例: Ubuntu 22.04]
- Python: [例: 3.11.5]
- プロジェクトバージョン: [例: 1.0.0]

## エラーメッセージ
```
[エラーメッセージをここに貼り付け]
```

## 追加情報
[その他の関連情報]
```

### 機能提案テンプレート

```markdown
## 提案する機能
[機能の簡潔な説明]

## 動機・背景
[なぜこの機能が必要か]

## 提案する実装方法
[実装のアイデア（あれば）]

## 代替案
[検討した代替案（あれば）]

## 追加コンテキスト
[その他の関連情報]
```

## Pull Request の提出

### 1. ブランチの作成

```bash
# 作業用ブランチを作成
git checkout -b feature/your-feature-name
# または
git checkout -b fix/your-bug-fix
```

### 2. 変更の実装

- **小さな変更**: 1つのPRで1つの機能や修正
- **コミットメッセージ**: 明確で説明的なメッセージ
- **日本語コメント**: コード内のコメントは日本語で記述

```bash
# 変更をコミット
git add .
git commit -m "feat: 新しいテキスト類似度計算機能を追加"
```

### 3. テストの実行

```bash
# すべてのテストが通ることを確認
pytest tests/ -v

# コード品質チェック
black src/ tests/  # フォーマット
flake8 src/ tests/  # Lint
```

### 4. Pull Request の作成

```bash
# フォークしたリポジトリにプッシュ
git push origin feature/your-feature-name
```

その後、GitHubでPull Requestを作成してください。

### Pull Request テンプレート

```markdown
## 変更内容
[変更の簡潔な説明]

## 動機と文脈
[なぜこの変更が必要か]

## 変更の種類
- [ ] バグ修正
- [ ] 新機能
- [ ] 破壊的変更
- [ ] ドキュメント更新

## チェックリスト
- [ ] テストが通ることを確認
- [ ] ドキュメントを更新（必要な場合）
- [ ] コーディング規約に従っている
- [ ] コミットメッセージが明確
- [ ] CHANGELOG.mdを更新（必要な場合）

## テスト方法
[この変更をテストする方法]

## スクリーンショット（該当する場合）
[スクリーンショットを貼り付け]

## 関連Issue
Closes #[issue番号]
```

## コーディング規約

### Pythonコーディング規約

本プロジェクトは**PEP 8**に準拠します。

#### フォーマット

```bash
# blackで自動フォーマット
black src/ tests/
```

#### スタイルガイド

```python
# 良い例
def calculate_similarity(
    text1: str,
    text2: str,
    model_name: str = "paraphrase-multilingual-mpnet-base-v2"
) -> float:
    """
    2つのテキストの類似度を計算

    Args:
        text1: 比較するテキスト1
        text2: 比較するテキスト2
        model_name: 使用するモデル名

    Returns:
        コサイン類似度（0.0-1.0）
    """
    # 実装
    pass


# 悪い例
def calc(t1,t2,m="model"):  # 型ヒントなし、短すぎる変数名
    pass  # docstringなし
```

#### 重要な規約

1. **型ヒント**: すべての関数に型ヒントを付ける
2. **Docstring**: すべての関数・クラスにdocstringを記載
3. **コメント**: 日本語で記述
4. **変数名**: 説明的な名前（英語）
5. **関数名**: スネークケース（例: `generate_prompt`）
6. **クラス名**: パスカルケース（例: `PromptGenerator`）

### ファイル構造

```
src/
├── __init__.py
├── module_name.py      # モジュール
└── utils/             # ユーティリティ
    ├── __init__.py
    └── helpers.py

tests/
├── test_module_name.py  # モジュールに対応するテスト
└── conftest.py         # pytest設定
```

## テストの実行

### 単体テスト

```bash
# すべてのテスト
pytest tests/ -v

# 特定のテストファイル
pytest tests/test_api_client.py -v

# 特定のテスト関数
pytest tests/test_api_client.py::TestOpenRouterClientInit::test_init_with_api_key -v
```

### カバレッジ

```bash
# カバレッジレポート生成
pytest tests/ --cov=src --cov-report=html

# HTMLレポートを開く
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### 新しいテストの追加

```python
# tests/test_new_feature.py
import pytest
from src.new_feature import NewFeature


class TestNewFeature:
    """NewFeatureのテスト"""

    def test_basic_functionality(self):
        """基本的な機能のテスト"""
        feature = NewFeature()
        result = feature.do_something()
        assert result == expected_value

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        feature = NewFeature()
        with pytest.raises(ValueError):
            feature.do_invalid_operation()
```

## ドキュメントの更新

### ドキュメントの種類

1. **README.md**: プロジェクト概要、セットアップ手順
2. **docs/methodology.md**: 実験方法論
3. **docs/manual.md**: データ分析マニュアル
4. **docs/results.md**: 結果報告テンプレート
5. **コード内docstring**: 関数・クラスの説明

### ドキュメント更新時の注意点

- **正確性**: 実装と一致する内容を記載
- **明確性**: 初学者にも理解できる説明
- **例の提供**: 具体的な使用例を含める
- **日本語**: 学術論文査読者向けに日本語で記述

## コミュニティ

### コミュニケーションチャネル

- **GitHub Issues**: バグ報告、機能提案、質問
- **Pull Requests**: コードレビューと議論
- **GitHub Discussions**: 一般的な議論（有効な場合）

### レビュープロセス

1. **提出**: Pull Requestを作成
2. **自動チェック**: テストとLintが自動実行
3. **コードレビュー**: メンテナーによるレビュー
4. **修正**: フィードバックに基づき修正
5. **マージ**: 承認後、mainブランチにマージ

### レスポンス時間の目安

- **バグ報告**: 1-3営業日以内に初回レスポンス
- **Pull Request**: 1週間以内にレビュー開始
- **機能提案**: 2週間以内に検討結果を返信

## 謝辞

本プロジェクトへの貢献者は、[CONTRIBUTORS.md](CONTRIBUTORS.md)に記載されます。

## ライセンス

貢献したコードは、プロジェクトのライセンス（CC BY 4.0）の下で公開されることに同意したものとみなされます。

---

**質問がある場合は、遠慮なくIssueで質問してください！**

ご協力ありがとうございます。
