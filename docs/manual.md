# データ分析マニュアル

## 概要

本マニュアルは、LLM地域観光紹介性能比較実験の共同研究者向けに、データ構造、分析手順、実用的なコード例を提供します。

**対象読者**: 共同研究者、データアナリスト、再現研究を実施する研究者
**最終更新日**: 2025-01-21
**バージョン**: 1.0.0
**ライセンス**: CC BY 4.0

---

## 目次

1. [データ構造](#1-データ構造)
2. [データ読み込み](#2-データ読み込み)
3. [基礎分析](#3-基礎分析)
4. [テキスト類似度分析](#4-テキスト類似度分析)
5. [感情分析](#5-感情分析)
6. [可視化](#6-可視化)
7. [高度な分析](#7-高度な分析)
8. [トラブルシューティング](#8-トラブルシューティング)

---

## 1. データ構造

### 1.1 ディレクトリ構造

```
data/
├── raw/                          # 生データ（JSON）
│   ├── exp_000_20250121_143022.json
│   ├── exp_001_20250121_143025.json
│   └── ...
└── processed/                    # 処理済みデータ（CSV）
    ├── experiment_results.csv    # メインデータ
    └── statistics_summary.txt    # 統計サマリー
```

### 1.2 JSON データ構造

各実験結果は個別のJSONファイルとして保存されます。

```json
{
  "session_id": "session_20250121_143022",
  "experiment_id": "exp_000",
  "timestamp": "2025-01-21T14:30:22.123456+09:00",
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
  "prompt": "あなたは旅行代理店スタッフです。下関の観光名所を紹介してください。\n紹介相手は、家族旅行を計画しています。1000文字程度で要約してください。",
  "response": "下関は山口県の西端に位置し、...",
  "performance": {
    "tokens_used": 1234,
    "latency_ms": 2345
  },
  "metadata": {
    "model_description": "OpenAI社の最新モデル",
    "persona_description": "プロフェッショナルな旅行提案を行う",
    "persona_characteristics": [
      "正確な情報提供",
      "予算や時間を考慮した提案",
      "顧客ニーズの理解"
    ],
    "travel_type_description": "子供連れの家族向け",
    "target_audience": [
      "小学生の子供を持つ親",
      "3世代旅行",
      "安全性重視"
    ]
  },
  "api_parameters": {
    "temperature": 0.7,
    "max_tokens": 2000,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  }
}
```

### 1.3 CSV データ構造

`experiment_results.csv` のカラム定義：

| カラム名 | データ型 | 説明 | 例 |
|----------|----------|------|-----|
| `session_id` | string | セッション識別子 | session_20250121_143022 |
| `experiment_id` | string | 実験識別子 | exp_000 |
| `timestamp` | datetime | 実行日時（ISO 8601） | 2025-01-21T14:30:22.123456+09:00 |
| `model` | string | モデル名 | openai/gpt-5.1-chat |
| `model_display_name` | string | モデル表示名 | OpenAI GPT-5.1 |
| `persona_id` | string | ペルソナID | travel_agent |
| `persona_name` | string | ペルソナ名 | 旅行代理店スタッフ |
| `travel_type_id` | string | 旅行タイプID | family |
| `travel_type_name` | string | 旅行タイプ名 | 家族旅行 |
| `prompt` | text | 入力プロンプト | あなたは旅行代理店... |
| `response` | text | 生成テキスト | 下関は山口県の... |
| `tokens_used` | integer | 使用トークン数 | 1234 |
| `latency_ms` | integer | 応答時間（ミリ秒） | 2345 |
| `response_char_count` | integer | レスポンス文字数 | 1150 |
| `response_word_count` | integer | レスポンス単語数（推定） | 25 |
| `response_line_count` | integer | レスポンス行数 | 8 |
| `prompt_char_count` | integer | プロンプト文字数 | 98 |

---

## 2. データ読み込み

### 2.1 CSVファイルの読み込み

```python
import pandas as pd

# データ読み込み
df = pd.read_csv('data/processed/experiment_results.csv')

# データ確認
print(f"総レコード数: {len(df)}")
print(f"カラム数: {len(df.columns)}")
print("\nカラム一覧:")
print(df.columns.tolist())

# 先頭5行を表示
print("\nデータサンプル:")
print(df.head())

# データ型の確認
print("\nデータ型:")
print(df.dtypes)
```

### 2.2 日時データの処理

```python
# タイムスタンプをdatetime型に変換
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 日時から情報抽出
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.day_name()

print("実験実施日時の範囲:")
print(f"開始: {df['timestamp'].min()}")
print(f"終了: {df['timestamp'].max()}")
```

### 2.3 JSONファイルの読み込み（個別）

```python
import json
from pathlib import Path

# 単一JSONファイルの読み込み
json_file = Path('data/raw/exp_000_20250121_143022.json')
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(json.dumps(data, ensure_ascii=False, indent=2))
```

### 2.4 全JSONファイルの一括読み込み

```python
import json
from pathlib import Path

def load_all_json_files(directory='data/raw'):
    """すべてのJSONファイルを読み込み"""
    json_dir = Path(directory)
    results = []

    for json_file in json_dir.glob('*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results.append(data)

    return results

# 使用例
all_data = load_all_json_files()
print(f"{len(all_data)}件のJSONファイルを読み込みました")
```

---

## 3. 基礎分析

### 3.1 記述統計

```python
import pandas as pd

# データ読み込み
df = pd.read_csv('data/processed/experiment_results.csv')

# 数値カラムの基礎統計量
numeric_stats = df.describe()
print(numeric_stats)

# カテゴリカルカラムの基礎統計量
categorical_cols = ['model', 'persona_name', 'travel_type_name']
for col in categorical_cols:
    print(f"\n{col}の分布:")
    print(df[col].value_counts())
```

### 3.2 グループ別統計

#### 3.2.1 モデル別統計

```python
# モデル別の平均値
model_stats = df.groupby('model').agg({
    'response_char_count': ['mean', 'std', 'min', 'max'],
    'tokens_used': ['mean', 'std', 'min', 'max'],
    'latency_ms': ['mean', 'std', 'min', 'max']
}).round(2)

print("【モデル別統計】")
print(model_stats)

# CSV保存
model_stats.to_csv('data/processed/model_statistics.csv')
```

#### 3.2.2 ペルソナ別統計

```python
persona_stats = df.groupby('persona_name').agg({
    'response_char_count': ['mean', 'std'],
    'tokens_used': ['mean', 'std']
}).round(2)

print("【ペルソナ別統計】")
print(persona_stats)
```

#### 3.2.3 旅行タイプ別統計

```python
travel_stats = df.groupby('travel_type_name').agg({
    'response_char_count': ['mean', 'std'],
    'tokens_used': ['mean', 'std']
}).round(2)

print("【旅行タイプ別統計】")
print(travel_stats)
```

### 3.3 クロス集計

```python
# モデル × ペルソナのクロス集計（平均文字数）
cross_tab = pd.pivot_table(
    df,
    values='response_char_count',
    index='model',
    columns='persona_name',
    aggfunc='mean'
).round(0)

print("【モデル × ペルソナ クロス集計】")
print(cross_tab)

# Excel出力
cross_tab.to_excel('data/processed/cross_tabulation.xlsx')
```

---

## 4. テキスト類似度分析

### 4.1 環境構築

```bash
# 必要なライブラリのインストール
pip install sentence-transformers scikit-learn
```

### 4.2 基本的な類似度計算

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# モデルのロード
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# レスポンステキストの埋め込み
responses = df['response'].tolist()
embeddings = model.encode(responses, show_progress_bar=True)

# コサイン類似度行列の計算
similarity_matrix = cosine_similarity(embeddings)

print(f"類似度行列のサイズ: {similarity_matrix.shape}")
print(f"平均類似度: {np.mean(similarity_matrix):.4f}")
print(f"類似度の標準偏差: {np.std(similarity_matrix):.4f}")
```

### 4.3 モデル間類似度の分析

```python
# 同一条件下での異なるモデル間の類似度を計算
def calculate_inter_model_similarity(df, embeddings):
    """同一ペルソナ・旅行タイプでの異なるモデル間の類似度"""
    results = []

    for persona in df['persona_id'].unique():
        for travel_type in df['travel_type_id'].unique():
            # 同一条件でフィルタ
            mask = (df['persona_id'] == persona) & (df['travel_type_id'] == travel_type)
            subset = df[mask]

            if len(subset) < 2:
                continue

            # モデルごとの埋め込みを取得
            model_embeddings = {}
            for idx, row in subset.iterrows():
                model_name = row['model']
                embedding_idx = df.index.get_loc(idx)
                model_embeddings[model_name] = embeddings[embedding_idx]

            # 全組み合わせで類似度計算
            models = list(model_embeddings.keys())
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    emb1 = model_embeddings[models[i]].reshape(1, -1)
                    emb2 = model_embeddings[models[j]].reshape(1, -1)
                    sim = cosine_similarity(emb1, emb2)[0][0]

                    results.append({
                        'persona': persona,
                        'travel_type': travel_type,
                        'model1': models[i],
                        'model2': models[j],
                        'similarity': sim
                    })

    return pd.DataFrame(results)

# 実行
inter_model_sim = calculate_inter_model_similarity(df, embeddings)
print(inter_model_sim.groupby(['model1', 'model2'])['similarity'].mean())
```

### 4.4 クラスタリング

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# K-meansクラスタリング
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(embeddings)

# クラスタごとの分布
print("【クラスタ分布】")
print(df['cluster'].value_counts().sort_index())

# クラスタごとのモデル分布
print("\n【クラスタごとのモデル分布】")
print(pd.crosstab(df['cluster'], df['model']))
```

---

## 5. 感情分析

### 5.1 環境構築

```bash
# 必要なライブラリのインストール
pip install transformers torch fugashi unidic-lite
```

### 5.2 感情極性分析

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# モデルのロード（日本語BERT）
model_name = "daigo/bert-base-japanese-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text):
    """感情極性スコアを計算"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)

    return {
        'negative': scores[0][0].item(),
        'neutral': scores[0][1].item(),
        'positive': scores[0][2].item()
    }

# サンプル実行
sample_text = df.iloc[0]['response']
sentiment = analyze_sentiment(sample_text)
print(f"サンプルテキストの感情分析:")
print(sentiment)
```

### 5.3 全データの感情分析

```python
from tqdm import tqdm

# 全レスポンスの感情分析（時間がかかる場合があります）
sentiment_scores = []

for text in tqdm(df['response'].tolist(), desc="感情分析中"):
    try:
        scores = analyze_sentiment(text)
        sentiment_scores.append(scores)
    except Exception as e:
        print(f"エラー: {e}")
        sentiment_scores.append({'negative': 0, 'neutral': 0, 'positive': 0})

# DataFrameに追加
df['sentiment_negative'] = [s['negative'] for s in sentiment_scores]
df['sentiment_neutral'] = [s['neutral'] for s in sentiment_scores]
df['sentiment_positive'] = [s['positive'] for s in sentiment_scores]

# 保存
df.to_csv('data/processed/experiment_results_with_sentiment.csv', index=False)
```

### 5.4 感情分析結果の可視化

```python
import seaborn as sns
import matplotlib.pyplot as plt

# モデル別の平均感情スコア
sentiment_by_model = df.groupby('model')[
    ['sentiment_negative', 'sentiment_neutral', 'sentiment_positive']
].mean()

sentiment_by_model.plot(kind='bar', figsize=(12, 6))
plt.title('モデル別の感情極性分布')
plt.xlabel('モデル')
plt.ylabel('平均スコア')
plt.legend(['ネガティブ', 'ニュートラル', 'ポジティブ'])
plt.tight_layout()
plt.savefig('docs/figures/sentiment_by_model.png', dpi=300)
plt.show()
```

---

## 6. 可視化

### 6.1 基本的な可視化

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 日本語フォント設定（必要に応じて）
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

# スタイル設定
sns.set_style("whitegrid")
sns.set_palette("husl")
```

### 6.2 箱ひげ図

```python
# モデル別の文字数分布
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='model', y='response_char_count')
plt.title('モデル別の生成テキスト文字数分布', fontsize=14)
plt.xlabel('モデル', fontsize=12)
plt.ylabel('文字数', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('docs/figures/model_char_count_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 6.3 ヒートマップ

```python
# ペルソナ × 旅行タイプのヒートマップ
pivot_table = pd.pivot_table(
    df,
    values='response_char_count',
    index='persona_name',
    columns='travel_type_name',
    aggfunc='mean'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5)
plt.title('ペルソナ × 旅行タイプ別の平均文字数', fontsize=14)
plt.xlabel('旅行タイプ', fontsize=12)
plt.ylabel('ペルソナ', fontsize=12)
plt.tight_layout()
plt.savefig('docs/figures/persona_travel_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 6.4 バイオリンプロット

```python
# ペルソナ別のトークン使用量分布
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='persona_name', y='tokens_used')
plt.title('ペルソナ別のトークン使用量分布', fontsize=14)
plt.xlabel('ペルソナ', fontsize=12)
plt.ylabel('トークン数', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('docs/figures/persona_tokens_violin.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 6.5 散布図

```python
# 文字数とトークン数の関係
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='response_char_count', y='tokens_used', hue='model', alpha=0.7)
plt.title('文字数とトークン数の関係', fontsize=14)
plt.xlabel('文字数', fontsize=12)
plt.ylabel('トークン数', fontsize=12)
plt.legend(title='モデル', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('docs/figures/char_token_scatter.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 7. 高度な分析

### 7.1 分散分析（ANOVA）

```python
from scipy import stats

# モデル間の文字数の差の検定
groups = [df[df['model'] == model]['response_char_count']
          for model in df['model'].unique()]

f_stat, p_value = stats.f_oneway(*groups)

print("【分散分析結果】")
print(f"F統計量: {f_stat:.4f}")
print(f"p値: {p_value:.4f}")

if p_value < 0.05:
    print("結論: モデル間に有意な差があります（α=0.05）")
else:
    print("結論: モデル間に有意な差はありません（α=0.05）")
```

### 7.2 多重比較（Post-hoc検定）

```python
from scipy.stats import tukey_hsd

# Tukey HSD検定
res = tukey_hsd(*groups)

print("\n【Tukey HSD検定結果】")
print(res)
```

### 7.3 相関分析

```python
# 数値変数間の相関行列
numeric_cols = ['response_char_count', 'tokens_used', 'latency_ms',
                'response_word_count', 'response_line_count']

correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            square=True, linewidths=0.5)
plt.title('変数間の相関行列', fontsize=14)
plt.tight_layout()
plt.savefig('docs/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 7.4 主成分分析（PCA）

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 数値特徴量の抽出と標準化
features = df[numeric_cols].fillna(0)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# PCA実行
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)

# DataFrame化
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]

# 可視化
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='model', style='persona_name', s=100)
plt.title(f'PCA: 主成分分析（寄与率: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}）',
          fontsize=14)
plt.xlabel(f'第1主成分 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
plt.ylabel(f'第2主成分 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('docs/figures/pca_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 8. トラブルシューティング

### 8.1 よくあるエラー

#### エラー1: CSVファイルが開けない

```python
# 解決策: エンコーディングを明示的に指定
df = pd.read_csv('data/processed/experiment_results.csv', encoding='utf-8-sig')
```

#### エラー2: メモリ不足

```python
# 解決策: チャンクで読み込み
chunk_size = 1000
chunks = []
for chunk in pd.read_csv('data/processed/experiment_results.csv', chunksize=chunk_size):
    # チャンクごとに処理
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
```

#### エラー3: 日本語の文字化け

```python
# Matplotlibのフォント設定
import matplotlib.pyplot as plt

# フォントを変更
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

# または、日本語フォントをインストール
# Linux: sudo apt-get install fonts-noto-cjk
# Mac: すでにインストール済み
# Windows: すでにインストール済み
```

### 8.2 パフォーマンス最適化

```python
# 大規模データの場合、データ型を最適化
def optimize_dtypes(df):
    """データ型を最適化してメモリ使用量を削減"""
    # 整数型の最適化
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    # 浮動小数点型の最適化
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # カテゴリ型への変換
    categorical_cols = ['model', 'persona_name', 'travel_type_name']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    return df

# 使用例
df = optimize_dtypes(df)
print(df.memory_usage(deep=True))
```

---

## 9. 完全な分析パイプライン例

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ========== 1. データ読み込み ==========
df = pd.read_csv('data/processed/experiment_results.csv')
print(f"データ読み込み完了: {len(df)}レコード")

# ========== 2. 基礎統計 ==========
print("\n【基礎統計】")
print(df.describe())

# ========== 3. グループ別統計 ==========
print("\n【モデル別統計】")
model_stats = df.groupby('model')['response_char_count'].describe()
print(model_stats)

# ========== 4. 可視化 ==========
# 箱ひげ図
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='model', y='response_char_count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('analysis_boxplot.png', dpi=300)
plt.close()

# ========== 5. テキスト類似度分析 ==========
print("\n【テキスト類似度分析中...】")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
embeddings = model.encode(df['response'].tolist())
similarity_matrix = cosine_similarity(embeddings)
print(f"平均類似度: {np.mean(similarity_matrix):.4f}")

# ========== 6. レポート生成 ==========
report = f"""
# 分析レポート

## データサマリー
- 総レコード数: {len(df)}
- モデル数: {df['model'].nunique()}
- ペルソナ数: {df['persona_name'].nunique()}
- 旅行タイプ数: {df['travel_type_name'].nunique()}

## 統計結果
- 平均文字数: {df['response_char_count'].mean():.0f}
- 平均トークン数: {df['tokens_used'].mean():.0f}
- 平均応答時間: {df['latency_ms'].mean():.0f}ms

## テキスト類似度
- 平均類似度: {np.mean(similarity_matrix):.4f}
"""

with open('analysis_report.md', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n✓ 分析完了: analysis_report.md を確認してください")
```

---

## 付録

### A. 推奨ツール

- **データ分析**: Jupyter Notebook, VS Code
- **統計解析**: R Studio（Rでの分析も可能）
- **可視化**: matplotlib, seaborn, plotly
- **レポート作成**: Quarto, R Markdown, Jupyter Book

### B. 参考リンク

- Pandas Documentation: https://pandas.pydata.org/docs/
- Matplotlib Gallery: https://matplotlib.org/stable/gallery/
- Sentence Transformers: https://www.sbert.net/
- Transformers (Hugging Face): https://huggingface.co/docs/transformers/

---

**本マニュアルの引用方法**:

```
[著者名]. (2025). LLM地域観光紹介性能比較実験 - データ分析マニュアル.
GitHub. https://github.com/nshrhm/llm-tourism/docs/manual.md
Licensed under CC BY 4.0.
```

**問い合わせ**: GitHubのIssuesまでお願いします。
