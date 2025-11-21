# 実験結果報告

## 概要

本文書は、LLM地域観光紹介性能比較実験の結果報告テンプレートです。実験完了後、このテンプレートに従って結果をまとめることで、学術論文として発表可能な形式の報告書を作成できます。

**最終更新日**: 2025-01-21
**バージョン**: 1.0.0
**ライセンス**: CC BY 4.0

---

## 1. 実験概要

### 1.1 実験実施情報

| 項目 | 詳細 |
|------|------|
| **実験実施日** | [YYYY-MM-DD] |
| **セッションID** | [session_YYYYMMDD_HHMMSS] |
| **総実験数** | 64パターン |
| **成功数** | [X]件 |
| **失敗数** | [X]件 |
| **実行時間** | [X]分 |

### 1.2 使用モデル

1. OpenAI GPT-5.1 Chat
2. Anthropic Claude Sonnet 4.5
3. Google Gemini 2.5 Flash
4. xAI Grok-4 Fast

### 1.3 実験パラメータ

- **ペルソナ**: 4種類（旅行代理店、YouTuber、自治体職員、インフルエンサー）
- **旅行タイプ**: 4種類（家族、カップル、外国人、シニア）
- **対象地域**: 山口県下関市
- **API設定**: temperature=0.7, max_tokens=2000

---

## 2. 定量分析結果

### 2.1 基礎統計量

#### 2.1.1 モデル別統計

**生成テキストの文字数**

| モデル | 平均 | 標準偏差 | 最小値 | 最大値 |
|--------|------|----------|--------|--------|
| GPT-5.1 | [X] | [X] | [X] | [X] |
| Claude Sonnet 4.5 | [X] | [X] | [X] | [X] |
| Gemini 2.5 Flash | [X] | [X] | [X] | [X] |
| Grok-4 Fast | [X] | [X] | [X] | [X] |

**使用トークン数**

| モデル | 平均 | 標準偏差 | 最小値 | 最大値 |
|--------|------|----------|--------|--------|
| GPT-5.1 | [X] | [X] | [X] | [X] |
| Claude Sonnet 4.5 | [X] | [X] | [X] | [X] |
| Gemini 2.5 Flash | [X] | [X] | [X] | [X] |
| Grok-4 Fast | [X] | [X] | [X] | [X] |

**応答時間（ミリ秒）**

| モデル | 平均 | 標準偏差 | 最小値 | 最大値 |
|--------|------|----------|--------|--------|
| GPT-5.1 | [X] | [X] | [X] | [X] |
| Claude Sonnet 4.5 | [X] | [X] | [X] | [X] |
| Gemini 2.5 Flash | [X] | [X] | [X] | [X] |
| Grok-4 Fast | [X] | [X] | [X] | [X] |

#### 2.1.2 分析コード例

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# データ読み込み
df = pd.read_csv('data/processed/experiment_results.csv')

# モデル別の基礎統計
model_stats = df.groupby('model').agg({
    'response_char_count': ['mean', 'std', 'min', 'max'],
    'tokens_used': ['mean', 'std', 'min', 'max'],
    'latency_ms': ['mean', 'std', 'min', 'max']
}).round(2)

print(model_stats)
```

### 2.2 比較分析

#### 2.2.1 モデル間の統計的検定

**分散分析（ANOVA）**

```python
from scipy import stats

# 文字数の分散分析
groups = [df[df['model'] == model]['response_char_count']
          for model in df['model'].unique()]
f_stat, p_value = stats.f_oneway(*groups)

print(f"F統計量: {f_stat:.4f}")
print(f"p値: {p_value:.4f}")
```

**結果**:
- F統計量: [X.XXXX]
- p値: [X.XXXX]
- 解釈: [有意差の有無とその解釈]

#### 2.2.2 多重比較（Post-hoc検定）

```python
from scipy.stats import tukey_hsd

# Tukey HSD検定
res = tukey_hsd(*groups)
print(res)
```

### 2.3 可視化

#### 2.3.1 箱ひげ図（Box Plot）

```python
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='model', y='response_char_count')
plt.title('モデル別の文字数分布')
plt.xlabel('モデル')
plt.ylabel('文字数')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('docs/figures/model_char_count_boxplot.png', dpi=300)
plt.show()
```

![モデル別文字数分布](figures/model_char_count_boxplot.png)
*図1: モデル別の生成テキスト文字数分布*

#### 2.3.2 ヒートマップ

```python
# ペルソナ×旅行タイプの平均文字数
pivot_table = df.pivot_table(
    values='response_char_count',
    index='persona_name',
    columns='travel_type_name',
    aggfunc='mean'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd')
plt.title('ペルソナ×旅行タイプ別の平均文字数')
plt.tight_layout()
plt.savefig('docs/figures/persona_traveltype_heatmap.png', dpi=300)
plt.show()
```

![ヒートマップ](figures/persona_traveltype_heatmap.png)
*図2: ペルソナ×旅行タイプ別の平均文字数ヒートマップ*

---

## 3. テキスト類似度分析

### 3.1 手法

Sentence Transformersを用いた多言語対応の意味的類似度分析。

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# モデルのロード
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# 埋め込みベクトルの生成
responses = df['response'].tolist()
embeddings = model.encode(responses)

# コサイン類似度の計算
similarity_matrix = cosine_similarity(embeddings)
```

### 3.2 結果

#### 3.2.1 モデル間類似度

**同一条件下での異なるモデル間の平均類似度**

| モデル1 | モデル2 | 平均類似度 |
|---------|---------|-----------|
| GPT-5.1 | Claude Sonnet 4.5 | [X.XX] |
| GPT-5.1 | Gemini 2.5 Flash | [X.XX] |
| GPT-5.1 | Grok-4 Fast | [X.XX] |
| Claude Sonnet 4.5 | Gemini 2.5 Flash | [X.XX] |
| Claude Sonnet 4.5 | Grok-4 Fast | [X.XX] |
| Gemini 2.5 Flash | Grok-4 Fast | [X.XX] |

#### 3.2.2 クラスタリング分析

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 階層的クラスタリング
linkage_matrix = linkage(embeddings, method='ward')

plt.figure(figsize=(15, 8))
dendrogram(linkage_matrix, labels=df['experiment_id'].tolist())
plt.title('生成テキストの階層的クラスタリング')
plt.xlabel('実験ID')
plt.ylabel('距離')
plt.tight_layout()
plt.savefig('docs/figures/clustering_dendrogram.png', dpi=300)
plt.show()
```

![クラスタリング](figures/clustering_dendrogram.png)
*図3: 生成テキストの階層的クラスタリング*

---

## 4. 感情分析結果

### 4.1 手法

日本語BERTモデルを用いた感情極性分析。

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# モデルとトークナイザーのロード
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 感情スコアの計算
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    return scores[0].detach().numpy()

df['sentiment_positive'] = df['response'].apply(
    lambda x: analyze_sentiment(x)[1]  # ポジティブスコア
)
```

### 4.2 結果

#### 4.2.1 モデル別感情傾向

| モデル | 平均ポジティブスコア | 標準偏差 |
|--------|---------------------|----------|
| GPT-5.1 | [X.XX] | [X.XX] |
| Claude Sonnet 4.5 | [X.XX] | [X.XX] |
| Gemini 2.5 Flash | [X.XX] | [X.XX] |
| Grok-4 Fast | [X.XX] | [X.XX] |

#### 4.2.2 ペルソナ別感情傾向

| ペルソナ | 平均ポジティブスコア |
|---------|---------------------|
| 旅行代理店スタッフ | [X.XX] |
| 旅行系YouTuber | [X.XX] |
| 自治体職員 | [X.XX] |
| SNSインフルエンサー | [X.XX] |

---

## 5. 定性分析結果

### 5.1 内容分析

#### 5.1.1 言及された観光資源

**主要観光スポットの言及頻度**

| 観光資源 | 言及回数 | 言及率 |
|---------|---------|--------|
| 関門海峡 | [X] | [X%] |
| 唐戸市場 | [X] | [X%] |
| 巌流島 | [X] | [X%] |
| 赤間神宮 | [X] | [X%] |
| ふぐ料理 | [X] | [X%] |

```python
# キーワード出現頻度分析
keywords = ['関門海峡', '唐戸市場', '巌流島', '赤間神宮', 'ふぐ']

for keyword in keywords:
    count = df['response'].str.contains(keyword).sum()
    rate = (count / len(df)) * 100
    print(f"{keyword}: {count}回 ({rate:.1f}%)")
```

#### 5.1.2 表現の特徴

**ペルソナ別の特徴的な表現**

| ペルソナ | 特徴 | 例 |
|---------|------|-----|
| 旅行代理店 | 具体的な情報、価格への言及 | [具体例] |
| YouTuber | カジュアルな表現、感嘆詞 | [具体例] |
| 自治体職員 | 歴史的背景、公式情報 | [具体例] |
| インフルエンサー | SNS用語、トレンドワード | [具体例] |

### 5.2 ターゲット適合性評価

#### 5.2.1 評価基準

1. **家族旅行**: 子供向け施設、安全性、教育的価値の言及
2. **カップル旅行**: ロマンチックな表現、2人向けスポット
3. **外国人観光客**: 日本文化、英語対応、異文化体験
4. **シニア旅行**: バリアフリー、ゆったりペース、健康配慮

#### 5.2.2 評価結果

[評価者による5段階評価の結果を記載]

---

## 6. 考察

### 6.1 主要な発見

1. **モデル性能の差異**
   - [モデル間の主要な違いについて記述]

2. **ペルソナの影響**
   - [ペルソナ設定が出力に与えた影響について記述]

3. **ターゲット適応性**
   - [旅行タイプへの適応性について記述]

### 6.2 研究上の示唆

- [学術的な意義について記述]
- [観光産業への応用可能性について記述]
- [LLM評価手法への貢献について記述]

### 6.3 限界と今後の課題

1. **研究の限界**
   - 単一地域のみを対象
   - 特定時点でのモデルバージョン
   - 評価の主観性

2. **今後の課題**
   - 複数地域での検証
   - 時系列での性能追跡
   - 多言語対応の実験

---

## 7. 結論

[実験結果の総括を記述]

---

## 8. 参考資料

### 8.1 生成されたファイル

- 生データ: `data/raw/`
- CSV: `data/processed/experiment_results.csv`
- 統計サマリー: `data/processed/statistics_summary.txt`

### 8.2 可視化例

すべての図表は `docs/figures/` ディレクトリに保存されています。

### 8.3 分析スクリプト

詳細な分析スクリプトは `docs/manual.md` を参照してください。

---

## 付録: 追加分析コード

### A. 詳細な統計分析

```python
# 3元配置分散分析（モデル × ペルソナ × 旅行タイプ）
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('response_char_count ~ C(model) * C(persona_id) * C(travel_type_id)',
            data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
```

### B. 高度な可視化

```python
# 3Dプロット
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# データのエンコーディング
model_codes = pd.Categorical(df['model']).codes
persona_codes = pd.Categorical(df['persona_id']).codes

scatter = ax.scatter(model_codes, persona_codes, df['response_char_count'],
                    c=df['tokens_used'], cmap='viridis')
ax.set_xlabel('モデル')
ax.set_ylabel('ペルソナ')
ax.set_zlabel('文字数')
plt.colorbar(scatter, label='トークン数')
plt.savefig('docs/figures/3d_scatter.png', dpi=300)
plt.show()
```

---

**本文書の引用方法**:

```
[著者名]. (2025). LLM地域観光紹介性能比較実験 - 結果報告.
GitHub. https://github.com/nshrhm/llm-tourism/docs/results.md
Licensed under CC BY 4.0.
```
