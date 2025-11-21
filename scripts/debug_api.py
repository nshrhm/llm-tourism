"""
API接続デバッグスクリプト

OpenRouter APIの接続をテストし、詳細なエラー情報を表示します。
"""

import requests
import json
import os

# .envから直接読み込み
env_vars = {}
with open('.env', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            env_vars[key] = value

api_key = env_vars.get('OPENROUTER_API_KEY', '')

print("=" * 80)
print("OpenRouter API 接続テスト")
print("=" * 80)
print()
print(f"APIキー: {api_key[:15]}...（長さ: {len(api_key)}文字）")
print()

# シンプルなテストリクエスト
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

payload = {
    "model": "openai/gpt-3.5-turbo",  # より安価なモデルでテスト
    "messages": [
        {
            "role": "user",
            "content": "こんにちは"
        }
    ],
    "max_tokens": 10
}

print("リクエストURL:", url)
print("モデル:", payload["model"])
print()
print("リクエスト送信中...")

try:
    response = requests.post(url, headers=headers, json=payload, timeout=30)

    print(f"ステータスコード: {response.status_code}")
    print()

    if response.status_code == 200:
        print("✓ 成功！")
        data = response.json()
        print()
        print("レスポンス:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print("✗ エラー")
        print()
        print("レスポンステキスト:")
        print(response.text)
        print()

        # JSONとして解析できるか試す
        try:
            error_data = response.json()
            print("エラー詳細:")
            print(json.dumps(error_data, indent=2, ensure_ascii=False))
        except:
            pass

except Exception as e:
    print(f"✗ 例外発生: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
