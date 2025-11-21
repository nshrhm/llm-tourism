#!/bin/bash

# GitHubリポジトリセットアップスクリプト
# 前提条件：
# 1. ghコマンドがインストールされていること
# 2. gh auth login でGitHubにログイン済みであること

# Git設定値（固定）
GIT_NAME="nshrhm"
GIT_EMAIL="shirahama-na@eco.shimonoseki-cu.ac.jp"

echo "=== GitHubリポジトリセットアップ ==="
echo ""

# リポジトリ名の入力
read -p "作成するリポジトリ名を入力してください: " REPO_NAME

# 入力チェック
if [ -z "$REPO_NAME" ]; then
    echo "エラー: リポジトリ名が入力されていません"
    exit 1
fi

echo "リポジトリ名: $REPO_NAME"

# ghコマンドの確認
if ! command -v gh &> /dev/null; then
    echo "エラー: ghコマンドが見つかりません"
    echo "GitHub CLIをインストールしてください: https://cli.github.com/"
    exit 1
fi

# GitHub認証状態の確認
if ! gh auth status &> /dev/null; then
    echo "エラー: GitHubにログインしていません"
    echo "以下のコマンドでログインしてください: gh auth login"
    exit 1
fi

# GitHubユーザー名を取得
GITHUB_USERNAME=$(gh api user --jq '.login')
echo "GitHubユーザー名: $GITHUB_USERNAME"

# プライベートリポジトリの作成
echo "プライベートリポジトリ '$REPO_NAME' を作成します..."
if gh repo create $REPO_NAME --private --description "$REPO_NAME プロジェクト" --clone=false; then
    echo "リポジトリが正常に作成されました"
else
    echo "警告: リポジトリの作成に失敗しました（既に存在する可能性があります）"
fi

# Gitの設定（固定値を使用）
echo "Gitのユーザー設定を行います..."
echo "名前: $GIT_NAME"
echo "メール: $GIT_EMAIL"

git config --global user.name "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"

# リモートリポジトリの追加
echo "リモートリポジトリを追加します..."
git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git

# メインブランチの設定
git branch -M main

# GitHubにプッシュ
echo "GitHubにプッシュします..."
git push -u origin main

echo ""
echo "=== セットアップ完了 ==="
echo "$REPO_NAME プロジェクトがGitHubのプライベートリポジトリにアップロードされました！"
echo "リポジトリURL: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
