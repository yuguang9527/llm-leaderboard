#!/bin/bash

# 全件評価実行スクリプト
echo "Solar Pro 全件評価を開始します..."
echo "設定: config-solar-pro-openai-compatible.yaml"
echo "testmode: false (全データセット)"
echo "時刻: $(date)"

# 既存のコンテナを停止・削除
echo "既存の評価コンテナを停止中..."
docker stop llm-leaderboard 2>/dev/null || true
docker rm llm-leaderboard 2>/dev/null || true

# 全件評価を開始
echo "全件評価を開始します..."
./run_with_compose.sh config-solar-pro-openai-compatible.yaml -d

echo "評価開始完了。進捗監視スクリプトを確認してください。"
echo "W&B Dashboard: https://wandb.ai/llm-leaderboard/nejumi-leaderboard4"
