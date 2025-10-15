#!/bin/bash

# 改良版 Solar Pro 評価実行スクリプト
echo "🚀 Solar Pro 評価 (改良版) を開始します..."
echo "設定: config-solar-pro-openai-compatible.yaml"
echo "改良点:"
echo "- inference_interval: 1秒 (前回: 0秒)"
echo "- HTTP timeout拡張: 600秒 (前回: 300秒)"
echo "- Connect timeout拡張: 30秒 (前回: 10秒)"
echo "時刻: $(date)"
echo ""

# 既存のコンテナを確実に停止・削除
echo "🧹 既存の評価環境をクリーンアップ中..."
docker stop llm-leaderboard 2>/dev/null || true
docker rm llm-leaderboard 2>/dev/null || true

# 少し待機
sleep 3

# 改良版設定で評価を開始
echo "🎯 改良版設定で評価を開始..."
./run_with_compose.sh config-solar-pro-openai-compatible.yaml -d

if [ $? -eq 0 ]; then
    echo "✅ 評価開始完了"
    echo "📊 進捗監視: ./monitor_progress.sh"
    echo "🔗 W&B Dashboard: https://wandb.ai/llm-leaderboard/nejumi-leaderboard4"
    echo ""
    echo "改良点により以前のタイムアウト問題が改善される見込みです"
else
    echo "❌ 評価開始に失敗しました"
fi
