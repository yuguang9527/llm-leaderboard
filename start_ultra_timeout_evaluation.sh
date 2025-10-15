#!/bin/bash

# 超拡張タイムアウト版 Solar Pro 評価実行スクリプト
echo "🚀 Solar Pro 評価 (超拡張タイムアウト版) を開始します..."
echo "設定: config-solar-pro-openai-compatible.yaml"
echo ""
echo "🔧 タイムアウト設定 (大幅拡張):"
echo "- inference_interval: 0秒 (即座実行)"
echo "- connect timeout: 300秒 (5分)"
echo "- read timeout: 6000秒 (100分)"
echo "- write timeout: 6000秒 (100分)" 
echo "- pool timeout: 600秒 (10分)"
echo ""
echo "前回のAPIタイムアウト問題を完全に解消する設定です！"
echo "時刻: $(date)"
echo ""

# 既存のコンテナを確実に停止・削除
echo "🧹 既存の評価環境をクリーンアップ中..."
docker stop llm-leaderboard 2>/dev/null || true
docker rm llm-leaderboard 2>/dev/null || true
sleep 2

# 超拡張タイムアウト版設定で評価を開始
echo "🎯 超拡張タイムアウト設定で評価を開始..."
./run_with_compose.sh config-solar-pro-openai-compatible.yaml -d

if [ $? -eq 0 ]; then
    echo "✅ 評価開始完了"
    echo "📊 進捗監視: ./monitor_progress.sh" 
    echo "🔗 W&B Dashboard: https://wandb.ai/llm-leaderboard/nejumi-leaderboard4"
    echo ""
    echo "🛡️ 超拡張タイムアウト (最大100分) により"
    echo "   Solar ProのAPIレスポンス遅延に完全対応できます"
else
    echo "❌ 評価開始に失敗しました"
fi
