#!/bin/bash

# Reasoning無効版 Solar Pro 評価実行スクリプト
echo "🚀 Solar Pro 評価 (Reasoning無効版) を開始します..."
echo "設定: config-solar-pro-openai-compatible.yaml"
echo ""
echo "🔧 最適化設定:"
echo "- enable_thinking: false (reasoning無効)"
echo "- 超拡張タイムアウト: 6000秒 (100分)"
echo "- inference_interval: 0秒 (最速実行)"
echo ""
echo "🎯 期待される改善:"
echo "- 推論速度大幅向上"
echo "- APIレスポンス時間短縮"  
echo "- より安定したevaluation"
echo ""
echo "時刻: $(date)"
echo ""

# 既存のコンテナを確実に停止・削除
echo "🧹 既存の評価環境をクリーンアップ中..."
docker stop llm-leaderboard 2>/dev/null || true
docker rm llm-leaderboard 2>/dev/null || true
sleep 2

# Reasoning無効版設定で評価を開始
echo "🎯 Reasoning無効版設定で評価を開始..."
./run_with_compose.sh config-solar-pro-openai-compatible.yaml -d

if [ $? -eq 0 ]; then
    echo "✅ 評価開始完了"
    echo "📊 進捗監視: ./monitor_progress.sh"
    echo "🔗 W&B Dashboard: https://wandb.ai/llm-leaderboard/nejumi-leaderboard4" 
    echo ""
    echo "🚄 Reasoning無効により大幅な高速化を実現"
    echo "   前回よりも遥かに高速な評価が期待できます"
else
    echo "❌ 評価開始に失敗しました"
fi
