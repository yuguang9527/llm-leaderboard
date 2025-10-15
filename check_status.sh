#!/bin/bash

# 現在の評価状況をチェック
echo "=== Solar Pro 評価状況チェック ==="
echo "時刻: $(date)"
echo ""

# コンテナ状況
echo "📊 コンテナ状況:"
if docker ps | grep -q "llm-leaderboard.*Up"; then
    echo "✅ コンテナ実行中"
    echo "   起動時刻: $(docker ps --format "table {{.CreatedAt}}\t{{.Status}}" | grep llm-leaderboard)"
else
    echo "❌ コンテナ停止"
fi
echo ""

# ログ統計
echo "📈 評価進捗:"
total_logs=$(docker logs llm-leaderboard 2>&1 | wc -l)
conversations=$(docker logs llm-leaderboard 2>&1 | grep -c "role=assistant\|role=user\|role=tool" || echo "0")
echo "   総ログ行数: $total_logs"
echo "   処理済み会話ターン: $conversations"
echo ""

# W&B情報
echo "🔗 W&B Dashboard:"
echo "   https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/runs/c02xdj7a"
echo ""

# 最新の進捗
echo "📋 最新の評価ログ (直近10行):"
docker logs llm-leaderboard --tail 10 2>&1 | grep -v "=================================================================================" | head -10
echo ""

# 選択肢を提示
echo "🤔 次の行動を選択してください:"
echo "1. 現在の評価完了を待つ (testmode)"
echo "2. 現在の評価を停止して全件評価を開始 (testmode: false)"
echo "3. このまま監視を続ける"
