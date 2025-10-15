#!/bin/bash

# 進捗監視スクリプト
echo "Solar Pro 評価進捗監視を開始します..."
echo "コンテナ: llm-leaderboard"
echo "時刻: $(date)"
echo "=================================="

while true; do
    # コンテナの状態をチェック
    if ! docker ps | grep -q "llm-leaderboard.*Up"; then
        echo "$(date): コンテナが停止しました。評価が完了した可能性があります。"
        break
    fi
    
    # 最新のログから進捗情報を抽出
    echo "$(date): 評価進行中..."
    docker logs llm-leaderboard --tail 20 2>&1 | grep -E "(STARTING|COMPLETED|FINISHED|ERROR|完了|Progress|[0-9]+/[0-9]+|🚀|✅)" | tail -5
    echo "=================================="
    
    # 30秒待機
    sleep 30
done

echo "監視を終了します。"
echo "最終ログ:"
docker logs llm-leaderboard --tail 50
