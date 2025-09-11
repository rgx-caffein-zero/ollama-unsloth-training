#!/bin/bash

# Ollamaサービスの起動
ollama serve &

# Ollamaが起動するまで待機
sleep 5

# 引数がある場合はそれを実行、なければbashを起動
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec bash
fi