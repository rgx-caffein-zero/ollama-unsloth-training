## 使い方

### 1. 環境の構築
```bash
# リポジトリのクローン/作成
mkdir ollama-unsloth-training
cd ollama-unsloth-training

# 必要なファイルを配置（上記のファイルをすべて作成）

# Dockerイメージのビルドと起動
docker-compose up -d

# コンテナに入る
docker exec -it ollama-unsloth-training bash
```

### 2. モデルのセットアップ
```bash
# Ollamaでモデルをダウンロード
python3 scripts/setup_model.py --model llama2:7b
```

### 3. ファインチューニングの実行
```bash
# サンプルデータでファインチューニング
python3 scripts/finetune.py \
    --model unsloth/llama-2-7b \
    --data /workspace/data/sample_finetune.jsonl \
    --output /workspace/models/finetuned_model
```

### 4. 継続事前学習の実行
```bash
# テキストデータで継続事前学習
python3 scripts/continued_pretrain.py \
    --model unsloth/llama-2-7b \
    --data /workspace/data/pretrain_data.txt \
    --output /workspace/models/continued_pretrained_model
```

### 5. Jupyter Notebookの起動
```bash
# コンテナ内で実行
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## 注意事項

1. **GPU要件**: NVIDIA GPUが必要です。CUDAドライバーとnvidia-dockerがインストールされている必要があります。

2. **メモリ要件**: 
   - 7Bモデル: 最低16GB VRAM推奨
   - 13Bモデル: 最低24GB VRAM推奨

3. **ストレージ**: モデルファイルのために十分なディスク容量（50GB以上推奨）が必要です。

4. **Ollamaモデル**: デフォルトではllama2を使用しますが、以下のモデルも利用可能です：
   - mistral:7b
   - codellama:7b
   - phi:latest

5. **データフォーマット**: 
   - ファインチューニング: JSONL形式（instruction, input, outputフィールド）
   - 継続事前学習: プレーンテキストまたはJSONL形式

## トラブルシューティング

### Ollamaが起動しない場合
```bash
# Ollamaサービスの再起動
ollama serve &

# モデルリストの確認
ollama list
```

### CUDA out of memoryエラー
```python
# バッチサイズを小さくする
# finetune.pyやcontinued_pretrain.pyの
# per_device_train_batch_size を 1 または 2 に設定
```

### Unslothのインストールエラー
```bash
# 最新版を再インストール
pip3 uninstall unsloth -y
pip3 install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
