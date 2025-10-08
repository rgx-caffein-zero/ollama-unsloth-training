"""
継続事前学習スクリプト (10.8GB VRAM最適化版)
"""
import os
import torch
import gc
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

def cleanup_memory():
    """メモリのクリーンアップ"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def load_model_for_pretraining(model_name="unsloth/llama-2-7b", max_seq_length=1024):
    """継続事前学習用のモデル読み込み (10.8GB VRAM最適化)"""
    cleanup_memory()
    
    # 10.8GB VRAMではmax_seq_lengthを1536に制限
    max_seq_length = min(max_seq_length, 1536)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto",
    )
    
    # 継続事前学習用のLoRA設定 (10.8GB用に削減)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # 10.8GB用にrankを16に削減
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )
    
    return model, tokenizer

def prepare_pretraining_dataset(data_path, tokenizer, max_seq_length=1024):
    """継続事前学習用データセットの準備"""
    
    # 10.8GB VRAMではmax_seq_lengthを1536に制限
    max_seq_length = min(max_seq_length, 1536)
    
    # テキストファイルまたはJSONLファイルからデータを読み込む
    if data_path.endswith('.txt'):
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = f.read().split('\n\n')  # パラグラフで分割
        dataset = Dataset.from_dict({"text": texts})
    elif data_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=data_path, split='train')
    else:
        dataset = load_dataset(data_path, split='train')
    
    # 長さ制限を適用
    def process_function(examples):
        texts = examples["text"]
        processed_texts = []
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_seq_length:
                # トークン数で正確にカット
                tokens = tokens[:max_seq_length]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
            processed_texts.append(text)
        return {"text": processed_texts}
    
    dataset = dataset.map(
        process_function,
        batched=True,
        num_proc=2,
    )
    
    return dataset

def continued_pretrain(model, tokenizer, dataset, output_dir="/workspace/models/continued_pretrained"):
    """継続事前学習の実行 (10.8GB VRAM最適化)"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # GPU VRAMに基づいて自動調整
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    
    # VRAM別設定
    if gpu_memory >= 24:
        batch_size = 2
        gradient_accumulation = 4
        max_seq = 2048
    elif gpu_memory >= 16:
        batch_size = 2
        gradient_accumulation = 4
        max_seq = 2048
    elif gpu_memory >= 12:
        batch_size = 1
        gradient_accumulation = 8
        max_seq = 2048
    elif gpu_memory >= 10:  # 10.8GB用
        batch_size = 1
        gradient_accumulation = 16
        max_seq = 1024
    elif gpu_memory >= 8:
        batch_size = 1
        gradient_accumulation = 32
        max_seq = 1024
    else:
        batch_size = 1
        gradient_accumulation = 64
        max_seq = 512
    
    print(f"Auto-configured for {gpu_memory:.1f}GB VRAM:")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation}")
    print(f"  Max sequence length: {max_seq}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # 継続事前学習は通常1エポック
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        warmup_steps=50,  # 削減
        max_steps=500,  # 削減
        learning_rate=5e-5,  # 継続事前学習では低めの学習率
        fp16=True,
        bf16=False,  # BF16非対応
        logging_steps=20,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        save_strategy="steps",
        save_steps=100,  # 削減
        save_total_limit=1,  # ディスク容量節約
        push_to_hub=False,
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
        gradient_checkpointing=True,
        dataloader_pin_memory=False,  # メモリ節約
        dataloader_num_workers=1,  # 削減
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq,
        dataset_num_proc=2,
        packing=False,  # 10.8GBではpackingを無効化（安定性重視）
        args=training_args,
    )
    
    # 訓練開始
    print("\n=== Starting Continued Pre-training ===")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Effective batch size: {batch_size * gradient_accumulation}")
    print(f"Max steps: {training_args.max_steps}")
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print("=========================================\n")
    
    try:
        trainer.train()
        
        # モデルの保存
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Continued pre-trained model saved to {output_dir}")
        
        # 推論用に最適化されたモデルも保存
        model.save_pretrained_merged(f"{output_dir}_merged", tokenizer, save_method="merged_16bit")
        print(f"✅ Merged model saved to {output_dir}_merged")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ OOM Error: {e}")
        print("\n10.8GB VRAM用の推奨設定:")
        print("  1. batch_size=1 を使用")
        print("  2. max_seq_length=512 に削減")
        print("  3. より小さいモデル (gemma-2b) を使用")
        cleanup_memory()
        raise
    finally:
        cleanup_memory()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Continued pre-training for 10.8GB VRAM")
    parser.add_argument("--model", type=str, default="unsloth/llama-2-7b",
                       help="Base model name or path to finetuned model")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to pre-training data")
    parser.add_argument("--output", type=str, default="/workspace/models/continued_pretrained",
                       help="Output directory")
    parser.add_argument("--max-seq-length", type=int, default=1024,
                       help="Maximum sequence length (max 1536 for 10.8GB, 1024 recommended)")
    
    args = parser.parse_args()
    
    # GPU確認
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Detected GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {gpu_mem:.1f} GB\n")
    
    # モデルとトークナイザーの読み込み
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_for_pretraining(args.model, args.max_seq_length)
    
    # データセットの準備
    print(f"Preparing dataset from: {args.data}")
    dataset = prepare_pretraining_dataset(args.data, tokenizer, args.max_seq_length)
    
    # 継続事前学習
    continued_pretrain(model, tokenizer, dataset, args.output)