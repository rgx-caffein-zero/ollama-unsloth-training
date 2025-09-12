"""
ファインチューニングスクリプト
"""
import os
import torch
import gc
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import warnings
warnings.filterwarnings("ignore")

def cleanup_memory():
    """メモリのクリーンアップ"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def load_model(model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit", 
               max_seq_length=2048):
    """モデル読み込み"""
    cleanup_memory()
    
    print(f"Loading model: {model_name}")
    print(f"Max sequence length: {max_seq_length}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=False if "bnb-4bit" in model_name else True,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # LoRA設定
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # デフォルトrank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )
    
    return model, tokenizer

def prepare_dataset(data_path, tokenizer, max_seq_length=2048):
    """データセット準備"""
    
    # データ読み込み
    if data_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=data_path, split='train')
    else:
        dataset = load_dataset(data_path, split='train')
    
    # Alpaca形式のプロンプトフォーマット
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    
    def formatting_prompts_func(examples):
        instructions = examples.get("instruction", [])
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples.get("output", [])
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            if input_text:
                text = alpaca_prompt.format(instruction, input_text, output)
            else:
                # inputがない場合のフォーマット
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            
            # 長さ制限
            if len(tokenizer.encode(text)) > max_seq_length:
                text = text[:max_seq_length * 3]  # 概算でカット
            
            texts.append(text)
        
        return {"text": texts}
    
    # データセットの変換
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        num_proc=2,
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    return dataset

def train(model, tokenizer, dataset, output_dir="/workspace/models/finetuned"):
    """訓練設定と実行"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # GPU VRAMに基づいて自動調整
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
    
    # VRAMに基づくバッチサイズの自動調整
    if gpu_memory >= 24:
        batch_size = 4
        gradient_accumulation = 4
    elif gpu_memory >= 16:
        batch_size = 3
        gradient_accumulation = 5
    elif gpu_memory >= 10:
        batch_size = 2
        gradient_accumulation = 8
    else:
        batch_size = 1
        gradient_accumulation = 16
    
    print(f"Auto-configured batch size: {batch_size}, gradient accumulation: {gradient_accumulation}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        warmup_steps=20,
        max_steps=-1,  # 全データを使用
        learning_rate=2e-4,
        fp16=True,  # FP16使用
        bf16=False if gpu_memory < 16 else torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="paged_adamw_8bit",  # 8bitオプティマイザ
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
    )
    
    # SFTTrainerの設定
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,  # 安定性重視
        args=training_args,
    )
    
    # GPU使用状況表示
    print("\n=== Training Configuration ===")
    print(f"Model: {model.config._name_or_path if hasattr(model.config, '_name_or_path') else 'Custom'}")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"Total epochs: {training_args.num_train_epochs}")
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print("================================\n")
    
    try:
        # 訓練開始
        trainer.train()
        
        # モデルの保存
        print("\nSaving model...")
        
        # LoRAアダプターを保存
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # マージ版も保存（推論用）
        print("Saving merged model for inference...")
        model.save_pretrained_merged(
            f"{output_dir}_merged",
            tokenizer,
            save_method="merged_16bit",
        )
        
        print(f"✅ Models saved to:")
        print(f"  - LoRA adapter: {output_dir}")
        print(f"  - Merged model: {output_dir}_merged")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ OOM Error: {e}")
        print("Suggestions:")
        print("  1. Reduce batch_size")
        print("  2. Reduce max_seq_length")
        print("  3. Reduce LoRA rank")
        print("  4. Enable CPU offloading")
        cleanup_memory()
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        cleanup_memory()
        
    finally:
        cleanup_memory()
        print(f"\nFinal GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Finetune model")
    parser.add_argument("--model", type=str, 
                       default="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
                       help="Model name")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--output", type=str, 
                       default="/workspace/models/finetuned",
                       help="Output directory")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Override auto batch size")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--lora-r", type=int, default=16,
                       help="LoRA rank")
    
    args = parser.parse_args()
    
    # メモリクリア
    cleanup_memory()
    
    # モデルとトークナイザーの読み込み
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, args.max_seq_length)
    
    # データセットの準備
    print(f"\nPreparing dataset from: {args.data}")
    dataset = prepare_dataset(args.data, tokenizer, args.max_seq_length)
    
    # 訓練
    train(model, tokenizer, dataset, args.output)