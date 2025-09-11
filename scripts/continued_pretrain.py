"""
継続事前学習スクリプト
"""
import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments, AutoTokenizer
from trl import SFTTrainer
from unsloth import FastLanguageModel

def load_model_for_continued_pretraining(model_name="unsloth/llama-2-7b", max_seq_length=4096):
    """継続事前学習用のモデル読み込み"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    # 継続事前学習用のLoRA設定（より大きなrankを使用）
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,  # 継続事前学習では大きめのrankを使用
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",
                       "embed_tokens", "lm_head"],  # より多くのレイヤーを対象に
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
    )
    
    return model, tokenizer

def prepare_pretraining_dataset(data_path, tokenizer, max_seq_length=4096):
    """継続事前学習用データセットの準備"""
    # テキストファイルまたはJSONLファイルからデータを読み込む
    if data_path.endswith('.txt'):
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = f.read().split('\n\n')  # パラグラフで分割
        dataset = Dataset.from_dict({"text": texts})
    elif data_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=data_path, split='train')
    else:
        dataset = load_dataset(data_path, split='train')
    
    # トークナイズ関数
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )
    
    # データセットのトークナイズ
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def continued_pretrain(model, tokenizer, dataset, output_dir="/workspace/models/continued_pretrained"):
    """継続事前学習の実行"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # 継続事前学習は通常1エポック
        per_device_train_batch_size=2,  # より大きなシーケンス長のため小さめに
        gradient_accumulation_steps=8,
        warmup_steps=100,
        max_steps=1000,
        learning_rate=5e-5,  # 継続事前学習では低めの学習率
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=20,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        push_to_hub=False,
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=4096,
        dataset_num_proc=4,
        packing=True,  # 継続事前学習では効率化のためpackingを使用
        args=training_args,
    )
    
    # 訓練開始
    print("Starting continued pre-training...")
    trainer.train()
    
    # モデルの保存
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Continued pre-trained model saved to {output_dir}")
    
    # 推論用に最適化されたモデルも保存
    model.save_pretrained_merged(f"{output_dir}_merged", tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {output_dir}_merged")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Continued pre-training using Unsloth")
    parser.add_argument("--model", type=str, default="unsloth/llama-2-7b",
                       help="Base model name or path to finetuned model")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to pre-training data")
    parser.add_argument("--output", type=str, default="/workspace/models/continued_pretrained",
                       help="Output directory for continued pre-trained model")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                       help="Maximum sequence length for pre-training")
    
    args = parser.parse_args()
    
    # モデルとトークナイザーの読み込み
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_for_continued_pretraining(args.model, args.max_seq_length)
    
    # データセットの準備
    print(f"Preparing dataset from: {args.data}")
    dataset = prepare_pretraining_dataset(args.data, tokenizer, args.max_seq_length)
    
    # 継続事前学習
    continued_pretrain(model, tokenizer, dataset, args.output)