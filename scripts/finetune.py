"""
Unslothを使用したファインチューニングスクリプト
"""
import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

def load_model_and_tokenizer(model_name="unsloth/llama-2-7b", max_seq_length=2048):
    """モデルとトークナイザーの読み込み"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    # LoRA設定
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
    )
    
    return model, tokenizer

def prepare_dataset(data_path, tokenizer, max_seq_length=2048):
    """データセットの準備"""
    # JSONLファイルからデータを読み込む例
    if data_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=data_path, split='train')
    else:
        dataset = load_dataset(data_path, split='train')
    
    # プロンプトフォーマット関数
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples["output"]
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            if input_text:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            texts.append(text)
        
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset

def train(model, tokenizer, dataset, output_dir="/workspace/models/finetuned"):
    """モデルの訓練"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="steps",
        save_steps=50,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    
    # 訓練開始
    trainer.train()
    
    # モデルの保存
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Finetune model using Unsloth")
    parser.add_argument("--model", type=str, default="unsloth/llama-2-7b",
                       help="Base model name")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--output", type=str, default="/workspace/models/finetuned",
                       help="Output directory for finetuned model")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # モデルとトークナイザーの読み込み
    model, tokenizer = load_model_and_tokenizer(args.model, args.max_seq_length)
    
    # データセットの準備
    dataset = prepare_dataset(args.data, tokenizer, args.max_seq_length)
    
    # 訓練
    train(model, tokenizer, dataset, args.output)