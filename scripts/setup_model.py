"""
モデルセットアップスクリプト
"""
import os
import subprocess
import torch
import gc
from pathlib import Path

def check_gpu_memory():
    """GPU メモリ状況を確認"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {gpu_memory:.2f} GB")
        return gpu_memory
    return 0

def download_model_with_ollama(model_name="mistral:7b-instruct-q4_0"):
    """Ollamaでモデルをダウンロード（量子化版を優先）"""
    print(f"Downloading {model_name} using Ollama...")
    
    # 推奨モデル
    recommended_models = {
        "mistral": "mistral:7b-instruct-q4_0",  # 4bit量子化版
        "llama2": "llama2:7b-chat-q4_0",        # 4bit量子化版
        "codellama": "codellama:7b-instruct-q4_0",
        "gemma": "gemma:7b-instruct-q4_0",
        "phi": "phi:medium",  # 2.7B
    }
    
    model_to_download = recommended_models.get(model_name.split(':')[0], model_name)
    
    try:
        subprocess.run(["ollama", "pull", model_to_download], check=True)
        print(f"Successfully downloaded {model_to_download}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}")
        return False

def prepare_model(model_type="mistral-7b"):
    """モデルを準備"""
    from unsloth import FastLanguageModel
    
    # メモリクリア
    gc.collect()
    torch.cuda.empty_cache()
    
    # モデル設定
    model_configs = {
        "mistral-7b": {
            "name": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
            "max_seq_length": 2048,
            "dtype": torch.float16,
            "load_in_4bit": False,  # すでに4bit量子化済み
        },
        "llama2-7b": {
            "name": "unsloth/llama-2-7b-chat-bnb-4bit",
            "max_seq_length": 2048,
            "dtype": torch.float16,
            "load_in_4bit": False,
        },
        "gemma-7b": {
            "name": "unsloth/gemma-7b-bnb-4bit",
            "max_seq_length": 2048,
            "dtype": torch.float16,
            "load_in_4bit": False,
        },
        "gemma-2b": {
            "name": "unsloth/gemma-2b",
            "max_seq_length": 2048,
            "dtype": torch.float16,
            "load_in_4bit": True,
        },
        "phi-medium": {
            "name": "microsoft/phi-2",
            "max_seq_length": 2048,
            "dtype": torch.float16,
            "load_in_4bit": True,
        }
    }
    
    config = model_configs.get(model_type, model_configs["mistral-7b"])
    
    print(f"\nLoading model: {config['name']}")
    print(f"Configuration:")
    print(f"  - Max sequence length: {config['max_seq_length']}")
    print(f"  - Dtype: {config['dtype']}")
    print(f"  - 4bit quantization: {config.get('load_in_4bit', False)}")
    
    # モデルとトークナイザーの読み込み
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['name'],
        max_seq_length=config['max_seq_length'],
        dtype=config['dtype'],
        load_in_4bit=config.get('load_in_4bit', False),
        device_map="auto",
        trust_remote_code=True,
    )
    
    # LoRA設定（バランス型）
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # 中程度のrank（8-32の範囲で調整可能）
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth最適化
        random_state=3407,
        use_rslora=False,  # RSLoRAは無効化（メモリ節約）
        max_seq_length=config['max_seq_length'],
    )
    
    # メモリ使用状況を表示
    print(f"\nModel loaded successfully!")
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    return model, tokenizer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup model for training")
    parser.add_argument("--model-type", type=str, default="mistral-7b",
                       choices=["mistral-7b", "llama2-7b", "gemma-7b", "gemma-2b", "phi-medium"],
                       help="Model type to use")
    parser.add_argument("--download-ollama", action="store_true",
                       help="Download model using Ollama first")
    
    args = parser.parse_args()
    
    # GPU情報を表示
    gpu_memory = check_gpu_memory()
    
    if gpu_memory < 10:
        print("\n⚠️ Warning: Less than 10GB VRAM detected. Using smaller model...")
        if gpu_memory < 8:
            args.model_type = "phi-medium"
        else:
            args.model_type = "gemma-2b"
    
    # Ollamaでモデルをダウンロード（オプション）
    if args.download_ollama:
        download_model_with_ollama(args.model_type.replace('-7b', '').replace('-2b', ''))
    
    # モデルの準備
    try:
        model, tokenizer = prepare_model(args.model_type)
        print("\n✅ Model successfully prepared for training!")
    except Exception as e:
        print(f"\n❌ Error preparing model: {e}")
        print("Try using a smaller model or reducing sequence length")