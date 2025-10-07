"""
モデルセットアップスクリプト (RTX 2080 Ti 12GB VRAM最適化版)
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
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        print(f"Total VRAM: {gpu_memory:.2f} GB")
        
        # RTX 2080 Ti検出
        if "2080" in gpu_name:
            print("✅ RTX 2080 Ti detected - using optimized settings")
        
        return gpu_memory
    return 0

def download_model_with_ollama(model_name="mistral:7b-instruct-q4_0"):
    """Ollamaでモデルをダウンロード（量子化版を優先）"""
    print(f"Downloading {model_name} using Ollama...")
    
    # RTX 2080 Ti (12GB)推奨モデル
    recommended_models = {
        "mistral": "mistral:7b-instruct-q4_0",  # 推奨: 4bit量子化版
        "llama2": "llama2:7b-chat-q4_0",
        "gemma": "gemma:7b-instruct-q4_0",
        "phi": "phi:medium",  # 2.7B - より安全
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
    """モデルを準備 (RTX 2080 Ti 12GB最適化)"""
    from unsloth import FastLanguageModel
    
    # メモリクリア
    gc.collect()
    torch.cuda.empty_cache()
    
    # RTX 2080 Ti (12GB) 用モデル設定
    model_configs = {
        "mistral-7b": {
            "name": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
            "max_seq_length": 2048,  # 12GB用に2048に制限
            "dtype": torch.float16,
            "load_in_4bit": False,  # すでに4bit量子化済み
            "lora_r": 16,
        },
        "llama2-7b": {
            "name": "unsloth/llama-2-7b-chat-bnb-4bit",
            "max_seq_length": 2048,
            "dtype": torch.float16,
            "load_in_4bit": False,
            "lora_r": 16,
        },
        "gemma-7b": {
            "name": "unsloth/gemma-7b-bnb-4bit",
            "max_seq_length": 2048,
            "dtype": torch.float16,
            "load_in_4bit": False,
            "lora_r": 16,
        },
        "gemma-2b": {
            "name": "unsloth/gemma-2b",
            "max_seq_length": 2048,
            "dtype": torch.float16,
            "load_in_4bit": True,
            "lora_r": 8,
        },
        "phi-medium": {
            "name": "microsoft/phi-2",
            "max_seq_length": 2048,
            "dtype": torch.float16,
            "load_in_4bit": True,
            "lora_r": 8,
        }
    }
    
    config = model_configs.get(model_type, model_configs["mistral-7b"])
    
    print(f"\n=== Model Configuration for 12GB VRAM ===")
    print(f"Model: {config['name']}")
    print(f"Max sequence length: {config['max_seq_length']}")
    print(f"Data type: {config['dtype']}")
    print(f"4bit quantization: {config.get('load_in_4bit', False)}")
    print(f"LoRA rank: {config.get('lora_r', 16)}")
    print("==========================================\n")
    
    # モデルとトークナイザーの読み込み
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['name'],
        max_seq_length=config['max_seq_length'],
        dtype=config['dtype'],
        load_in_4bit=config.get('load_in_4bit', False),
        device_map="auto",
        trust_remote_code=True,
    )
    
    # LoRA設定（12GB用に最適化）
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.get('lora_r', 16),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,  # メモリ節約
        max_seq_length=config['max_seq_length'],
    )
    
    # メモリ使用状況を表示
    allocated = torch.cuda.memory_allocated()/1024**3
    reserved = torch.cuda.memory_reserved()/1024**3
    
    print(f"✅ Model loaded successfully!")
    print(f"GPU Memory allocated: {allocated:.2f} GB")
    print(f"GPU Memory reserved: {reserved:.2f} GB")
    print(f"Available for training: ~{12 - reserved:.2f} GB")
    
    if reserved > 10:
        print("\n⚠️ Warning: High memory usage detected.")
        print("Consider using a smaller model or reducing sequence length.")
    
    return model, tokenizer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup model for RTX 2080 Ti (12GB VRAM)")
    parser.add_argument("--model-type", type=str, default="mistral-7b",
                       choices=["mistral-7b", "llama2-7b", "gemma-7b", "gemma-2b", "phi-medium"],
                       help="Model type to use (mistral-7b recommended for 12GB)")
    parser.add_argument("--download-ollama", action="store_true",
                       help="Download model using Ollama first")
    
    args = parser.parse_args()
    
    # GPU情報を表示
    gpu_memory = check_gpu_memory()
    
    if gpu_memory == 0:
        print("\n❌ No GPU detected!")
        exit(1)
    
    # 12GB以下の場合は警告
    if gpu_memory < 12:
        print(f"\n⚠️ Warning: {gpu_memory:.1f}GB VRAM detected. Recommended: 12GB+")
        if gpu_memory < 8:
            print("Using smallest model (phi-medium)...")
            args.model_type = "phi-medium"
        elif gpu_memory < 10:
            print("Using smaller model (gemma-2b)...")
            args.model_type = "gemma-2b"
    
    # 12GBちょうどの場合は最適化版を推奨
    if 11 <= gpu_memory <= 12.5:
        print("\n✅ 12GB VRAM optimal configuration")
        print("Recommended models: mistral-7b, llama2-7b (4-bit quantized)")
    
    # Ollamaでモデルをダウンロード（オプション）
    if args.download_ollama:
        download_model_with_ollama(args.model_type.replace('-7b', '').replace('-2b', ''))
    
    # モデルの準備
    try:
        print("\nInitializing model...")
        model, tokenizer = prepare_model(args.model_type)
        print("\n✅ Model successfully prepared for training!")
        print("\n=== Next Steps ===")
        print("1. Prepare your training data in JSONL format")
        print("2. Run training:")
        print("   python3 scripts/finetune.py \\")
        print("     --data /workspace/data/train.jsonl \\")
        print("     --output /workspace/models/finetuned \\")
        print("     --batch-size 1 \\")
        print("     --max-seq-length 2048")
        
    except torch.cuda.OutOfMemoryError:
        print("\n❌ Out of Memory Error!")
        print("Try using a smaller model:")
        print("  --model-type gemma-2b")
        print("  --model-type phi-medium")
    except Exception as e:
        print(f"\n❌ Error preparing model: {e}")
        print("Try using a smaller model or reducing sequence length")