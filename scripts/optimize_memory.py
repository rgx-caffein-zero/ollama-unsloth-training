"""
メモリ最適化設定を自動調整
"""
import torch
import subprocess
import json

def get_gpu_info():
    """GPU情報を取得"""
    if not torch.cuda.is_available():
        return None
    
    gpu_info = {
        "name": torch.cuda.get_device_name(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "capability": torch.cuda.get_device_capability(0),
    }
    
    # nvidia-smiから詳細情報を取得
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            total, free, util = result.stdout.strip().split(', ')
            gpu_info["memory_free_mb"] = int(free)
            gpu_info["utilization"] = int(util)
    except:
        pass
    
    return gpu_info

def recommend_settings(vram_gb):
    """VRAM容量に基づく推奨設定"""
    
    if vram_gb >= 24:
        return {
            "model": "unsloth/llama-2-13b-chat",
            "max_seq_length": 4096,
            "batch_size": 4,
            "gradient_accumulation": 4,
            "lora_rank": 64,
            "load_in_4bit": True,
        }
    elif vram_gb >= 16:
        return {
            "model": "unsloth/llama-2-7b-chat",
            "max_seq_length": 4096,
            "batch_size": 4,
            "gradient_accumulation": 4,
            "lora_rank": 32,
            "load_in_4bit": True,
        }
    elif vram_gb >= 10:
        return {
            "model": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
            "max_seq_length": 2048,
            "batch_size": 2,
            "gradient_accumulation": 8,
            "lora_rank": 16,
            "load_in_4bit": False,  # すでに量子化済み
        }
    elif vram_gb >= 8:
        return {
            "model": "unsloth/gemma-2b",
            "max_seq_length": 1024,
            "batch_size": 1,
            "gradient_accumulation": 16,
            "lora_rank": 8,
            "load_in_4bit": True,
        }
    else:
        return {
            "model": "unsloth/tinyllama-chat",
            "max_seq_length": 512,
            "batch_size": 1,
            "gradient_accumulation": 32,
            "lora_rank": 4,
            "load_in_4bit": True,
        }

def save_config(config, filepath="training_config.json"):
    """設定をファイルに保存"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {filepath}")

if __name__ == "__main__":
    print("=== GPU Memory Optimization Tool ===\n")
    
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("No GPU detected!")
        exit(1)
    
    print(f"GPU: {gpu_info['name']}")
    print(f"Total VRAM: {gpu_info['total_memory_gb']:.2f} GB")
    print(f"Compute Capability: {gpu_info['capability']}")
    
    if 'memory_free_mb' in gpu_info:
        print(f"Free VRAM: {gpu_info['memory_free_mb']/1024:.2f} GB")
        print(f"GPU Utilization: {gpu_info['utilization']}%")
    
    print("\n=== Recommended Settings ===")
    settings = recommend_settings(gpu_info['total_memory_gb'])
    
    for key, value in settings.items():
        print(f"{key}: {value}")
    
    # 設定を保存
    save_config(settings, "/workspace/training_config.json")
    
    print("\n=== Quick Start Commands ===")
    print(f"1. Setup model:")
    print(f"   python3 scripts/setup_model.py --model-type mistral-7b")
    print(f"\n2. Run training:")
    print(f"   python3 scripts/finetune.py \\")
    print(f"     --model {settings['model']} \\")
    print(f"     --data /workspace/data/train.jsonl \\")
    print(f"     --max-seq-length {settings['max_seq_length']} \\")
    print(f"     --batch-size {settings['batch_size']} \\")
    print(f"     --lora-r {settings['lora_rank']}")