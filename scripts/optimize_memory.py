"""
メモリ最適化設定を自動調整 (RTX 2080 Ti 12GB VRAM最適化版)
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
    """VRAM容量に基づく推奨設定 (RTX 2080 Ti 12GB最適化含む)"""
    
    if vram_gb >= 24:
        return {
            "model": "unsloth/llama-2-13b-chat",
            "max_seq_length": 4096,
            "batch_size": 4,
            "gradient_accumulation": 4,
            "lora_rank": 64,
            "load_in_4bit": True,
            "note": "24GB+ VRAM configuration"
        }
    elif vram_gb >= 16:
        return {
            "model": "unsloth/llama-2-7b-chat",
            "max_seq_length": 4096,
            "batch_size": 2,
            "gradient_accumulation": 8,
            "lora_rank": 32,
            "load_in_4bit": True,
            "note": "16GB VRAM configuration"
        }
    elif 11 <= vram_gb <= 13:  # RTX 2080 Ti (12GB)専用設定
        return {
            "model": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
            "max_seq_length": 2048,
            "batch_size": 1,
            "gradient_accumulation": 16,
            "lora_rank": 16,
            "load_in_4bit": False,  # すでに4bit量子化済み
            "note": "RTX 2080 Ti (12GB VRAM) optimized - DO NOT exceed seq_length 2048"
        }
    elif vram_gb >= 10:
        return {
            "model": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
            "max_seq_length": 1024,
            "batch_size": 1,
            "gradient_accumulation": 16,
            "lora_rank": 8,
            "load_in_4bit": False,
            "note": "10GB VRAM configuration"
        }
    elif vram_gb >= 8:
        return {
            "model": "unsloth/gemma-2b",
            "max_seq_length": 1024,
            "batch_size": 1,
            "gradient_accumulation": 16,
            "lora_rank": 8,
            "load_in_4bit": True,
            "note": "8GB VRAM configuration"
        }
    else:
        return {
            "model": "unsloth/tinyllama-chat",
            "max_seq_length": 512,
            "batch_size": 1,
            "gradient_accumulation": 32,
            "lora_rank": 4,
            "load_in_4bit": True,
            "note": "Low VRAM configuration"
        }

def save_config(config, filepath="training_config.json"):
    """設定をファイルに保存"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {filepath}")

if __name__ == "__main__":
    print("=== GPU Memory Optimization Tool ===")
    print("RTX 2080 Ti (12GB VRAM) Optimized Edition\n")
    
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("❌ No GPU detected!")
        exit(1)
    
    vram_gb = gpu_info['total_memory_gb']
    
    print(f"GPU: {gpu_info['name']}")
    print(f"Total VRAM: {vram_gb:.2f} GB")
    print(f"Compute Capability: {gpu_info['capability']}")
    
    # RTX 2080 Ti検出
    is_rtx_2080ti = "2080" in gpu_info['name']
    if is_rtx_2080ti:
        print("✅ RTX 2080 Ti detected!")
    
    if 'memory_free_mb' in gpu_info:
        print(f"Free VRAM: {gpu_info['memory_free_mb']/1024:.2f} GB")
        print(f"GPU Utilization: {gpu_info['utilization']}%")
    
    print("\n=== Recommended Settings ===")
    settings = recommend_settings(vram_gb)
    
    for key, value in settings.items():
        if key == "note":
            print(f"\nNote: {value}")
        else:
            print(f"{key}: {value}")
    
    # RTX 2080 Ti特有の警告
    if is_rtx_2080ti or (11 <= vram_gb <= 13):
        print("\n⚠️ RTX 2080 Ti (12GB) Important Notes:")
        print("  • DO NOT use max_seq_length > 2048")
        print("  • Use batch_size=1 for safety")
        print("  • BF16 is NOT supported (use FP16)")
        print("  • gradient_checkpointing is REQUIRED")
        print("  • Close all other GPU applications before training")
    
    # 設定を保存
    save_config(settings, "/workspace/training_config.json")
    
    print("\n=== Quick Start Commands for RTX 2080 Ti ===")
    print("\n1. Setup model:")
    print("   python3 scripts/setup_model.py --model-type mistral-7b")
    
    print("\n2. Run finetuning:")
    print("   python3 scripts/finetune.py \\")
    print(f"     --model {settings['model']} \\")
    print("     --data /workspace/data/sample_finetune.jsonl \\")
    print(f"     --max-seq-length {settings['max_seq_length']} \\")
    print(f"     --batch-size {settings['batch_size']} \\")
    print("     --epochs 3")
    
    print("\n3. Run continued pretraining:")
    print("   python3 scripts/continued_pretrain.py \\")
    print("     --model unsloth/llama-2-7b \\")
    print("     --data /workspace/data/pretrain_data.txt \\")
    print(f"     --max-seq-length {settings['max_seq_length']}")
    
    print("\n=== Memory Saving Tips ===")
    print("  • Start with small datasets to test")
    print("  • Monitor GPU usage with: watch -n 1 nvidia-smi")
    print("  • If OOM occurs, reduce max_seq_length to 1024")
    print("  • Consider using gemma-2b for safer operation")
    
    # メモリ予測
    print("\n=== Estimated Memory Usage ===")
    if is_rtx_2080ti or (11 <= vram_gb <= 13):
        print(f"Model loading: ~6-7 GB")
        print(f"Training overhead: ~3-4 GB")
        print(f"Available buffer: ~2-3 GB")
        print(f"\nRecommended: Keep 2GB free for stability")