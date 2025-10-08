"""
メモリ最適化設定を自動調整 (10.8GB VRAM最適化版)
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
    """VRAM容量に基づく推奨設定 (10.8GB最適化含む)"""
    
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
    elif vram_gb >= 12:
        return {
            "model": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
            "max_seq_length": 2048,
            "batch_size": 1,
            "gradient_accumulation": 16,
            "lora_rank": 16,
            "load_in_4bit": False,
            "note": "12GB VRAM configuration"
        }
    elif vram_gb >= 10:  # 10.8GB専用設定
        return {
            "model": "unsloth/gemma-2b",
            "max_seq_length": 1024,
            "batch_size": 1,
            "gradient_accumulation": 32,
            "lora_rank": 8,
            "load_in_4bit": True,
            "alternative_model": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
            "alternative_max_seq": 1024,
            "note": "10.8GB VRAM optimized - gemma-2b recommended, mistral-7b possible with seq_length=1024"
        }
    elif vram_gb >= 8:
        return {
            "model": "unsloth/gemma-2b",
            "max_seq_length": 1024,
            "batch_size": 1,
            "gradient_accumulation": 32,
            "lora_rank": 8,
            "load_in_4bit": True,
            "note": "8GB VRAM configuration"
        }
    else:
        return {
            "model": "unsloth/tinyllama-chat",
            "max_seq_length": 512,
            "batch_size": 1,
            "gradient_accumulation": 64,
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
    print("10.8GB VRAM Optimized Edition\n")
    
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("❌ No GPU detected!")
        exit(1)
    
    vram_gb = gpu_info['total_memory_gb']
    
    print(f"GPU: {gpu_info['name']}")
    print(f"Total VRAM: {vram_gb:.2f} GB")
    print(f"Compute Capability: {gpu_info['capability']}")
    
    # 10.8GB検出
    is_10gb = 10 <= vram_gb <= 11
    if is_10gb:
        print("✅ ~10.8GB VRAM detected!")
    
    if 'memory_free_mb' in gpu_info:
        print(f"Free VRAM: {gpu_info['memory_free_mb']/1024:.2f} GB")
        print(f"GPU Utilization: {gpu_info['utilization']}%")
    
    print("\n=== Recommended Settings ===")
    settings = recommend_settings(vram_gb)
    
    for key, value in settings.items():
        if key == "note":
            print(f"\nNote: {value}")
        elif key.startswith("alternative"):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")
    
    # 10.8GB特有の警告
    if is_10gb:
        print("\n⚠️ 10.8GB VRAM Important Notes:")
        print("  • RECOMMENDED: Use gemma-2b model (safest, stable)")
        print("  • ALTERNATIVE: mistral-7b-4bit with max_seq_length=1024 (tight on memory)")
        print("  • DO NOT use max_seq_length > 1024 with mistral-7b")
        print("  • DO NOT use max_seq_length > 1536 with gemma-2b")
        print("  • Use batch_size=1 always")
        print("  • BF16 is NOT supported (use FP16)")
        print("  • Close all other GPU applications before training")
        print("  • Monitor memory with: watch -n 1 nvidia-smi")
    
    # 設定を保存
    save_config(settings, "/workspace/training_config.json")
    
    print("\n=== Quick Start Commands for 10.8GB VRAM ===")
    print("\n1. Setup model (SAFEST - Gemma-2B):")
    print("   python3 scripts/setup_model.py --model-type gemma-2b")
    
    print("\n2. Run finetuning with Gemma-2B:")
    print("   python3 scripts/finetune.py \\")
    print(f"     --model {settings['model']} \\")
    print("     --data /workspace/data/sample_finetune.jsonl \\")
    print(f"     --max-seq-length {settings['max_seq_length']} \\")
    print(f"     --batch-size {settings['batch_size']} \\")
    print("     --epochs 3")
    
    if 'alternative_model' in settings:
        print("\n3. Alternative: Mistral-7B (more powerful, tighter memory):")
        print("   python3 scripts/finetune.py \\")
        print(f"     --model {settings['alternative_model']} \\")
        print("     --data /workspace/data/sample_finetune.jsonl \\")
        print(f"     --max-seq-length {settings['alternative_max_seq']} \\")
        print("     --batch-size 1 \\")
        print("     --epochs 3")
    
    print("\n=== Memory Saving Tips ===")
    print("  • Start with gemma-2b for guaranteed stability")
    print("  • Test with small datasets first")
    print("  • Monitor GPU: watch -n 1 nvidia-smi")
    print("  • If OOM with mistral-7b, use max-seq-length 512")
    print("  • Close browser and other applications")
    
    # メモリ予測
    print("\n=== Estimated Memory Usage ===")
    if is_10gb:
        print("Gemma-2B:")
        print(f"  Model loading: ~3-4 GB")
        print(f"  Training overhead: ~2-3 GB")
        print(f"  Available buffer: ~4-5 GB (SAFE)")
        print("\nMistral-7B (4bit):")
        print(f"  Model loading: ~5-6 GB")
        print(f"  Training overhead: ~3-4 GB")
        print(f"  Available buffer: ~1-2 GB (TIGHT)")
        print(f"\n⚠️ Recommended: Use Gemma-2B for 10.8GB VRAM")