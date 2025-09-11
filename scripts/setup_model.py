"""
Ollamaからモデルをダウンロードして、Unsloth用に変換するスクリプト
"""
import os
import subprocess
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_ollama_model(model_name="llama2:7b"):
    """Ollamaを使用してモデルをダウンロード"""
    print(f"Downloading {model_name} using Ollama...")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"Successfully downloaded {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}")
        return False

def export_ollama_model(model_name, output_dir="/workspace/models/exported"):
    """Ollamaモデルをエクスポート"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Ollamaモデルの情報を取得
    try:
        result = subprocess.run(
            ["ollama", "show", model_name, "--modelfile"],
            capture_output=True, text=True, check=True
        )
        
        modelfile_path = os.path.join(output_dir, "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(result.stdout)
        
        print(f"Model information exported to {modelfile_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error exporting model: {e}")
        return False

def prepare_unsloth_model(model_name="llama2:7b"):
    """Unsloth用にモデルを準備"""
    from unsloth import FastLanguageModel
    
    # Unslothで直接使用可能なモデル名にマッピング
    model_mapping = {
        "llama2:7b": "unsloth/llama-2-7b",
        "mistral:7b": "unsloth/mistral-7b",
        "phi:latest": "microsoft/phi-2",
    }
    
    unsloth_model_name = model_mapping.get(model_name, model_name)
    
    print(f"Loading model for Unsloth: {unsloth_model_name}")
    
    # モデルとトークナイザーの読み込み
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=unsloth_model_name,
        max_seq_length=2048,
        dtype=None,  # 自動検出
        load_in_4bit=True,  # 4bit量子化
    )
    
    # LoRA設定の追加
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup model using Ollama")
    parser.add_argument("--model", type=str, default="llama2:7b",
                       help="Model name to download from Ollama")
    parser.add_argument("--export-dir", type=str, 
                       default="/workspace/models/exported",
                       help="Directory to export model")
    
    args = parser.parse_args()
    
    # Ollamaからモデルをダウンロード
    if download_ollama_model(args.model):
        # モデルをエクスポート
        export_ollama_model(args.model, args.export_dir)
        
        # Unsloth用に準備
        try:
            model, tokenizer = prepare_unsloth_model(args.model)
            print("Model successfully prepared for Unsloth training!")
        except Exception as e:
            print(f"Error preparing model for Unsloth: {e}")