import torch
from datasets import load_dataset
from Tools import load_model_and_tokenizer, prepare_dataset, setup_trainer, update_config, parse_args
import argparse
import sys

# --- 默认配置 ---
config = {
    "model_name": "Qwen/Qwen1.5-7B-Chat",
    "dataset_name": "dongSHE/MindChat-R0-FT-Data",
    "lora": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "use_dora": True,
        "task_type": "CAUSAL_LM",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    },
    "training": {
        "output_dir": "./mindchat-qwen1.5-7b-finetuned",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "logging_steps": 10,
        "optim": "paged_adamw_8bit",
        "bf16": True,
        "gradient_checkpointing": True,
    },
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
    }
}

# --- 主执行流程 ---
def main():
    """主函数，串联整个微调流程。"""
    
    args = parse_args()
    final_config = update_config(config, args)

    print("🚀 开始执行 MindChat 微调流程 🚀")
    print(" final_config:",final_config)
    
    model, tokenizer = load_model_and_tokenizer(
        final_config["model_name"], 
        final_config["quantization"]
    )
    
    dataset = prepare_dataset(
        final_config["dataset_name"], 
        tokenizer
    )
    
    trainer = setup_trainer(
        model, 
        tokenizer, 
        dataset, 
        final_config["lora"], 
        final_config["training"]
    )

    print("\n🔥 开始模型微调... 🔥")
    trainer.train()
    print("🎉 模型微调完成！")

    final_output_dir = final_config["training"]["output_dir"]
    print(f"... 正在将训练好的 Adapter 保存到 {final_output_dir} ...")
    trainer.save_model(final_output_dir)
    print("✅ Adapter 保存成功！")

if __name__ == "__main__":
    main()