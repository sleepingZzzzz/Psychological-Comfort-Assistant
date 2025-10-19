from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse

# 🔴 已将 load_model_and_tokenizer 函数还原
def load_model_and_tokenizer(model_name, quantization_config):
    """加载模型和分词器 (从Hugging Face Hub), 应用 4-bit 量化。"""
    print("... 正在加载模型和分词器 ...")
    
    bnb_config = BitsAndBytesConfig(**quantization_config)

    # local_files_only=True 已被移除
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # local_files_only=True 已被移除
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ 模型和分词器加载完成")
    return model, tokenizer

def prepare_dataset(dataset_name, tokenizer):
    """
    加载并预处理新的对话数据集。
    该函数直接利用 'extra_info' 列中的 'full_conversation' 数据。
    """
    print("... 正在准备新的对话数据集 ...")
    
    # 假设数据集只有一个 'train' 分割
    dataset = load_dataset(dataset_name, split="train")

    def preprocess_function(examples):
        """
        预处理函数：直接应用聊天模板到 'full_conversation' 列。
        """
        # 1. 从 'extra_info' 列中提取 'full_conversation' 列表
        # 'examples['extra_info']' 是一个列表，其中每个元素都是一个字典
        all_conversations = [info['full_conversation'] for info in examples['extra_info']]
        
        # 2. 对每个对话列表应用聊天模板
        # 因为数据格式已经完美匹配，我们可以直接将其传递给 apply_chat_template
        processed_conversations = [
            tokenizer.apply_chat_template(
                conv, 
                tokenize=False, 
                add_generation_prompt=False
            ) 
            for conv in all_conversations
        ]

        # 3. 对格式化后的文本进行 tokenize
        tokenized_inputs = tokenizer(
            processed_conversations,
            max_length=512, # 数值越大，显存占用越大，训练效果越好
            truncation=True,
            padding=False,
        )
        return tokenized_inputs

    processed_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    
    print("✅ 数据集准备完成")
    return processed_dataset

def setup_trainer(model, tokenizer, dataset, lora_config_dict, training_config_dict):
    """设置 PEFT/LoRA 并配置训练器。"""
    print("... 正在设置训练器 ...")
    
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(**lora_config_dict)
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()

    training_args = TrainingArguments(**training_config_dict)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    model.config.use_cache = False
    
    print("✅ 训练器设置完成")
    return trainer

# --- 🔴 定义命令行参数解析 (已扩展) ---
def parse_args():
    parser = argparse.ArgumentParser(description="微调语言模型的脚本")
    
    # 模型与数据
    parser.add_argument("--model_name", type=str, help="要微调的基础模型名称或路径")
    parser.add_argument("--dataset_name", type=str, help="用于微调的数据集名称或路径")
    
    # LoRA / DoRA 参数
    parser.add_argument("--lora_r", type=int, help="LoRA/DoRA 的秩 r")
    parser.add_argument("--lora_alpha", type=int, help="LoRA/DoRA 的 alpha 值")
    parser.add_argument("--lora_dropout", type=float, help="LoRA 层的 dropout 率")
    # 🔴 新增 use_dora 布尔开关
    # action=argparse.BooleanOptionalAction 会自动创建 --use_dora 和 --no-use_dora 两个 flag
    parser.add_argument("--use_dora", action=argparse.BooleanOptionalAction, help="是否使用 DoRA 技术")

    # 训练参数
    parser.add_argument("--output_dir", type=str, help="训练输出的目录")
    parser.add_argument("--num_train_epochs", type=int, help="训练的总轮次")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--logging_steps", type=int, help="每隔多少步记录一次 log")
    parser.add_argument("--per_device_train_batch_size", type=int, help="每个设备的 batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="梯度累积步数")
    
    args = parser.parse_args()
    return args

# --- 🔴 更新默认配置的函数 (已扩展) ---
def update_config(config, args):
    # 使用 vars(args) 将 argparse.Namespace 对象转换为字典，方便遍历
    args_dict = vars(args)
    
    # 检查并更新提供的参数
    if args_dict.get("model_name") is not None:
        config['model_name'] = args_dict["model_name"]
    if args_dict.get("dataset_name") is not None:
        config['dataset_name'] = args_dict["dataset_name"]
        
    # 更新 lora 参数
    if args_dict.get("lora_r") is not None:
        config['lora']['r'] = args_dict["lora_r"]
    if args_dict.get("lora_alpha") is not None:
        config['lora']['lora_alpha'] = args_dict["lora_alpha"]
    if args_dict.get("lora_dropout") is not None:
        config['lora']['lora_dropout'] = args_dict["lora_dropout"]
    if args_dict.get("use_dora") is not None:
        config['lora']['use_dora'] = args_dict["use_dora"]

    # 更新 training 参数
    if args_dict.get("output_dir") is not None:
        config['training']['output_dir'] = args_dict["output_dir"]
    if args_dict.get("num_train_epochs") is not None:
        config['training']['num_train_epochs'] = args_dict["num_train_epochs"]
    if args_dict.get("learning_rate") is not None:
        config['training']['learning_rate'] = args_dict["learning_rate"]
    if args_dict.get("logging_steps") is not None:
        config['training']['logging_steps'] = args_dict["logging_steps"]
    if args_dict.get("per_device_train_batch_size") is not None:
        config['training']['per_device_train_batch_size'] = args_dict["per_device_train_batch_size"]
    if args_dict.get("gradient_accumulation_steps") is not None:
        config['training']['gradient_accumulation_steps'] = args_dict["gradient_accumulation_steps"]
    
    return config
