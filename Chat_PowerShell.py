import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from peft import PeftModel
import os
# --- 1. 配置模型路径 ---
# 🔴 请根据您的实际情况修改这里的路径

# 基础模型路径 (可以是 Hugging Face Hub 的 ID，也可以是您本地下载的路径)
# 如果您之前已经将基础模型下载到本地，强烈建议使用本地路径以加快加载速度
base_model_path = "Qwen/Qwen1.5-7B-Chat"  # 假设您之前换成了 4B 模型

# 您训练好的 LoRA/DoRA 适配器所在的文件夹路径
# 这应该是您上一轮训练时 'output_dir' 指定的那个文件夹
adapter_path = "./mindchat-qwen1.5-7B-finetuned" 

# --- 2. 加载模型和分词器 ---

print("... 正在加载基础模型和分词器 ...")

# 使用 4-bit 量化加载基础模型，节省显存
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("... 正在加载并融合 LoRA/DoRA 适配器 ...")

# 加载 PEFT 适配器并将其与基础模型融合
# 这是最关键的一步，PeftModel 会自动完成“穿上神装”的操作
model = PeftModel.from_pretrained(base_model, adapter_path)

# 将模型设置为评估模式
model.eval()

print("✅ 模型准备就绪！开始对话吧 (输入 'exit' 或 'quit' 退出)。\n")

# --- 3. 开始交互式对话 ---
streamer = TextStreamer(tokenizer, skip_prompt=True)
# 存储对话历史
history = []

while True:
    try:
        user_input = input("You: ")
        # ...
        history.append({"role": "user", "content": user_input})
        messages = tokenizer.apply_chat_template(
            history, 
            tokenize=False, 
            add_generation_prompt=True
        )
        model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")

        # 🔴 3. 在 generate 函数中，传入 streamer 参数
        #    注意：我们不再需要自己解码，streamer 会自动处理打印
        print("Assistant: ", end="") # 先打印出 "Assistant: "
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            streamer=streamer # <--- 核心改动在这里
        )

        # 🔴 4. 因为 streamer 已经完成了打印，我们需要手动获取完整回答并存入历史
        #    这一步稍微复杂一点，我们需要重新解码完整的输出
        full_response_ids = generated_ids[0][model_inputs.input_ids.shape[-1]:]
        response = tokenizer.decode(full_response_ids, skip_special_tokens=True)
        history.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        break

print("\n👋 对话结束。")