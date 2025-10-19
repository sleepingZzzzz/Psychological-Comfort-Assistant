import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import os

# --- 1. 配置模型路径 ---
base_model_path = "Qwen/Qwen1.5-7B-Chat" 

# --- 2. 加载模型和分词器 ---

print("... 正在加载原始基础模型和分词器 ...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# 🔴 关键改动 1: 不再将 pad_token 设置为 eos_token，以消除歧义
# tokenizer.pad_token = tokenizer.eos_token  <--- 删除或注释掉此行

# 🔴 关键改动 2: 对于生成任务，明确指定填充在左侧。这是 decoder-only 模型的最佳实践
tokenizer.padding_side = 'left'

model.eval()

print("✅ 原始模型准备就绪！开始对话吧 (输入 'exit' 或 'quit' 退出)。\n")

# --- 3. 开始交互式对话 ---

streamer = TextStreamer(tokenizer, skip_prompt=True)
history = []

while True:
    try:
        user_input = input("You (Base Model): ")
        
        if user_input.lower() in ["exit", "quit"]:
            break

        history.append({"role": "user", "content": user_input})
        
        messages = tokenizer.apply_chat_template(
            history, 
            tokenize=False, 
            add_generation_prompt=True
        )

        model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")

        print("Assistant (Base Model): ", end="")
        generated_ids = model.generate(
            **model_inputs,
            # 🔴 关键改动 3: 明确告诉 generate 函数，哪个是结束标记
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            streamer=streamer
        )

        full_response_ids = generated_ids[0][model_inputs.input_ids.shape[-1]:]
        response = tokenizer.decode(full_response_ids, skip_special_tokens=True)
        history.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        break

print("\n👋 对话结束。")