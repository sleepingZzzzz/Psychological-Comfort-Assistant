import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
import gradio as gr
from threading import Thread

# --- 全局变量 ---
model = None
tokenizer = None

# --- 1. 模型加载函数 (最终修正版) ---
def load_model(base_model_path, adapter_path):
    global model, tokenizer
    if not base_model_path:
        return "错误：基础模型路径不能为空。"

    print(f"... 正在从本地路径 '{base_model_path}' 加载基础模型 ...")
    
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # 🔴 关键点 1: 为加载基础模型强制使用本地文件
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True  # <--- 确保这里有
        )

        # 🔴 关键点 2: 为加载分词器强制使用本地文件
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            local_files_only=True  # <--- 确保这里有
        )
        tokenizer.padding_side = 'left'

        # 如果用户提供了适配器路径，就加载它
        if adapter_path:
            print(f"... 正在从本地路径 '{adapter_path}' 加载并融合 LoRA/DoRA 适配器 ...")
            # 🔴 关键点 3: 为加载适配器也强制使用本地文件
            model = PeftModel.from_pretrained(
                base_model, 
                adapter_path,
                local_files_only=True # <--- 确保这里也有
            )
        else:
            print("... 未提供适配器路径，将直接使用基础模型 ...")
            model = base_model
        
        model.eval()

        print("✅ 模型加载成功！")
        return "模型加载成功！您现在可以开始对话了。"
    except Exception as e:
        print(f"模型加载失败: {e}")
        return f"模型加载失败: {str(e)}"

# --- 2. 核心对话/生成函数 (无需改动) ---
def chat(message, history):
    if not history:
            history.append({
                "role": "system",
                "content": "你是一个体贴、温柔的心理咨询助手。你的所有回答都应该简洁且温暖, 在每段回答的结尾都加上一段鼓励的话并加上几个表情, 严格限制每次回答的字数不超过200个字。"
            })
            
    history.append({"role": "user", "content": message})
    if model is None or tokenizer is None:
        history.append({"role": "assistant", "content": "错误：请先点击“加载模型”按钮，并等待加载成功。"})
        yield history
        return

    conversation = history
    messages = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(**model_inputs, streamer=streamer, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95, repetition_penalty=1.1, eos_token_id=tokenizer.eos_token_id)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    history.append({"role": "assistant", "content": ""})
    for new_text in streamer:
            # 在将文本添加到历史之前，检查是否包含了结束标记
            if tokenizer.eos_token in new_text:
                # 如果包含，可能意味着一个不完整的标记，我们最好在这里直接停止
                break
            history[-1]["content"] += new_text
            yield history

# --- 3. Gradio 界面布局 (无需改动) ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧠 MindChat 微调模型测试界面")
    with gr.Row():
        with gr.Column(scale=1):
            base_model_input = gr.Textbox(label="基础模型本地路径", value="./PreModel/models--Qwen--Qwen1.5-7B-Chat/snapshots/5f4f5e69ac7f1d508f8369e977de208b4803444b")
            adapter_path_input = gr.Textbox(label="适配器 (Adapter) 本地路径", value="./mindchat-qwen1.5-7b-finetuned")
            load_button = gr.Button("加载模型", variant="primary")
            status_text = gr.Textbox(label="状态", interactive=False)
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话窗口", height=500, type="messages")
            msg_input = gr.Textbox(label="您的问题", placeholder="请输入您的问题，然后按 Enter...")
            clear_button = gr.Button("清除对话历史")
    load_button.click(fn=load_model, inputs=[base_model_input, adapter_path_input], outputs=[status_text])
    msg_input.submit(fn=chat, inputs=[msg_input, chatbot], outputs=[chatbot])
    msg_input.submit(lambda: "", None, msg_input)
    clear_button.click(lambda: None, None, chatbot, queue=False)

# --- 5. 启动应用 ---
if __name__ == "__main__":
    app.launch()