from modelscope.hub.snapshot_download import snapshot_download
import os

# --- 配置 ---
# 1. ModelScope 上的模型ID (来自您图中的信息)
# 注意：这是一个较早版本的 Qwen-7B 模型
model_id = 'qwen/Qwen-7B-Chat'

# 2. 您想把模型存放在本地的哪个文件夹
#    为了不和Hugging Face下载的模型混淆，我们创建一个新文件夹
local_dir = "D:/models/modelscope/qwen-7b-chat"

# 确保目标文件夹存在
os.makedirs(local_dir, exist_ok=True)

print(f"正在使用 ModelScope SDK 开始下载模型 {model_id} 到 {local_dir} ...")

# 使用 snapshot_download 下载整个模型仓库
# cache_dir 参数可以指定下载路径
snapshot_download(
    model_id=model_id,
    cache_dir=local_dir
)

print(f"模型下载完成！文件已保存在: {local_dir}")