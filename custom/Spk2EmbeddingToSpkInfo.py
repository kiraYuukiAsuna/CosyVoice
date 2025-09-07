import torch

# 加载原始文件
spk2embedding = torch.load(
    "/home/seele/Dev/DeepLearning/CosyVoice/examples/libritts/cosyvoice2/data/elysia_train/spk2embedding.pt")

# 创建新格式
spk2info = {}
for spk_id, embedding in spk2embedding.items():
    spk2info[spk_id] = {
        'embedding': torch.tensor([embedding]).to('cuda' if torch.cuda.is_available() else 'cpu')
    }
    # 如果需要使用inference_zero_shot，还需要添加其他必要的键
    # 对于inference_sft，只需要'embedding'键就足够了

# 保存转换后的文件
torch.save(spk2info, "/home/seele/Dev/DeepLearning/CosyVoice/pretrained_models/CosyVoice2-0.5B/spk2info.pt")
