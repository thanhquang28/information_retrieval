# !pip install transformers
# !pip install torch torchvision
# !pip install Pillow
# !pip install pandas

import pandas as pd
import requests
from PIL import Image

df = pd.read_csv("image_urls.csv")
image_urls = df["url"].dropna().tolist()

# Lấy tối đa 20 ảnh đầu để tránh hết RAM
max_images = 20
images = []
valid_urls = []
for i, url in enumerate(image_urls):
    if len(images) >= max_images:
        break
    try:
        response = requests.get(url, stream=True, timeout=10)
        img = Image.open(response.raw).convert('RGB')
        img = img.resize((224, 224))   # Resize nhỏ để tiết kiệm RAM
        images.append(img)
        valid_urls.append(url)
        print(f"OK {i+1}: {url}")
    except Exception as e:
        print(f"Error {i+1}: {url} - {e}")

print(f"Tổng số ảnh tải thành công: {len(images)}")

from transformers import CLIPProcessor, CLIPModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

from google.colab import files
uploaded = files.upload()
import io

ref_image = None
for filename in uploaded.keys():
    ref_image = Image.open(io.BytesIO(uploaded[filename])).convert('RGB')
    ref_image = ref_image.resize((224, 224))  # Resize reference image cho đồng bộ
    ref_image.show()
    break

feedback = input("Feedback tiếng Anh mô tả thay đổi (ví dụ: with long sleeves, brighter color): ")

ref_inputs = processor(images=ref_image, return_tensors="pt").to(device)
ref_feat = model.get_image_features(**ref_inputs)
ref_feat /= ref_feat.norm(dim=-1, keepdim=True)

text_inputs = processor(text=[feedback], return_tensors="pt").to(device)
text_feat = model.get_text_features(**text_inputs)
text_feat /= text_feat.norm(dim=-1, keepdim=True)

query_feat = ref_feat + text_feat

cand_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
cand_feats = model.get_image_features(**cand_inputs)
cand_feats /= cand_feats.norm(dim=-1, keepdim=True)

sim = (query_feat @ cand_feats.T).squeeze()
import matplotlib.pyplot as plt
top_k = 3
top_idx = sim.topk(top_k).indices.cpu().numpy()
for rank, idx in enumerate(top_idx):
    plt.figure()
    plt.imshow(images[idx])
    plt.axis('off')
    plt.title(f"Kết quả top {rank+1}")
    plt.show()