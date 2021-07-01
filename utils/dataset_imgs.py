import os
from PIL import Image

list = os.listdir("./data/CelebA")
for i, path in enumerate(list):
    img = Image.open("./data/CelebA/" + path).convert("RGB")
    img = img.resize((64, 64))
    img.save("./data/CelebA_resize/" + path)
