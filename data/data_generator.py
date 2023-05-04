import os
import torch
from torchvision.io.image import write_jpeg
import pandas as pd
# from doctr.datasets.generator import CharacterGenerator



# 1. make a vocab
vocab = "कखग‍ङचछजझ‍ञ‌‌‍टठडढणतथदधनपफबभमयरलवशषसह"

# fonts_family = ["sahadeva.ttf", "Gargi.ttf", "Samyak-Devanagari.ttf"]
# ds = CharacterGenerator(vocab=vocab, num_samples=50)

# max_num_imgs = 1500

# imgs_list, labels_list = [], []
# root_dir_imgs = "data/imgs"
# if not os.path.isdir(root_dir_imgs):
#     os.mkdir(root_dir_imgs)

# for i in range(max_num_imgs):
#     img, label = ds[i]
#     img_name = f"img_{label}_{i}.jpg"
#     imgs_list.append(img_name)
#     img_name = os.path.join(root_dir_imgs, img_name)
#     labels_list.append(label)
#     # print(img.shape)
#     write_jpeg(img.view(dtype=torch.uint8), img_name)

# csv_name = os.path.join("data", f"imgs_{max_num_imgs}.csv")    
# pd.DataFrame({"filename": imgs_list, "labels": labels_list}).to_csv(csv_name, index=False)

    
# 3. make an image with different fonts and transformations
# 4. save the image, save the label 
