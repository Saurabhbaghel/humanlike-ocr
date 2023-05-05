import os
from PIL import Image
from doctr.datasets.generator import CharacterGenerator
import pandas as pd
import torch
from torchvision.io.image import write_jpeg
from tqdm import tqdm
import subprocess
import random

# # 1. make a vocab
# vocab = "कखग‍ङचछजझ‍ञ‌‌‍टठडढणतथदधनपफबभमयरलवशषसह"

fonts_family = ["sahadeva.ttf", "Gargi.ttf", "Samyak-Devanagari.ttf"]
# # ds = CharacterGenerator(vocab=vocab, num_samples=10, font_family=fonts_family)

# # idx = 0
# # jdx = -1
# # new_label = ""
# # root_dir = "data/atoms"
# # data = {"filename":[], "label": []}

# # if not os.path.isdir(root_dir):
# #     os.mkdir(root_dir)

# # for image, label in ds:
# #     if idx == 3072:
# #         break
# #     filename = f"image_{label}_{idx}.jpg"
# #     write_jpeg(image.view(torch.uint8), os.path.join(root_dir, filename))
# #     idx += 1
# #     data["filename"].append(filename)
# #     data["label"].append(label)
    
# atoms_dir = "/media/ashatya/Data/work/self/thesis/humanlike-ocr/data/atomic_characters"
# out_dir = "/media/ashatya/Data/work/self/thesis/humanlike-ocr/data/atoms"

# # process = f"text2image --text {atoms_txt} --outputbase {out_dir} --find_fonts --fonts_dir /usr/share/fonts"

# # iterate over each char file
# for char_text_file in os.listdir(atoms_dir):
#     atoms_txt = os.path.join(atoms_dir, char_text_file)
#     out_dir_char = os.path.join(out_dir, char_text_file[:-4])
#     if not os.path.isdir(out_dir_char):
#         os.makedirs(out_dir_char)
#     subprocess.run(["text2image", "--text", f"{atoms_txt}", "--outputbase", f"{out_dir_char}", "--fonts_dir", "/usr/share/fonts", "--font", "Lohit Devanagari", "--degrade_image", "--max_pages", "128", "--xsize", "20", "--ysize", "20"])

   
# # pd.DataFrame(data).to_csv(os.path.join(root_dir, f"annotations_{idx}_.csv"), index=False)
    
# # 3. make an image with different fonts and transformations
# # 4. save the image, save the label 



from PIL import Image, ImageDraw, ImageFont

# Specify the parameters for image generation
image_size = (32, 32)  # Size of each image in pixels
background_color = (255, 255, 255)  # RGB color code for the background
text_color = (0, 0, 0)  # RGB color code for the text
font_dir = "/usr/share/fonts"  # Path to the font file
font_size = 24  # Size of the font

# List of Devanagari alphabet characters
devanagari_alphabets = [
    "अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ए", "ऐ", "ओ", "औ",
    "क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट",
    "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "प", "फ",
    "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह"
]

num_of_images = 100
out_dir = "data/atoms_4300"

data = {"filename": [], "label": []}

def font_generator():
    fonts_family = ["sahadeva.ttf", "Gargi.ttf", "Samyak-Devanagari.ttf"]
    return random.sample(fonts_family,k=1)

# Create images for each Devanagari alphabet
for idx, alphabet in enumerate(devanagari_alphabets):
    label = idx
    for n in range(num_of_images):
        
        # Create a new image with white background
        image = Image.new("RGB", image_size, background_color)

        # Load the font
        font_path = os.path.join(font_dir, font_generator()[0])
        font = ImageFont.truetype(font_path, font_size)

        # Calculate the text position to center it in the image
        text_width, text_height = font.getsize(alphabet)
        text_position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

        # Draw the text on the image
        draw = ImageDraw.Draw(image)
        draw.text(text_position, alphabet, font=font, fill=text_color)

        # Save the image
        basename = f"{label}_{n}_.png"
        image.save(os.path.join(out_dir, basename))

        # write the csv
        data["filename"].append(basename)
        data["label"].append(label)
        
atoms_4300_df = pd.DataFrame(data)

# saving the df
atoms_4300_df.to_csv("atoms_4300.csv", index=False)
    # Optionally, you can also show the image
    # image.show()
