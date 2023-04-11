import subprocess
import os
from generator import CharacterGenerator



# 1. make a vocab
vocab = "कखग‍ङचछजझ‍ञ‌‌‍टठडढणतथदधनपफबभमयरलवशषसह"

fonts_family = ["sahadeva.ttf", "Gargi.ttf", "Samyak-Devanagari.ttf"]
ds = CharacterGenerator(vocab=vocab, num_samples=100,)


    
# 3. make an image with different fonts and transformations
# 4. save the image, save the label 
