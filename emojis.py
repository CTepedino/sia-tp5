import numpy as np
from PIL import Image

emoji_size = (20, 20)
emoji_images = []


img = np.asarray(Image.open('emojis.png').convert("L"))
emojis_per_row = img.shape[1] / emoji_size[1]

n_emojis = (img.shape[0] // emoji_size[0]) * emojis_per_row
for i in range((int)(n_emojis // 1)):
    y = int((i // emojis_per_row) * emoji_size[0])
    x = int((i % emojis_per_row) * emoji_size[1])
    emoji_matrix = img[y:(y + emoji_size[1]), x:(x + emoji_size[0])] / 255
    emoji_vector = emoji_matrix.flatten()
    emoji_images.append(emoji_vector)

