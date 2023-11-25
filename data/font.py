import os
from PIL import Image, ImageDraw, ImageFont, ImageChops
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale, functional
from torch.utils.data import Dataset
from fontTools.ttLib import TTFont
from torch.utils.data import ConcatDataset


## the valid chars is based on the image part. I thinks it's better to only feed chars that exist in the image dataset 
# because it contains more samples.
with open('./data/chars.txt', 'r', encoding='utf8') as f:
    valid_chars = f.read()
    valid_chars = valid_chars.split('\n')


def list_supported_chars(font_path):
    font = TTFont(font_path)
    supported_chars = set()

    for table in font['cmap'].tables:
        for char_code in table.cmap.keys():
            # if 0x4E00 < char_code < 0x9FFF+1:
            #     supported_chars.add(chr(char_code))
            if chr(char_code) in valid_chars:
                supported_chars.add(chr(char_code))

    return list(supported_chars)


def get_union_charset(font_paths):
    charsets = [list_supported_chars(font_path) for font_path in font_paths]
    # find the union
    union = set()
    for charset in charsets:
        union = union.union(charset)
    return union


def union_dataset(font_paths, image_size=64):
    datasets = [ChineseCharacterDatasetFromFont(font_path, font_size=image_size) for font_path in font_paths]
    datasets.append(ChineseCharacterDatasetFromImage('images/kaishu', font_size=image_size))
    n_classes = len(valid_chars)
    datasets = ConcatDataset(datasets)
    return datasets, n_classes


class Font:
    def __init__(self, font_path, font_size=64):
        self.font = ImageFont.truetype(font_path, font_size)
        self.font_size = font_size

    def draw_text(self, character, color=255):
        image_size = self.font_size
        image = Image.new("L", (image_size, image_size), color=0)
        draw = ImageDraw.Draw(image)
        x1, y1, x2, y2 = self.font.getbbox(character)
        width = x2 - x1
        height = y2 - y1
        x = (image_size - width) / 2 - x1
        y = (image_size - height) / 2 - y1
        draw.text((x, y), character, font=self.font, fill=color)
        return image
    

class ChineseCharacterDatasetFromFont(Dataset):
    def __init__(self, font_path, font_size=64):
        self.font = Font(font_path, font_size)
        self.supported_chars = list_supported_chars(font_path)  # Unicode for surpported Chinese characters
        self.transform = ToTensor()

    def __len__(self):
        return len(self.supported_chars)

    def __getitem__(self, idx):
        character = self.supported_chars[idx]
        image = self.font.draw_text(character)  
        image = self.transform(image)
        label = valid_chars.index(character)
        return image, label
    


class ChineseCharacterDatasetFromImage(Dataset):
    def __init__(self, image_root, font_size=64):
        with open(os.path.join(image_root, 'labels.txt'), 'r', encoding='utf8') as f:
            self.labels = f.read()
            self.labels = self.labels.split('\n') # list of labels 
        # list all the images under the root
        self.n_images = len(os.listdir(image_root)) - 1  # exclude the labels.txt
        self.font_size = font_size
        self.transform = Compose([
            Grayscale(), 
            ToTensor(), 
            Resize((font_size, font_size)),
            functional.invert
        ])
        self.image_root = image_root

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_root, str(idx)+'.png'))
        image = self.transform(image)
        character = self.labels[idx]
        label = valid_chars.index(character)
        return image, label
    
