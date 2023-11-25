from PIL import Image, ImageDraw, ImageFont

# Load your image and font
font = ImageFont.truetype('font/HuangTingJianShuFaZiTi-2.ttf', size=100)
# Your text
char = "你"
print(font.getbbox(char))

# Measure character size
width, height = font.getsize(char)
image_size = 100

image = Image.new('RGB', (image_size, image_size), color='black')

# Draw the character
draw = ImageDraw.Draw(image)
x1, y1, x2, y2 = font.getbbox(char)
width = x2 - x1
height = y2 - y1
x = (image_size - width) / 2 - x1
y = (image_size - height) / 2 - y1
print(x, y)
draw.text((x, y), char, font=font, fill='white')
draw.rectangle(draw.textbbox((x, y), char, font=font))
print(draw.textbbox((x, y), char, font=font))

# Save or display the image
image.save(f'output.png')  # Saves the image as 'A.png', 'B.png', etc.