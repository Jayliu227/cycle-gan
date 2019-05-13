from detect import detect
from PIL import Image, ImageFont

img_path = '2008_005266.jpg'
original_image = Image.open(img_path)
original_image = original_image.convert('RGB')
font = ImageFont.load_default()

detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200)