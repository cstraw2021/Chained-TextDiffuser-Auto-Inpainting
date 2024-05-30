from PIL import Image, ImageDraw, ImageFont
import easyocr
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
def display(image):
    # Display image using Matplotlib
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# Function to fit text within a bounding box
def fit_text_in_box(draw, text, box, font_path='arial.ttf'):
    p0, p1, p2, p3 = box
    box_width = max(p1[0] - p0[0], p2[0] - p3[0])
    box_height = max(p3[1] - p0[1], p2[1] - p1[1])
    
    # Start with a large font size and decrease until the text fits within the box
    font_size = 100
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = draw.textsize(text, font=font)
    
    while text_width > box_width or text_height > box_height:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = draw.textsize(text, font=font)
    
    return font
# Function to draw bounding boxes around detected text on a given image
def draw_boxes(image, bounds, color='yellow', width=2, fill_color=None, replace=None):
    draw = ImageDraw.Draw(image)
    for i in range(len(bounds)):
        bound = bounds[i]
        box = bound[0]
        
        p0, p1, p2, p3 = box
        if fill_color:
            # Fill the box with the fill color
            draw.polygon([*p0, *p1, *p2, *p3], fill=fill_color)
        
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=(0,255,0), width=width)
        
    return image

# Function to draw bounding boxes around detected text on a given image
def draw_mask(image, bounds, color='yellow', width=2, fill_color=None, replace=None):
    draw = ImageDraw.Draw(image)
    for i in range(len(bounds)):
        bound = bounds[i]
        box = bound[0]
        text = replace[i] if replace else bound[1]
        
        p0, p1, p2, p3 = box
        if fill_color:
            # Fill the box with the fill color
            draw.polygon([*p0, *p1, *p2, *p3], fill=fill_color)
        
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=(193,193,193), width=width)
        
        font = fit_text_in_box(draw, text, box)
        text_width, text_height = draw.textsize(text, font=font)
        
        text_x = p0[0] + (p1[0] - p0[0] - text_width) / 2
        text_y = p0[1] + (p3[1] - p0[1] - text_height) / 2
        draw.text((text_x, text_y), text, fill=(0,0,0), font=font)
    
    return image



# Function to perform OCR and draw bounding boxes on the image and a blank canvas
def inference(image_path, lang=['en']):
    reader = easyocr.Reader(lang)
    bounds = reader.readtext(image_path)
    print(bounds)
    
    im_with_boxes = Image.open(image_path)
    draw_boxes(im_with_boxes, bounds, (0, 255, 0), 5)
    
    # Create a blank canvas with the same size as the original image
    mask = Image.new('RGB', im_with_boxes.size, (255, 255, 255))
    draw_mask(mask, bounds, (0, 0, 0), 5, fill_color=(193,193,193))
    
    return im_with_boxes, mask

def inference_piped(image, lang=['en'], replace=None):
    
    image_arr = np.array(image)
    reader = easyocr.Reader(lang)
    bounds = reader.readtext(image_arr)
    print(bounds)
    im_with_boxes = image.copy()
    draw_boxes(im_with_boxes, bounds, (0, 255, 0), 5)
    
    # Create a blank canvas with the same size as the original image
    mask = Image.new('RGB', image.size, (255, 255, 255))
    draw_mask(mask, bounds, (0, 0, 0), 5, fill_color=(193,193,193), replace=replace)
    return im_with_boxes, mask

def stitch(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    stitched = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        stitched.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return stitched