from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from utils import *

import sys
import warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.append('./object_detection')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = './object_detection/BEST_checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
best_loss = checkpoint['best_loss']

print('Finished loading detection model.')

model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

def get_mask(tensor, min_score=0.2, max_overlap=0.5, top_k=200, suppress=None):

    original_image = convert_to_PIL(tensor)
    
    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    if det_labels == ['background']:
        return convert_to_mask(original_image.height, original_image.width, None)
    else:
        return convert_to_mask(original_image.height, original_image.width, largest_box(det_boxes))

def convert_to_PIL(tensor):    
    '''
    input: 
        tensor (1, 3, H, W) ~ (0, 1) pytorch tensor
    output:
        PIL image
    '''
    tensor = tensor.view(3, tensor.shape[2], tensor.shape[3])
    image = transforms.ToPILImage()(tensor)
    return image

def largest_box(boxes):
    largest = None
    area = 0
    for box in boxes:
        cur_area = (box[2] - box[0]) * (box[3] - box[1])
        if cur_area > area:
            area = cur_area
            largest = box
    return largest

def convert_to_mask(height, width, box):
    mask = np.zeros((height, width))    
    
    if box is None or box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
        return mask

    mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1
    return mask
    
    
def shift_mask_to_upper_left(mask):
    '''
    input:
        img (2d np array)
    return:
        img (2d np array) – img shifted to the upper left
    '''
    if np.count_nonzero(mask) == 0:
        return mask
    while np.count_nonzero(mask[0,:]) == 0:
        mask = np.concatenate((mask[1:,],mask[:1,]))
    while np.count_nonzero(mask[:,0]) == 0:
        mask = np.concatenate((mask[:,1:],mask[:,:1]), axis=1)
    return mask


def shape_sim(mask_a, mask_b):
    '''
    input:
        mask_a, mask_b: 2D numpy array where 1 is foreground and 0 is background
    output:
        scaler, shape similarity between (0, 1)
    '''
    mask_a = shift_mask_to_upper_left(mask_a)
    mask_b = shift_mask_to_upper_left(mask_b)
    a = np.count_nonzero(np.minimum(mask_a,mask_b))
    b = max(np.count_nonzero(mask_a), np.count_nonzero(mask_b))
    return a / max(1e-5, b)
