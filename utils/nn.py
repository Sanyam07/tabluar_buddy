import random

import torch
from torch import nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torchvision.transforms.functional as FT

#==============================================================================
# 常用功能
#==============================================================================
def init_conv2d(m):
    """Init parameters of convolution layer(初始化卷积层参数)

       Parameters
       ----------
       m: pytorch model

       Example
       -------
       class model(nn.Module):

           def __init__(self):
               super().__init__()

               # 初始化卷积层权重
               init_conv2d(self)
    """
    # 遍历网络子节点
    for c in m.modules():
        # 初始化卷积层
        if isinstance(c, nn.Conv2d):
            nn.init.xavier_uniform_(c.weight)
            nn.init.kaiming_normal_(c.weight, mode='fan_out', nonlinearity='relu')
            if c.bias is not None:
                nn.init.constant_(c.bias, 0.)
        # 初始BatchNorm层
        elif isinstance(c, nn.BatchNorm2d):
            nn.init.constant_(c.weight, 1.)
            nn.init.constant_(c.bias, 0.)
        # 初始线性层
        elif isinstance(c, nn.Linear):
            nn.init.normal_(c.weight, 0., 0.01)
            nn.init.constant_(c.bias, 0.)

def clip_gradient(optimizer, grad_clip):
    """Clip gradients computed during backpropagation to prevent gradient explosion.

     Parameters
     ----------
     optimizer : pytorch optimizer
     	Optimized with the gradients to be clipped.
     grad_clip: double
     	Gradient clip value.

     Examples
     --------
     from torch.optim import Adam
     from torchvision import models

     model = models.AlexNet()
     optimizer = Adam(model.parameters())
     clip_gradient(optimizer, 5)
     """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(optimizer, scale_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rates must be decayed
    :param scale_factor: factor to scale by
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def get_learning_rate(optimizer):
    """
    Get learning rate.

    :param optimizer: optimizer whose learning rates must be decayed
    """
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def decimate(tensor, dims):
    """将tensor的维度变为dims
       Parameters
       ----------
       tensor: pytorch tensor
       dims: list

       Example
       -------
       x = torch.rand(4096, 512, 7, 7)
       decimate(x, [1024, None, 3, 3])
    """
    assert tensor.dim() == len(dims)

    for i in range(len(dims)):
        if dims[i] is not None:
            tensor = tensor.index_select(dim=i, index=torch.linspace(0, tensor.size()[i]-1, dims[i]).long())

    return tensor



#==============================================================================
# 图像相关
#==============================================================================
def xyccwd_to_xymmmm(cxcy):
    """
        Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

        :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
        :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
        """
    return torch.cat([cxcy[:,:2] - cxcy[:,2:]/2, cxcy[:,:2] + cxcy[:,2:]/2], dim=1)

def xymmmm_to_xyccwd(xy):
    """
        Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

        :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
        :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
        """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h

def xyccwd_to_xygcgcgwgh(cxcy, priors_cxcy):
    """
        Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

        For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
        For the size coordinates, scale by the size of the prior box, and convert to the log-space.

        In the model, we are predicting bounding box coordinates in this encoded form.

        :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
        :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
        :return: encoded bounding boxes, a tensor of size (n_priors, 4)
        """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h

def xygcgcgwgh_to_xyccwd(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h

def find_intersection(set_1, set_2):
    """
        Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

        :param set_1: set 1, a tensor of dimensions (n1, 4)
        :param set_2: set 2, a tensor of dimensions (n2, 4)
        :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
        """
    lower_left = torch.max(set_1[:,:2].unsqueeze(1), set_2[:,:2].unsqueeze(0)) # (n1, n2, 4)
    upper_right = torch.min(set_1[:,2:].unsqueeze(1), set_2[:,2:].unsqueeze(0)) # (n1, n2, 4)
    dims_intersection = torch.clamp(upper_right - lower_left, min=0) # (n1, n2, 2)
    return dims_intersection[:,:,0] * dims_intersection[:,:,1] # (n1, n2)

def find_jaccard_overlap(set_1, set_2):
    """
        Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

        :param set_1: set 1, a tensor of dimensions (n1, 4)
        :param set_2: set 2, a tensor of dimensions (n2, 4)
        :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
        """
    areas_intersection = find_intersection(set_1, set_2) # (n1, n2)

    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    areas_union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - areas_intersection # (n1, n2)

    return areas_intersection / areas_union # (n1, n2)

# Some augmentation functions below have been adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.

    Helps to learn to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes

def random_crop(image, boxes, labels, difficulties):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.

    Note that some objects may be cut out entirely.

    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels, difficulties

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties

def flip(image, boxes):
    """
    Flip image horizontally.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes

def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).

    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes

def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image

def transform(image, boxes, labels, difficulties, split):
    """
    Apply the transformations above.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    assert split in {'TRAIN', 'TEST'}

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Skip the following operations if validation/evaluation
    if split == 'TRAIN':
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        new_image = photometric_distort(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        if 0:
         # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
            # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
            if random.random() < 0.5:
                new_image, new_boxes = expand(new_image, boxes, filler=mean)

            # Randomly crop image (zoom in)
            new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                             new_difficulties)

        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)

        # Flip image with a 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties

def show_image(path : str):
    """展示图片，输入为图片路径
       Parameters
       ----------
       path: string
           image path

       Example
       -------
       img_path = "../data/raw/gamble/pictures/_更新页面_xdqp__仙豆棋牌__0.png"
       image = show_image(img_path)
    """
    image = Image.open(path, mode='r')
    image = image.convert('RGB')
    return image

def draw_rect(image : Image.Image, location_box : list, color : str):
    """画方框
       Parameters
       ----------
       image: string
           image path
       location_box: list
           [x_min, y_min, x_max, y_max]

       Example
       -------
       img_path = "../data/raw/gamble/pictures/_更新页面_xdqp__仙豆棋牌__0.png"
       image = show_image(img_path)
       location_box = [50, 50, 100, 100]
       image = draw_rect(image, location_box, 'red')
    """
    draw = ImageDraw.Draw(image)
    draw.rectangle(xy = location_box, outline = color)
    draw.rectangle(xy = [l+1 for l in location_box], outline = color)
    return image

def draw_text(image : Image.Image, xy : list, color : str, text : str):
    """画文字
       Parameters
       ----------
       image: string
           image path

       Example
       -------
       img_path = "../data/raw/gamble/pictures/_更新页面_xdqp__仙豆棋牌__0.png"
       image = show_image(img_path)
       location_box = [50, 50, 100, 100]
       image = draw_rect(image, location_box, 'red')
    """
    font = ImageFont.truetype("../fonts/calibri/Calibri.ttf", 15)
    draw = ImageDraw.Draw(image)
    text_size = font.getsize(text)
    location_text = [xy[0] + 2., xy[1] - text_size[1]]
    location_textbox = [xy[0], xy[1] - text_size[1], xy[0] + text_size[0] + 4., xy[1]]
    draw.rectangle(xy = location_textbox, fill = color)
    draw.text(xy = location_text, text = text, fill='white', font=font)
    return image

#==============================================================================
# 自然语言相关
#==============================================================================
def word_idx(sentences):
    '''
        sentences should be a 2-d array like [['a', 'b'], ['c', 'd']]
    '''
    word_2_idx = {}
    for sentence in sentences:
        for word in sentence:
            if word not in word_2_idx:
                word_2_idx[word] = len(word_2_idx)

    idx_2_word = dict(zip(word_2_idx.values(), word_2_idx.keys()))
    num_unique_words = len(word_2_idx)
    return word_2_idx, idx_2_word, num_unique_words

def unicode_to_ascii(s):
    '''
        Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
        Example: print(unicode_to_ascii('Ślusàrski'))
    '''

    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
        )
