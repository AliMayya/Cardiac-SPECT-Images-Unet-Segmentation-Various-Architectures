import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from PIL import Image



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
def predict_image_mask_miou(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score
def predict_image_mask_pixel(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc
def miou_score(model, test_set):
    score_iou = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
    return score_iou
def pixel_acc(model, test_set):
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, acc = predict_image_mask_pixel(model, img, mask)
        accuracy.append(acc)
    return accuracy
def predict_image_mask_dice(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        score = dice_score(output, mask)  # Compute Dice score
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score

def dice_score_model(model, test_set):
    score_dice = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_dice(model, img, mask)  # Use the new function
        score_dice.append(score)
    return score_dice
import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )
    return conv

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class UNet4(nn.Module):
    def __init__(self, num_classes):
        super(UNet4, self).__init__()

        self.backbone = models.inception_v3(pretrained=True, aux_logits=True)

        self.encoder1 = nn.Sequential(self.backbone.Conv2d_1a_3x3,
                                      self.backbone.Conv2d_2a_3x3,
                                      self.backbone.Conv2d_2b_3x3)
        self.encoder2 = nn.Sequential(self.backbone.Conv2d_3b_1x1,
                                      self.backbone.Conv2d_4a_3x3)
        self.encoder3 = nn.Sequential(self.backbone.Mixed_5b,
                                      self.backbone.Mixed_5c,
                                      self.backbone.Mixed_5d)
        self.encoder4 = nn.Sequential(self.backbone.Mixed_6a,
                                      self.backbone.Mixed_6b,
                                      self.backbone.Mixed_6c,
                                      self.backbone.Mixed_6d,
                                      self.backbone.Mixed_6e)
        self.encoder5 = nn.Sequential(self.backbone.Mixed_7a,
                                      self.backbone.Mixed_7b,
                                      self.backbone.Mixed_7c)

        self.center = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        )

        self.decoder4 = self._decoder_block(1024 + 768, 768)
        self.decoder3 = self._decoder_block(768 + 288, 288)
        self.decoder2 = self._decoder_block(288 + 192, 192)
        self.decoder1 = self._decoder_block(192 + 64, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # 299x299 -> 149x149
        e2 = self.encoder2(e1)  # 149x149 -> 74x74
        e3 = self.encoder3(e2)  # 74x74 -> 35x35
        e4 = self.encoder4(e3)  # 35x35 -> 17x17
        e5 = self.encoder5(e4)  # 17x17 -> 8x8

        # Center
        center = self.center(e5)  # 8x8 -> 16x16
        center = F.interpolate(center, size=e4.shape[2:], mode='bilinear', align_corners=False)

        # Decoder
        d4 = self.decoder4(torch.cat([center, e4], dim=1))  # 16x16 -> 32x32
        d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))  # 32x32 -> 64x64
        d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))  # 64x64 -> 128x128
        d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d2, e1], dim=1))  # 128x128 -> 256x256

        out = self.final_conv(d1)  # 256x256 -> output size

        # Apply padding to match target size
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out


##########################Load Our UNet trained model on SPECT images####
# Specify the path to your saved model
model_path = "Unet-InceptionNet_mIoU-0.922.pt"

# Load the model onto the CPU
model = torch.load(model_path, map_location=torch.device('cpu'))

# Set the model in evaluation mode
model.eval()
print(model)

########################################################################
def predict_image_mask(model, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)

        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.transforms.functional import normalize
from torchvision import transforms
import re
# Step 1: Load the images
folder_path = 'source/'
image_files = os.listdir(folder_path)
image_files.sort(key=lambda f: int(re.findall(r'DS_(\d+)', f)[0]))


images = [Image.open(os.path.join(folder_path, image_file)).convert("RGB") for image_file in image_files]

# Now, each image should have three channels and match the input dimensions expected by your model.
pred_masks = [predict_image_mask(model, image) for image in images]


# Ensure the mask is on the same device as the image tensor
#pred_mask = pred_mask.to(image_tensor.device)

# Apply the mask to the image tensor
#masked_image = image_tensor * pred_mask.unsqueeze(0)

# Step 3: Save the predicted masks
mask_folder_path = 'output1/'
os.makedirs(mask_folder_path, exist_ok=True)
from scipy import ndimage

for i, pred_mask in enumerate(pred_masks):
    # Convert the mask to uint8
    image_tensor = transforms.ToTensor()(images[i])
    pred_mask = pred_mask.to(image_tensor.device)


    masked_image = image_tensor * pred_mask.unsqueeze(0)


    # Convert the tensor to a numpy array
    masked_image_np = masked_image.permute(1, 2, 0).cpu().detach().numpy()

    # Label each connected component (region) in the mask
    labeled_mask, num_labels = ndimage.label(masked_image_np)

    # Measure the size of each region and find the largest one
    sizes = ndimage.sum(masked_image_np, labeled_mask, range(num_labels + 1))
    mask_size = sizes < max(sizes)

    # Remove small regions
    remove_pixel = mask_size[labeled_mask]
    masked_image_np[remove_pixel] = 0

    # Save the image
    Image.fromarray((masked_image_np * 255).astype(np.uint8)).save(os.path.join(mask_folder_path, f'mask_{i}.png'))



