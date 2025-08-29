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

class UNet2(nn.Module):
    def __init__(self, num_classes, retain_dim=True):
        super(UNet2, self).__init__()

        # Load pretrained MobileNetV2 model and extract the features
        self.encoder = models.mobilenet_v2(pretrained=True).features
        self.backbone_out_channels = 1280

        # Define the upsampling layers and double_conv layers
        self.up_trans_1 = nn.ConvTranspose2d(in_channels=self.backbone_out_channels, out_channels=320, kernel_size=2, stride=2)
        self.up_conv_1 = double_conv(320 + 96, 320)

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=320, out_channels=96, kernel_size=2, stride=2)
        self.up_conv_2 = double_conv(96 + 32, 96)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=96, out_channels=32, kernel_size=2, stride=2)
        self.up_conv_3 = double_conv(32 + 24, 32)

        self.up_trans_4 = nn.ConvTranspose2d(in_channels=32, out_channels=24, kernel_size=2, stride=2)
        self.up_conv_4 = double_conv(24 + 16, 24)  # Adjusted to match the expected input channels

        self.out = nn.Conv2d(in_channels=24, out_channels=num_classes, kernel_size=1)
        self.retain_dim = retain_dim

    def forward(self, image, out_size=(64, 64)):
        # Encoder part
        enc_features = []
        x = image
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            if i in {1, 3, 6, 13, 18}:  # Save features at specific layers for concatenation
                enc_features.append(x)

        # Decoder part
        x = self.up_trans_1(enc_features[-1])
        y = crop_img(enc_features[-2], x)
        x = self.up_conv_1(torch.cat([x, y], 1))

        x = self.up_trans_2(x)
        y = crop_img(enc_features[-3], x)
        x = self.up_conv_2(torch.cat([x, y], 1))

        x = self.up_trans_3(x)
        y = crop_img(enc_features[-4], x)
        x = self.up_conv_3(torch.cat([x, y], 1))

        x = self.up_trans_4(x)
        y = crop_img(enc_features[-5], x)
        x = self.up_conv_4(torch.cat([x, y], 1))

        x = self.out(x)
        if self.retain_dim:
            x = F.interpolate(x, out_size)
        x = F.softmax(x, dim=1)
        return x



##########################Load Our UNet trained model on SPECT images####
# Specify the path to your saved model
model_path = "Unet-Mobilenet_v2_mIoU-0.893.pt"

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



