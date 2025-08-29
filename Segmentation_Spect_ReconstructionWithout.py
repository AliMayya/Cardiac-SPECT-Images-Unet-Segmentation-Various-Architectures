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
import re


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
def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3,padding=1),
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
    delta= tensor_size - target_size
    delta = delta//2
    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]

class UNet(nn.Module):
    def __init__(self, num_channels, num_classes,retain_dim=True):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(num_channels, out_c=64)
        self.down_conv_2 = double_conv(in_c=64, out_c=128)
        self.down_conv_3 = double_conv(in_c=128, out_c=256)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2,stride=2)
        self.up_conv_1 = double_conv(256, 128)
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2,stride=2)
        self.up_conv_2 = double_conv(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels = num_classes, # Number of objects to segment
            kernel_size=1,
        )
        self.retain_dim = retain_dim

    def forward(self, image,out_size=(64, 64)):
        # encoder part
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)

        # decoder part
        x = self.up_trans_1(x5)
        y = crop_img(x3, x)
        x = self.up_conv_1(torch.cat([x, y], 1))

        x = self.up_trans_2(x)
        y = crop_img(x1, x)
        x = self.up_conv_2(torch.cat([x, y], 1))

        x = self.out(x)
        if self.retain_dim:
            x = F.interpolate(x, out_size)
        x = F.softmax(x, dim=1)
        return x

##########################Load Our UNet trained model on SPECT images####
# Specify the path to your saved model
model_path = "Unet-VGGAhmad.pt"

# Load the model onto the CPU
model = torch.load(model_path, map_location=torch.device('cpu'))

# Set the model in evaluation mode
model.eval()
print(model)
def predict_image_mask(model, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    pred_mask = torch.argmax(output, dim=1).squeeze(0)
    return pred_mask

# Step 1: Load the images
folder_path = 'source/'
image_files = os.listdir(folder_path)
image_files.sort(key=lambda f: int(re.findall(r'DS_(\d+)', f)[0]))

images = [Image.open(os.path.join(folder_path, image_file)).convert("RGB") for image_file in image_files]

# Step 2: Process images in groups of 8
mask_folder_path = 'output/'
os.makedirs(mask_folder_path, exist_ok=True)

num_images = len(images)

for i in range(num_images):
    # Handle the last few images by ensuring we have 4 images to sum
    if i > num_images - 2:
        indices = list(range(num_images - 2, num_images))
    else:
        indices = list(range(i, i + 2))
    
    # Sum the images in this group
    summed_image = np.zeros_like(np.array(images[0]), dtype=np.uint8)
    for idx in indices:
        summed_image += np.array(images[idx], dtype=np.uint8)
    #summed_image /= 4
    summed_image = summed_image.astype(np.uint8)
    
    # Apply UNet model to the summed image
    summed_image_pil = Image.fromarray(summed_image)
    pred_mask = predict_image_mask(model, summed_image_pil)
    
    # Resize pred_mask to the original image size
    pred_mask_resized = T.Resize((images[0].height, images[0].width))(pred_mask.unsqueeze(0)).squeeze(0)
    
    # Apply the mask to the current image
    image_tensor = T.ToTensor()(images[i])
    pred_mask_expanded = pred_mask_resized.unsqueeze(0).repeat(3, 1, 1)
    pred_mask_expanded = pred_mask_expanded.to(image_tensor.device)
    masked_image = image_tensor * pred_mask_expanded
    
    # Convert the tensor to a numpy array
    masked_image_np = masked_image.permute(1, 2, 0).cpu().detach().numpy()
    from scipy import ndimage  
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

