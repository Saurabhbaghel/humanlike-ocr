import numpy as np
import cv2
import torch
from torch import nn
from torchvision import transforms

class FeatureExtractor(nn.Module):
    def __init__(self, img_size:int = 112) -> None:
        super().__init__()
        model_ = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True)
        self.model = torch.nn.Sequential(*(list(model_.children)))
        self.preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # to convert to Grayscale 
            transforms.Resize(img_size), # resizing to 112
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, img):
        self.model.eval()
        input_tensor = self.preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # create a minibatch as expected by the model
        with torch.no_grad():
            output = self.model(input_batch)


def remove_shirorekha(img):
    """
    removes the shirorekha (horizontal line) from the characters.

    params
    ------
    img: a numpy array

    returns
    -------
    black-white image without shirorekha
    """
    assert isinstance(img, np.ndarray), "The img should be a np array. Use opencv to read the image."

    # convert the image to grayscale
    if img.shape != 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = cv2.bitwise_not(gray) # if the text is black on white background

    # thresholding
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    horizontal = np.copy(bw)

    # specify size on horizontal axis 
    cols = horizontal.shape[1]
    horizontal_size = cols // 3

    # create structure element for extracting horizontal line through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    return bw-horizontal


def get_histogram_pixels(img):
    """
    returns the list of number of pixels with the value 1 along the x-axis
    """

    # black-white image without shirorekha
    bw_wo_shirorekha = remove_shirorekha(img)

    # histogram
    y = [(bw_wo_shirorekha[:, i]/255.0).sum() for i in range(bw_wo_shirorekha.shape[1])]

    return y
