import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
import timm
from pytorch_grad_cam import (
    GradCAM, XGradCAM, GradCAMPlusPlus
)
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GuidedBackpropReLUModel
from PIL import Image

def save_image_with_dpi(img_array, output_path, dpi=300):
    """
    Save an image with a specific DPI using PIL.
    :param img_array: Image array in OpenCV format (BGR).
    :param output_path: Path to save the image.
    :param dpi: Desired DPI.
    """
    img_pil = Image.fromarray(img_array)
    img_pil.save(output_path, dpi=(dpi, dpi))

# get target layer
def get_target_layer(model_name, model):
    model_name_lower = model_name.lower()
    if 'resnet' in model_name_lower and not 'v2' in model_name_lower:
        return [model.layer4]
    elif 'resnext' in model_name_lower:
        return [model.layer4]
    elif 'vgg' in model_name_lower or 'dennet' in model_name_lower:
        return [model.features[-1]]
    elif 'inception_v4' in model_name_lower:
        return [model.features[-1]]
    elif 'inception_v3' in model_name_lower:
        return [model.Mixed_7c]
    elif 'alexnet' in model_name_lower:
        return [model.features[-1]]
    elif 'resnetv2' in model_name_lower:
        return [model.stages[-1].blocks[-1].conv3]
    else:
        raise ValueError(f"Model {model_name} not supported")


def deprocess_image(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

def process_image(image_path, model_name, model, cam_methods, folder_name, output_dir, use_cuda):
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255

    # transfer to tensor
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    target_layers = get_target_layer(model_name, model)
    if use_cuda:
        model = model.to('cuda')
        input_tensor = input_tensor.to('cuda')

    """    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)"""

    model.to('cuda')
    device = torch.device("cuda" if use_cuda else "cpu")

    gb_model = GuidedBackpropReLUModel(model=model, device=device)

    # create folder
    folder_output_dir = os.path.join(output_dir, folder_name)
    model_output_dir = os.path.join(folder_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    for method_name, cam_algorithm in cam_methods.items():
        with cam_algorithm(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=None, aug_smooth=True, eigen_smooth=True)

            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            # Create the child-folder for every models
            cam_method_output_dir = os.path.join(model_output_dir, method_name)
            os.makedirs(cam_method_output_dir, exist_ok=True)

            # Save cam images
            cam_output_path = os.path.join(cam_method_output_dir, f'{method_name}_{os.path.basename(image_path)}')
            save_image_with_dpi(cam_image, cam_output_path)

            # Guided Backpropagation
            gb = gb_model(input_tensor, target_category=None)
            gb = deprocess_image(gb)
            gb_output_path = os.path.join(cam_method_output_dir, f'{method_name}_gb_{os.path.basename(image_path)}')
            save_image_with_dpi(gb, gb_output_path)


def process_images_in_folder(folder_path, models_to_use, cam_methods, output_dir, use_cuda):
    folder_name = os.path.basename(folder_path)
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        for model_name, model in models_to_use.items():
            process_image(img_path, model_name, model, cam_methods, folder_name, output_dir, use_cuda)
            print(f"Processed {img_name} with {model_name}")



def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using GPU for acceleration' if use_cuda else 'Using CPU for computation')

    # Initial model
    models_to_use = {


        "AlexNet": models.alexnet(weights="AlexNet_Weights.IMAGENET1K_V1").to(device),
        "Vgg16": models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1").to(device),
        "Vgg19": models.vgg19(weights="VGG19_Weights.IMAGENET1K_V1").to(device),
        "ResNetv2_50": timm.create_model('resnetv2_50.a1h_in1k', pretrained=True).to(device),
        "ResNetv2_101": timm.create_model('resnetv2_101.a1h_in1k', pretrained=True).to(device),
        "ResNet152": models.resnet152(weights="ResNet152_Weights.IMAGENET1K_V1").to(device),
        "ResNext101": models.resnext101_32x8d(weights="ResNeXt101_32X8D_Weights.IMAGENET1K_V1").to(device),
        "Inception_v4": timm.create_model("inception_v4", pretrained=True).to(device),
        "Inception_v3": timm.create_model('inception_v3', pretrained=True).to(device),
        "DenNet121": models.densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1").to(device),
        "DenNet169": models.densenet169(weights="DenseNet169_Weights.IMAGENET1K_V1").to(device),
        "DenNet201": models.densenet201(weights="DenseNet201_Weights.IMAGENET1K_V1").to(device)
    }

    # Define cam
    cam_methods = {
        "gradcam": GradCAM,
        "Xgradcam": XGradCAM,
        "gradcam++": GradCAMPlusPlus
    }


    folder_paths = ['illusion/color', 'illusion/grid', 'illusion/zoller', 'illusion/muller', 'illusion/pogbuff']
    output_dir = 'output'

    for folder_path in folder_paths:
        process_images_in_folder(folder_path, models_to_use, cam_methods, output_dir, use_cuda)

    print("All images processed.")


if __name__ == '__main__':
    main()
