import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms
import timm
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler



def load_image(image_path, device):
    try:
        # load images
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image at {image_path}. The file may be corrupt or not an image.")

        # Convert image color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert numpy array to PIL image
        image = Image.fromarray(image)

        # Apply image transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0).to(device)

        return image

    except Exception as e:
        # print errors
        print(f"An error occurred while loading the image: {e}")
        return None

def extract_features(model, image):
    model.eval()
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy()

def compute_rdm(features):
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    dissimilarity = pdist(features, metric='euclidean')
    rdm = squareform(dissimilarity)
    return rdm

def plot_rdm(rdm, title,labels, save_path, cmap="hot"):
    plt.imshow(rdm, cmap=cmap, interpolation='nearest')
    plt.title(title,fontsize=21)
    plt.colorbar()
    plt.xticks(ticks=range(len(labels)), labels=labels,fontsize=16)
    plt.yticks(ticks=range(len(labels)), labels=labels,fontsize=16)
    plt.tight_layout()

    plt.savefig(save_path,dpi=300)
    plt.close()  

def plot_muller(rdm, title, save_path, x_labels=None, y_labels=None, cmap="YlGnBu"):
    plt.figure(figsize=(12, 12))  
    plt.imshow(rdm, cmap=cmap, interpolation='nearest')
    plt.title(title,fontsize=16)
    plt.colorbar()
    if x_labels is not None:
        plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=90,fontsize=12)
    if y_labels is not None:
        plt.yticks(ticks=range(len(y_labels)), labels=y_labels,fontsize=12)
    plt.tight_layout()

    plt.savefig(save_path,dpi=300)
    plt.close()  

def plot_all_color(rdm, title, labels, save_path, cmap='hot'):
    plt.figure(figsize=(12, 12))  

    plt.imshow(rdm, cmap=cmap, interpolation='nearest')
    plt.title(title,fontsize=21)
    plt.colorbar()

    tick_positions = [i * 4 + 1.5 for i in range(len(labels))]

    plt.xticks(ticks=tick_positions, labels=labels, rotation=45, ha='right',fontsize=16)
    plt.yticks(ticks=tick_positions, labels=labels, rotation=45, va='top',fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path,dpi=300)
    plt.close()  

def plot_rdm_grid(rdm, title, labels, save_path, cmap='hot'):
    plt.figure(figsize=(12, 12))  

    plt.imshow(rdm, cmap=cmap, interpolation='nearest')
    plt.title(title,fontsize=21)
    plt.colorbar()
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=90, ha='right',fontsize=16) 
    plt.yticks(ticks=range(len(labels)), labels=labels, va='top',fontsize=16)  
    plt.tight_layout()
    plt.savefig(save_path,dpi=300)
    plt.close()  

def plot_all_grid(rdm, title, labels, save_path, cmap='hot'):
    plt.figure(figsize=(12, 12))  

    plt.imshow(rdm, cmap=cmap, interpolation='nearest')
    plt.title(title,fontsize=18)
    plt.colorbar()


    tick_positions = [i * 25 + 12 for i in range(len(labels))] 

    plt.xticks(ticks=tick_positions, labels=labels, ha='right',fontsize=16)
    plt.yticks(ticks=tick_positions, labels=labels,  va='top',fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path,dpi=300)
    plt.close() 

def process_pogbuff_illusion(model, device, base_folder, result_dir):
    real_features, sub_features = [], []
    real_labels, sub_labels = [], []

    # Process images and collect labels
    for image_file in sorted(os.listdir(base_folder)):
        image_path = os.path.join(base_folder, image_file)
        image = load_image(image_path, device)
        feature = extract_features(model, image)
        feature = np.squeeze(feature)

        if image_file.startswith("real"):
            real_features.append(feature)
            real_labels.append(image_file.split('.')[0])
        elif image_file.startswith("sub"):
            sub_features.append(feature)
            sub_labels.append(image_file.split('.')[0])

    # Compute RDM between real and sub
    rdm_real_sub = np.zeros((len(real_features), len(sub_features)))
    for i, real_feat in enumerate(real_features):
        for j, sub_feat in enumerate(sub_features):
            rdm_real_sub[i, j] = np.linalg.norm(real_feat - sub_feat)

    scaler = MinMaxScaler()
    rdm_real_sub = scaler.fit_transform(rdm_real_sub)
    # Save RDMs
    save_path = os.path.join(result_dir, "Poggendorff_Real_Sub_RDM.png")
    plot_muller(rdm_real_sub, 'RDM for Poggendorff - Real and Sub', save_path, real_labels, sub_labels)

    # Compute and save overall RDM
    combined_features = np.vstack([real_features, sub_features])
    combined_labels = real_labels + sub_labels


    overall_rdm = compute_rdm(combined_features)
    save_path = os.path.join(result_dir, "Pogbuff_Combined_RDM.png")
    overall_rdm = scaler.fit_transform(overall_rdm)
    plot_muller(overall_rdm, 'Overall RDM for Pogbuff Illusion', save_path, combined_labels, combined_labels)

def process_muller_illusion(model, device, base_folder, result_dir):
    ori_features, per_length1_features, per_length2_features = [], [], []
    ori_labels, per_length1_labels, per_length2_labels  = [], [], []

    arrow_feature1,arrow_feature2 = [],[]
    arrow_label1,arrow_label2 = [],[]

    # Process images
    for image_file in sorted(os.listdir(base_folder)):
        image_path = os.path.join(base_folder, image_file)
        image = load_image(image_path, device)
        feature = extract_features(model, image)
        feature = np.squeeze(feature)

        if image_file.startswith("Ori"):
            ori_features.append(feature)
            ori_labels.append(image_file.split('.')[0])

        elif "Per" in image_file and "_1" in image_file :
            per_length1_features.append(feature)
            per_length1_labels.append(image_file.split('.')[0])

        elif "Per" in image_file and "_2" in image_file:
            per_length2_features.append(feature)
            per_length2_labels.append(image_file.split('.')[0])

        elif "Arrow" in image_file and "_1" in image_file:
            arrow_feature1.append(feature)
            arrow_label1.append(image_file.split('.')[0])

        elif "Arrow" in image_file and "_2" in image_file:
            arrow_feature2.append(feature)
            arrow_label2.append(image_file.split('.')[0])

    # Compute RDM between Ori and Per_length 1
    rdm_ori_per_length1 = np.zeros((10, 10))
    for i, ori_feat in enumerate(ori_features):
        for j, per_length1_feat in enumerate(per_length1_features):
            rdm_ori_per_length1[i, j] = np.linalg.norm(ori_feat - per_length1_feat)

    # Compute RDM between Ori and Per_length 2
    rdm_ori_per_length2 = np.zeros((10, 10))
    for i, ori_feat in enumerate(ori_features):
        for j, per_length2_feat in enumerate(per_length2_features):
            rdm_ori_per_length2[i, j] = np.linalg.norm(ori_feat - per_length2_feat)

    # Compute RDM between Per_length 1 and Per_length 2
    rdm_per_length1_2 = np.zeros((10, 10))
    for i, per_length1_feat in enumerate(per_length1_features):
        for j, per_length2_feat in enumerate(per_length2_features):
            rdm_per_length1_2[i, j] = np.linalg.norm(per_length1_feat - per_length2_feat)

    rdm_arrow = np.zeros((10, 10))
    for i, ar_length1_feat in enumerate(arrow_feature1):
        for j, ar_length2_feat in enumerate(arrow_feature2):
            rdm_arrow[i, j] = np.linalg.norm(ar_length1_feat - ar_length2_feat)

    rdm_ori_ar1 = np.zeros((10, 10))
    for i, ori_feat in enumerate(ori_features):
        for j, ar_length1_feat in enumerate(arrow_feature1):
            rdm_ori_ar1[i, j] = np.linalg.norm(ori_feat - ar_length1_feat)

    rdm_ori_ar2 = np.zeros((10, 10))
    for i, ori_feat in enumerate(ori_features):
        for j, ar_length2_feat in enumerate(arrow_feature2):
            rdm_ori_ar2[i, j] = np.linalg.norm(ori_feat - ar_length2_feat)

    scaler = MinMaxScaler()
    rdm_ori_per_length1= scaler.fit_transform(rdm_ori_per_length1)
    rdm_ori_per_length2= scaler.fit_transform(rdm_ori_per_length2)
    rdm_per_length1_2 = scaler.fit_transform(rdm_per_length1_2)
    rdm_arrow = scaler.fit_transform(rdm_arrow)
    rdm_ori_ar1 = scaler.fit_transform(rdm_ori_ar1)
    rdm_ori_ar2= scaler.fit_transform(rdm_ori_ar2)


    # Use modified plot_rdm function to save RDMs
    plot_muller(rdm_ori_per_length1, 'RDM for Muller - Ori and Per Length Outward',
             os.path.join(result_dir, "Muller_Ori_Per_Length1_RDM.png"), x_labels=ori_labels,
             y_labels=per_length1_labels)
    plot_muller(rdm_ori_per_length2, 'RDM for Muller - Ori and Per Length Inward',
             os.path.join(result_dir, "Muller_Ori_Per_Length2_RDM.png"), x_labels=ori_labels,
             y_labels=per_length2_labels)
    plot_muller(rdm_per_length1_2, 'RDM for Muller - Per Length Outward and Inward',
             os.path.join(result_dir, "Muller_Per_Length1_2_RDM.png"), x_labels=per_length1_labels,
             y_labels=per_length2_labels)

    plot_muller(rdm_arrow, 'RDM for Muller - Arrow Ori Length Outward and Inward ',
             os.path.join(result_dir, "RDM-Arrow_1_2.png"), x_labels=arrow_label1,
             y_labels=arrow_label2)
    plot_muller(rdm_ori_ar1, 'RDM for Muller - Ori and Arrow Outward',
             os.path.join(result_dir, "RDM-Ori-Arrow-1.png"), x_labels=ori_labels,
             y_labels=arrow_label1)
    plot_muller(rdm_ori_ar2, 'RDM for Muller - Ori and Arrow Inward',
             os.path.join(result_dir, "RDM-Ori-Arrow-2.png"), x_labels=ori_labels,
             y_labels=arrow_label2)

def process_zoller_illusion(model, device, base_folder, result_dir):
    stimi_features, subject_features = [], []
    stimi_labels, subject_labels = [], []

    # Process images
    for image_file in sorted(os.listdir(base_folder)):
        image_path = os.path.join(base_folder, image_file)
        image = load_image(image_path, device)
        feature = extract_features(model, image)
        feature = np.squeeze(feature)

        if image_file.startswith("stimi"):
            stimi_features.append(feature)
            stimi_labels.append(image_file.split('.')[0])
        elif image_file.startswith("subject"):
            subject_features.append(feature)
            subject_labels.append(image_file.split('.')[0])


    # Compute RDM between stimi and subject
    rdm_stimi_subject = np.zeros((len(stimi_features), len(subject_features)))
    for i, stimi_feat in enumerate(stimi_features):
        for j, subject_feat in enumerate(subject_features):
            rdm_stimi_subject[i, j] = np.linalg.norm(stimi_feat - subject_feat)

    scaler = MinMaxScaler()
    rdm_stimi_subject= scaler.fit_transform(rdm_stimi_subject)

    # Save RDMs
    save_path = os.path.join(result_dir, "Zoller_Stimi_Subject_RDM.png")
    plot_muller(rdm_stimi_subject, 'RDM for Zoller - Stimi and Subject', save_path, stimi_labels, subject_labels)

    # Compute and save overall RDM
    combined_features = np.vstack([stimi_features, subject_features])
    combined_labels = stimi_labels + subject_labels
    overall_rdm = compute_rdm(combined_features)
    save_path = os.path.join(result_dir, "Zoller_Combined_RDM.png")
    overall_rdm= scaler.fit_transform(overall_rdm)

    plot_muller(overall_rdm, 'Overall RDM for Zoller', save_path, combined_labels, combined_labels)

def process_grid_illusion(model, device, base_folder, result_dir):
    images = os.listdir(base_folder)
    all_features = []

    # Assume the color name is the part of the file name before the first underscore
    color_names = set([img.split('_')[0] for img in images])
    labels_all = color_names

    gradient_map = {'0': 'μ_1', '51': 'μ_2', '102': 'μ_3', '153': 'μ_4', '204': 'μ_5'}
    dsize_map = {'6': 'dsize_6', '7': 'dsize_7', '8': 'dsize_8', '9': 'dsize_9', '10': 'dsize_10'}
    scaler = MinMaxScaler()

    for color_name in color_names:
        color_images = [img for img in images if img.startswith(color_name)]
        # sort out
        color_images.sort(key=lambda x: (gradient_map[x.split('_')[1]], dsize_map[x.split('_')[2].split('.')[0]]))
        features_per_color = []

        for image_file in color_images:
            image_path = os.path.join(base_folder, image_file)
            image = load_image(image_path, device)
            feature = extract_features(model, image)
            features_per_color.append(feature)
            all_features.append(feature)  

        # Generate a list of tags for RDM for each color
        labels_per_color = [f'{gradient_map[grad]}_{dsize_map[dsize]}' for grad in gradient_map for dsize in dsize_map]

        # Compute and save RDM for each color
        features_per_color = np.vstack(features_per_color)
        rdm_per_color = compute_rdm(features_per_color)
        save_path = os.path.join(result_dir, f"{color_name}_RDM.png")
        rdm_per_color = scaler.fit_transform(rdm_per_color)
        plot_rdm_grid(rdm_per_color, f'RDM for Grid Illusion - {color_name}', labels_per_color, save_path)


    # Compute and save overall RDM
    all_features = np.vstack(all_features)
    overall_rdm = compute_rdm(all_features)
    save_path = os.path.join(result_dir, "Grid_Illusion_Overall_RDM.png")
    overall_rdm = scaler.fit_transform(overall_rdm)
    plot_all_grid(overall_rdm, 'Overall RDM for Grid Illusion', labels_all, save_path)


def process_color_illusion(model, device, base_folder, model_result_dir):
    images = os.listdir(base_folder)
    all_features = []
    labels = ['cube', 'depth_1', 'depth_2', 'depth_3']

    if not os.path.exists(model_result_dir):
        os.makedirs(model_result_dir)

    color_names = set([img.split('_')[0] for img in images])
    scaler = MinMaxScaler()
    for color_name in color_names:
        features_per_color = []
        
        for label in labels:
            image_file = f"{color_name}_{label}.png"
            if image_file in images:
                image_path = os.path.join(base_folder, image_file)
                image = load_image(image_path, device)
                feature = extract_features(model, image)
                features_per_color.append(feature)
                all_features.append(feature)

        # Compute and save RDM for each color
        #features_per_color = np.vstack(features_per_color)
        #rdm_per_color = compute_rdm(features_per_color)
        #save_path = os.path.join(model_result_dir, f"{color_name}_RDM.png")
        #rdm_per_color =scaler.fit_transform(rdm_per_color)
        #plot_rdm(rdm_per_color, f'RDM for Color Illusion - {color_name}', labels, save_path)

    labels_all = list(color_names)  

    # Compute and save overall RDM
    all_features = np.vstack(all_features)
    overall_rdm = compute_rdm(all_features)
    save_path = os.path.join(model_result_dir, "Color_Illusion_Overall_RDM.png")
    overall_rdm = scaler.fit_transform(overall_rdm)

    plot_all_color(overall_rdm, 'Overall RDM for Color Illusion', labels_all, save_path)


def extract_features_at_layer(model, layer_name, image):
    model.eval()
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    getattr(model, layer_name).register_forward_hook(get_activation(layer_name))
    with torch.no_grad():
        model(image)
    return activation[layer_name].squeeze()

def compute_average_distance(model, device, image_pairs, layer_names):
    average_distances = []
    for layer_name in layer_names:
        distances = []
        for img_file1, img_file2 in image_pairs:
            img_path1 = os.path.join(image_folder, img_file1)
            img_path2 = os.path.join(image_folder, img_file2)

            img1 = load_image(img_path1, device)
            img2 = load_image(img_path2, device)

            feature1 = extract_features_at_layer(model, layer_name, img1)
            feature2 = extract_features_at_layer(model, layer_name, img2)

            distance = np.linalg.norm(feature1.numpy() - feature2.numpy())
            distances.append(distance)

        average_distance = np.mean(distances)
        average_distances.append(average_distance)

    return average_distances



def get_feature_extracting_model(model_name, model):
    if model_name in ["AlexNet", "Vgg16", "Vgg19"]:
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name in ["ResNetv2_50", "ResNetv2_101", "ResNet152", "ResNext101"]:
        model = torch.nn.Sequential(*list(model.children())[:-2])
    elif model_name in ["DenNet121", "DenNet169", "DenNet201"]:
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name == "Inception_v3":
        model = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name == "Inception_v4":
        model = torch.nn.Sequential(*list(model.children())[:-1])
    else:
        raise ValueError("Unsupported model name")
    return model



def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using GPU for acceleration' if use_cuda else 'Using CPU for computation')

    # Initial models
    models_to_use = {
        "AlexNet": models.alexnet(weights="AlexNet_Weights.IMAGENET1K_V1").to(device),
        "Vgg16": models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1").to(device),
        "Vgg19": models.vgg19(weights="VGG19_Weights.IMAGENET1K_V1").to(device),
        "ResNetv2_50": timm.create_model('resnetv2_50.a1h_in1k', pretrained=True).to(device),
        "ResNetv2_101":timm.create_model('resnetv2_101.a1h_in1k', pretrained=True).to(device),
        "ResNet152":models.resnet152(weights="ResNet152_Weights.IMAGENET1K_V1").to(device),
        "ResNext101":models.resnext101_32x8d(weights="ResNeXt101_32X8D_Weights.IMAGENET1K_V1").to(device),
        "Inception_v4": timm.create_model("inception_v4",pretrained=True).to(device),
        "Inception_v3": timm.create_model('inception_v3',pretrained=True).to(device),
        "DenNet121": models.densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1").to(device),
        "DenNet169": models.densenet169(weights="DenseNet169_Weights.IMAGENET1K_V1").to(device),
        "DenNet201": models.densenet201(weights="DenseNet201_Weights.IMAGENET1K_V1").to(device)
    }

    result_dir = "result_rdm"
    illusion_folder = 'illusion'
    datasets = ['zoller', 'pogbuff','color','muller','grid'] 

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for dataset in datasets:
        dataset_folder = os.path.join(illusion_folder, dataset)
        dataset_result_dir = os.path.join(result_dir, dataset)

        # childfolder
        if not os.path.exists(dataset_result_dir):
            os.makedirs(dataset_result_dir)

        for model_name, model in models_to_use.items():
            print(f"Processing {dataset} with {model_name} model")
            model = get_feature_extracting_model(model_name, model)
         
            model_dataset_result_dir = os.path.join(dataset_result_dir, model_name)
            if not os.path.exists(model_dataset_result_dir):
                os.makedirs(model_dataset_result_dir)

            if dataset == 'pogbuff':
                process_pogbuff_illusion(model, device, dataset_folder, model_dataset_result_dir)

            if dataset == 'color':
                process_color_illusion(model, device, dataset_folder, model_dataset_result_dir)
                
            if dataset == 'grid':
                process_grid_illusion(model, device, dataset_folder, model_dataset_result_dir)

            if dataset == 'zoller':
                process_zoller_illusion(model, device, dataset_folder, model_dataset_result_dir)

            if dataset == 'muller':
                process_muller_illusion(model, device, dataset_folder, model_dataset_result_dir)


if __name__ == '__main__':
    main()
