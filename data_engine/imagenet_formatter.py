import os
import json
import scipy
import tarfile
import pandas as pd
from glob import glob 
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from PIL import Image

root_path = ""


def create_pairs(mat_file_path, tar_files):
    metadata_map = {}
    tar_file_map = {}
    
    mat_data = scipy.io.loadmat(mat_file_path)
    for metadata in mat_data['synsets']:
        metadata_map[metadata[0][1][0]] = {"label": metadata[0][2][0].split(",")[0]}

    for tar_file in tar_files:
        tar_file_map[os.path.basename(tar_file).split(".")[0]] = tar_file
    
    for key in tar_file_map.keys():
        if key in  metadata_map.keys():
            metadata_map[key]['tar_file'] = tar_file_map[key]
            
    return metadata_map, tar_file_map
    

def untar_into_directory(tar_file, target_directory):
    os.makedirs(target_directory, exist_ok=True)
    with tarfile.open(tar_file, 'r') as tar:
        tar.extractall(path=target_directory)
        

def create_directory_for_tar_file(metadata, root_path):
    if 'tar_file' in metadata.keys():
        sub_folder = os.path.join(f"{root_path}/train", os.path.basename(metadata['tar_file']).split(".")[0])
        untar_into_directory(metadata['tar_file'], sub_folder)
        return sub_folder
    

def check_train_image(image_label):
    image, label = image_label
    try:
        img = Image.open(image)
        img.verify()
        if img.mode not in ['RGB']:  # 'L' for grayscale, 'RGB' for standard color
            return None
        return image_label
    except:
        return None


def check_valid_image(image):
    try:
        img = Image.open(image)
        img.verify()
        if img.mode not in ['RGB']:  # 'L' for grayscale, 'RGB' for standard color
            return None
        return image
    except:
        return None
    
    
if __name__ =="__main__":
    f_path = ""
    tar_files = glob("../*.tar")
    root_path = ""
    meta_map, tar_pairs = create_pairs(f_path, tar_files)
    updated_map = process_map(create_directory_for_tar_file, meta_map.values(), max_workers=8, chunksize=4)
    
    train_images = glob(f"{root_path}/train/*/*.JPEG")
    valid_images = glob(f"{root_path}/val/*/*.JPEG")

    
    image_label_paris = []

    for image in train_images:
        image_label_paris.append((image, os.path.basename(os.path.dirname(image))))
        
    new_pairs = []
    for pair in tqdm(image_label_paris):
        new_pair  = check_train_image(pair)
        if new_pair is not None:
            new_pairs.append(new_pair)
            
            
    folder_label_map = {}
    for pair in new_pairs:
        key = pair[1]
        label = meta_map[pair[1]]['label']
        folder_label_map[key] = label
        
        
    with open('folder_label_map.json', 'w') as fp:
        json.dump(folder_label_map, fp)
        
    df_train = pd.DataFrame(new_pairs, columns=['image', 'label'])
    
    new_pairs = []
    for pair in tqdm(valid_images):
        new_pair  = check_valid_image(pair)
        if new_pair is not None:
            new_pairs.append(new_pair)
            
    image_label_pairs_val = []

    for image in new_pairs:
        image_label_pairs_val.append((image, os.path.basename(os.path.dirname(image))))
    
    df_val = pd.DataFrame(image_label_pairs_val, columns=['image', 'label'])
    
    df_val.to_csv('eval.csv', index=False, sep="|")
    df_train.to_csv('train.csv', index=False, sep="|")