import os
import pandas as pd
from glob import glob


def main(root_path: str, output_path: str):
    """output path should be the folder path. this creates and the train/ test csv files"""
    img_files = glob(f"{root_path}/*/*.jpg")
    metadata = []
    for img in img_files:
        label = os.path.basename(os.path.dirname(img)) # it is the parent folder
        relative_path = img.replace(f"{root_path}/", "")
        metadata.append({"img_file": relative_path, "label": label})
    
    df = pd.DataFrame(metadata)
    # make a train test split
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    # save the csv files
    test_df.to_csv(f"{output_path}/test.csv", index=False, sep="|")
    train_df.to_csv(f"{output_path}/train.csv", index=False, sep="|")
    

main("/home/logan/projects/paper-implementations/image/gradient-based-learning-applied-to-document-recognition/data/animal-dataset/train",
     "/home/logan/projects/paper-implementations/image/gradient-based-learning-applied-to-document-recognition/data/animal-dataset/train")