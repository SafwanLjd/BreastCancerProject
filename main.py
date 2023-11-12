import pandas as pd
import tensorflow as tf
import os


BASE_DIR = "dataset"
CALC_CSV_PATH = f"{BASE_DIR}/calc_case(with_jpg_img).csv"
MASS_CSV_PATH = f"{BASE_DIR}/mass_case(with_jpg_img).csv"
BATCH_SIZE = 32

def preprocess_image(image, label):
    image = tf.io.read_file(image) # Load
    image = tf.image.decode_jpeg(image, channels=1) # Grey Scale
    image = tf.image.resize(image, [224, 224]) # Scale Down
    image = image / 255.0 # Normalize

    return image, label


def create_dataset(image_paths, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset


def create_label(abnormality_type, pathology):
    if pathology in ["BENIGN", "BENIGN_WITHOUT_CALLBACK"]:
        pathology_label = 0  # Benign
    else:
        pathology_label = 1  # Malignant

    if abnormality_type == "MASS":
        return pathology_label  # 0: Benign Mass; 1: Malignant Mass

    else:
        return 2 + pathology_label  # 2: Benign Calcification; 3: Malignant Calcification



calc_case_df = pd.read_csv(CALC_CSV_PATH)
mass_case_df = pd.read_csv(MASS_CSV_PATH)

image_paths = []
labels = []

for _, row in calc_case_df.iterrows():
    image_path = os.path.join(BASE_DIR, row["jpg_fullMammo_img_path"])
    label = create_label(row["abnormality type"], row["pathology"])
    image_paths.append(image_path)
    labels.append(label)

for _, row in mass_case_df.iterrows():
    image_path = os.path.join(BASE_DIR, row["jpg_fullMammo_img_path"])
    label = create_label(row["abnormality type"], row["pathology"])
    image_paths.append(image_path)
    labels.append(label)


train_image_paths, train_labels, test_image_paths, test_labels = [], [], [], []

for image_path, label in zip(image_paths, labels):
    if "Train" in image_path:
        train_image_paths.append(image_path)
        train_labels.append(label)

    elif "Test" in image_path:
        test_image_paths.append(image_path)
        test_labels.append(label)



train_dataset = create_dataset(train_image_paths, train_labels, batch_size)
test_dataset = create_dataset(test_image_paths, test_labels, batch_size)