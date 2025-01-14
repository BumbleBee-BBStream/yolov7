"""
This script automates the process of training a YOLOv7 object detection model with custom datasets. 
It includes several stages specified in the main function: downloading datasets, preparing data, 
training the model, and converting the trained model to the ONNX format. Therefore the best way to
understand this script is to start from the main function, and dive into specific steps if needed.

The script is designed to handle multiple datasets simultaneously and uses multithreading for 
efficient data processing.

Key functionalities include:

1. Download and Extraction: Downloads datasets from the remote labeling server based on provided 
   dataset IDs. The datasets are then extracted and organized into a specific directory structure.

2. Data Preparation: Parses XML files to extract class information and bounding box details. 
   The data is then split into training and validation sets. YOLO-format label files are created 
   for each image in the dataset.

3. YOLOv7 Training: Initiates the training of a YOLOv7 model using the prepared data. Training 
   parameters such as epochs, batch size, and model configuration can be customized through 
   command-line arguments.

4. Model Conversion to ONNX: After training, the best-performing model is converted to the ONNX 
   format for deployment and inference purposes. 

5. Progress Tracking: The script provides real-time feedback on the progress of data downloading 
   and model training.

6. Error Handling: Includes custom exception handling for download errors and uses a safe exit 
   strategy in case of failures during dataset download or model training.

7. Command-Line Arguments: Allows customization of various parameters like dataset IDs, training 
   settings, and paths through command-line arguments.

8. Polling Mechanism: After training, the script continuously checks for the completion of the 
   training process before proceeding to model conversion (to ONNX format).

Usage:
Run the script from the command line, optionally passing arguments to customize the training 
process. The script requires an internet connection for downloading datasets and may require 
access to a GPU for model training.

Example:
python3 auto_train.py --dataset-ids 12345 67890 --epochs 50 --batch-size 16

Notes:
- The script is designed to be run in an environment where YOLOv7 and its dependencies are installed.
- Ensure that the dataset server's URL and file structure are compatible with the script's 
  download logic.
- The script uses multi-threading and subprocesses; ensure that the running environment 
  supports these features.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import requests
import os
import shutil
import yaml
import zipfile
import xml.etree.ElementTree as ET
import random
import subprocess
import sys
import time
import threading


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## -- variables --
dataset_ids = ["667430116996943872"]  # dataset id from http://www.deepvision-tech.cn:8090/Home/Index 
project_path = "/home/workspace/BatteryDataSet"
project_version = "14_defect_5_JinChuan_auto_script_test"
training_data_percentage = 0.75  # how much % of data is used for training, with the rest as validation
yolov7_epochs = 100
yolov7_workers = 4
yolov7_device = 0
yolov7_batch_size = 32
yolov7_img_size = 640
yolov7_weights = "/home/workspace/yolo_weights/yolov7-tiny.pt"  # weights for the yolov7 model
yolov7_hyper = "data/hyp.scratch.tiny.yaml"  # hyper parameters for the yolov7 model
# parameters for ONNX conversion
yolov7_iou_thres = 0.65  
yolov7_conf_thres = 0.35
yolov7_topk_all = 100

## -- constants that are less likely to change --
YOLOV7_PATH = "/home/workspace/yolov7"  # yolov7 path, this is also the path under which this script resides
DATASRC_HOST = "http://www.deepvision-tech.cn:8090/"  # labeling server host address
DOWNLOAD_URI = DATASRC_HOST + "TaskManage/TaskData/ExportData"  # URI endpoint for dataset download


## -- helper functions --
download_progress = {}
lock = threading.Lock()
def download_file(url, filename, id):
    """
    Downloads a file from a given URL and saves it to a specified local filename.

    The function makes an HTTP GET request to the provided URL and streams the
    content to the local file system. The content is written in chunks to handle
    large files efficiently. It also logs the progress of the download, displaying
    the percentage of the file that has been downloaded.

    Parameters:
    url (str): The URL from which to download the file.
    filename (str): The local path where the downloaded file will be saved.
    id (str): The dataset id for tracking purpose.

    This function uses the 'requests' library to perform the HTTP request and
    stream the content. The file is opened in binary write mode ('wb'), and the
    content is written in chunks of 1 Kibibyte. The progress of the download is
    calculated based on the total content length obtained from the HTTP headers and
    the amount of data downloaded. This progress is logged to the console.
    """
    global download_progress
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_length = int(r.headers.get('content-length', 0))
        chunk_size = 1024  # 1 Kibibyte
        downloaded = 0

        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    percentage = 100 * downloaded / total_length
                    # Update shared progress tracker
                    with lock:
                        download_progress[id] = percentage


def print_progress():
    # print download progress for all datasets
    global download_progress
    download_progress = {id: 0 for id in dataset_ids}  # initialize progress to 0 for all datasets
    while any(progress < 100 for progress in download_progress.values()):
        with lock:
            progress_str = ', '.join(f"ID {id}: {progress:.2f}%" for id, progress in download_progress.items())
        print(f"\r{progress_str}", end='', flush=True)
        time.sleep(0.5)  # Update every half second


def unzip_and_organize(zip_file):
    # Create directories for images and xmls if they don't exist
    images_path = os.path.join(project_raw_data_path, 'images')
    xmls_path = os.path.join(project_raw_data_path, 'xmls')

    # Unzipping the file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Extract all the contents into the current directory
        zip_ref.extractall()

        # Iterate over each file in the zip archive
        for file in zip_ref.namelist():
            if file.endswith('.jpg'):
                # Move jpg files to images folder
                os.rename(file, os.path.join(images_path, file))
            elif file.endswith('.xml'):
                # Move xml files to xmls folder
                os.rename(file, os.path.join(xmls_path, file))


def extract_classes(xml_dir):
    """
    Extracts and returns a sorted list of unique class names from XML files in a specified directory.

    This function iterates through all XML files in the given directory. It parses each XML file
    to extract the 'name' tags within 'object' tags, which typically represent class names in
    object detection datasets. The function collects all unique class names found across all
    XML files and returns them as a sorted list. This is useful for preparing class information
    for machine learning tasks, particularly for object detection models.

    Parameters:
    xml_dir (str): The directory containing XML files with object annotations.

    Returns:
    list: A sorted list of unique class names found in the XML files.

    The function uses the ElementTree XML API to parse XML files. It creates a set to store 
    unique class names and then converts this set to a list which is sorted before returning.
    This ensures that the class names are unique and are in a consistent order.
    """
    classes = set()
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()
            for obj in root.iter('object'):
                classes.add(obj.find('name').text.strip())
    return sorted(classes)


def setup_yolov7_data_directories(base_dir):
    dirs = {
        'train_images': os.path.join(base_dir, 'images', 'train'),
        'val_images': os.path.join(base_dir, 'images', 'val'),
        'train_labels': os.path.join(base_dir, 'labels', 'train'),
        'val_labels': os.path.join(base_dir, 'labels', 'val')
    }

    for dir in dirs.values():
        if not os.path.exists(dir):
            os.makedirs(dir)

    return dirs


def create_data_yaml_file(base_dir, classes):
    yaml_content = {
        'train': os.path.join(base_dir, 'train.txt'),
        'val': os.path.join(base_dir, 'val.txt'),
        'nc': len(classes),
        'names': classes
    }

    with open(os.path.join(base_dir, 'data.yaml'), 'w') as file:
        yaml.dump(yaml_content, file, sort_keys=False)


def create_cfg_yaml_file(input_yaml_path, output_yaml_path, classes):
    # Read the existing YAML file
    with open(input_yaml_path, 'r') as file:
        yaml_content = yaml.safe_load(file)

    # Update the 'nc' field with the number of classes
    yaml_content['nc'] = len(classes)

    # Write the updated YAML content to a new file
    with open(output_yaml_path, 'w') as file:
        yaml.dump(yaml_content, file, sort_keys=False)


def parse_xml(file, class_map):
    """
    Parses an XML file to extract object bounding boxes and class information.

    This function reads an XML file that contains annotations for objects in an image.
    It extracts the bounding box coordinates for each annotated object and maps the 
    object's class name to a class ID using the provided class_map. 
    The bounding box coordinates are normalized by the dimensions of the image.

    Parameters:
    file (str): The path to the XML file containing annotations.
    class_map (dict): A dictionary mapping class names (str) to class IDs (int).

    Returns:
    list of tuples: A list where each tuple represents an object's annotation.
                    Each tuple contains:
                    - class_id (int): The class ID of the object.
                    - b (tuple of floats): The normalized bounding box of the object,
                      represented as (xmin, xmax, ymin, ymax).

    The function processes each 'object' node in the XML file, retrieves the class
    name, maps it to its corresponding class ID, and extracts the bounding box
    coordinates (xmin, xmax, ymin, ymax). These coordinates are normalized by the
    width and height of the image, which are also extracted from the XML file.
    """
    tree = ET.parse(file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    boxes = []
    for obj in root.iter('object'):
        cls = obj.find('name').text.strip()
        if cls in class_map:
            xmlbox = obj.find('bndbox')
            b = (
                float(xmlbox.find('xmin').text) / w,
                float(xmlbox.find('xmax').text) / w,
                float(xmlbox.find('ymin').text) / h,
                float(xmlbox.find('ymax').text) / h
            )
            boxes.append((class_map[cls], b))
    return boxes


def write_labels(filename, boxes):
    """
    Writes the label information to a file in YOLO format.

    This function takes the bounding box information for each object in an image 
    and writes it to a specified file. The format written is specific to the YOLOv7 
    model, which requires each line in the label file to contain class ID and normalized 
    bounding box coordinates (center x, center y, width, height).

    Parameters:
    filename (str): The path of the file where the label data will be written.
    boxes (list of tuples): A list where each tuple represents a detected object.
        Each tuple contains:
        - class_id (int): The class ID of the object.
        - xmin (float): The x coordinate of the top-left corner of the bounding box.
        - xmax (float): The x coordinate of the bottom-right corner of the bounding box.
        - ymin (float): The y coordinate of the top-left corner of the bounding box.
        - ymax (float): The y coordinate of the bottom-right corner of the bounding box.
        Coordinates are normalized to the range [0, 1].

    The function calculates the center coordinates (x_center, y_center) and the 
    width and height of the bounding box by using xmin, xmax, ymin, and ymax. 
    These values are then written to the file in the YOLO format.
    """
    with open(filename, 'w') as f:
        for class_id, (xmin, xmax, ymin, ymax) in boxes:
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def split_and_write_labels(image_dir, annotation_dir, dirs, class_map):
    """
    Splits the dataset into training and validation sets and writes label files.

    This function processes a dataset of images and their corresponding annotation
    files, divides the dataset into training and validation subsets based on a 
    predefined ratio, and writes the YOLO-formatted label files for each subset.
    It also creates text files listing the images in each subset.

    Parameters:
    image_dir (str): Directory containing the image files.
    annotation_dir (str): Directory containing the corresponding XML annotation files.
    dirs (dict): A dictionary containing paths to output directories for training 
                 and validation images and labels.
    class_map (dict): A mapping of class names to their respective integer IDs.

    The function first shuffles the list of images to ensure random distribution 
    into training and validation sets. It then iterates over each image, checks 
    for a corresponding annotation file, and if found, parses the XML file to get 
    bounding box information. This information is converted to YOLO format and 
    written to label files in the respective training or validation directories.
    Additionally, paths to images are written to 'train.txt' and 'val.txt' files
    for use during model training.
    """
    logging.info("Splitting dataset and writing label files...")
    split_ratio = training_data_percentage
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    random.shuffle(images)
    split = int(len(images) * split_ratio)
    with open(os.path.join(project_data_coco_path, 'train.txt'), 'w') as train_txt, open(os.path.join(project_data_coco_path, 'val.txt'), 'w') as val_txt:
        for i, image_name in enumerate(images):
            base_name = os.path.splitext(image_name)[0]
            annotation_file = os.path.join(annotation_dir, base_name + '.xml')
            image_file = os.path.join(image_dir, image_name)

            if os.path.exists(annotation_file):
                boxes = parse_xml(annotation_file, class_map)

                if i < split:
                    output_image_dir = dirs['train_images']
                    output_label_dir = dirs['train_labels']
                    train_txt.write(os.path.join(output_image_dir, image_name) + '\n')
                else:
                    output_image_dir = dirs['val_images']
                    output_label_dir = dirs['val_labels']
                    val_txt.write(os.path.join(output_image_dir, image_name) + '\n')

                label_file = os.path.join(output_label_dir, base_name + '.txt')
                shutil.copy(image_file, output_image_dir)
                write_labels(label_file, boxes)


class DownloadError(Exception):
    """Custom exception class for download errors."""
    pass


def download_and_extract_dataset(id):
    """
    Downloads and extracts a dataset given its ID.

    This function sends a POST request to a server to obtain the download URL for a dataset
    identified by the given ID. It then downloads the dataset from the URL and extracts it 
    to a specified directory. If the POST request fails or the response does not contain 
    the expected data, a DownloadError exception is raised.

    Parameters:
    id (str): The unique identifier for the dataset to be downloaded.

    Raises:
    DownloadError: If the POST request fails or the 'Data' field is not found in the response.
    """
    payload = {"id": id, "type": 1}
    response = requests.post(DOWNLOAD_URI, data=payload)
    if response.status_code == 200:
        response_json = response.json()
        file_url_suffix = response_json.get("Data")
        if file_url_suffix:
            file_url = DATASRC_HOST + file_url_suffix
            file_name = file_url_suffix.split('/')[-1]
            file_path = os.path.join(project_raw_data_path, file_name)
            download_file(file_url, file_path, id)
            unzip_and_organize(file_path)
        else:
            raise DownloadError(f"Data field not found in the response for {id}. Check {DATASRC_HOST}.")
    else:
        raise DownloadError(f"POST request failed for {id}. Check {DATASRC_HOST}.")


def count_lines(filename):
    # count number of lines in a file.
    try:
        with open(filename, 'r') as file:
            line_count = sum(1 for line in file)
        return line_count
    except FileNotFoundError:
        return -1

## -- main functions --
def parse_arguments():
    global dataset_ids, project_path, project_version, training_data_percentage
    global yolov7_epochs, yolov7_workers, yolov7_device, yolov7_batch_size, yolov7_img_size, yolov7_weights, yolov7_hyper
    global yolov7_topk_all, yolov7_iou_thres, yolov7_conf_thres
    parser = argparse.ArgumentParser(description='Run YOLOv7 Training Script with Custom Dataset')
    parser.add_argument('--dataset-ids', type=str, nargs='+', default=dataset_ids, help='One or more Dataset IDs from the data source')
    parser.add_argument('--project-path', type=str, default=project_path, help='Base path for the project')
    parser.add_argument('--project-version', type=str, default=project_version, help='Version identifier for the dataset')
    parser.add_argument('--training-data-percentage', type=float, default=training_data_percentage, help='Percentage of data used for training (remainder used for validation)')
    
    # Arguments for YOLOv7 training
    parser.add_argument('--epochs', type=int, default=yolov7_epochs, help='Number of training epochs')
    parser.add_argument('--workers', type=int, default=yolov7_workers, help='Number of workers for data loading')
    parser.add_argument('--device', type=int, default=yolov7_device, help='Device ID for training (e.g., 0 for first GPU)')
    parser.add_argument('--batch-size', type=int, default=yolov7_batch_size, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=yolov7_img_size, help='Image size for training (width and height)')
    parser.add_argument('--weights', type=str, default=yolov7_weights, help='Path to the weights file for YOLOv7')
    parser.add_argument('--hyp', type=str, default=yolov7_hyper, help='Path to the hyperparameter file')

    # Arguments for ONNX conversion
    parser.add_argument('--topk-all', type=int, default=yolov7_topk_all, help='Top-K detections to keep for all classes')
    parser.add_argument('--iou-thres', type=float, default=yolov7_iou_thres, help='IoU threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=yolov7_conf_thres, help='Confidence threshold for predictions')

    opt = parser.parse_args()

    dataset_ids = opt.dataset_ids
    project_path = opt.project_path
    project_version = opt.project_version
    training_data_percentage = opt.training_data_percentage
    yolov7_epochs = opt.epochs
    yolov7_workers = opt.workers
    yolov7_device = opt.device
    yolov7_batch_size = opt.batch_size
    yolov7_img_size = opt.img_size
    yolov7_weights = opt.weights
    yolov7_hyper = opt.hyp
    yolov7_topk_all = opt.topk_all
    yolov7_iou_thres = opt.iou_thres
    yolov7_conf_thres = opt.conf_thres



def setup_project_directory_and_derived_variables():
    # we reset these variables here according to user input parameters
    global project_raw_data_path, project_data_coco_path, train_run_path, train_result_path, trained_best_model_path, yolov7_name
    project_raw_data_path = os.path.join(project_path, "Data_xml", project_version)
    project_data_coco_path = os.path.join(project_path, "DataCOCO", project_version)
    yolov7_name = project_version + '_' + ''.join(str(random.randint(0, 9)) for _ in range(6))  # append a random 6 digit number to avoid name duplication
    train_run_path = os.path.join(YOLOV7_PATH, "runs", "train", yolov7_name)
    train_result_path = os.path.join(train_run_path, "results.txt")
    trained_best_model_path = os.path.join(train_run_path, "weights", "best.pt")

    if os.path.exists(project_raw_data_path):
        shutil.rmtree(project_raw_data_path)
    os.makedirs(project_raw_data_path)
    logging.info(f"Raw data will be stored at {project_raw_data_path}.")
    images_path = os.path.join(project_raw_data_path, 'images')
    xmls_path = os.path.join(project_raw_data_path, 'xmls')
    os.makedirs(images_path)
    os.makedirs(xmls_path)
    if os.path.exists(project_data_coco_path):
        shutil.rmtree(project_data_coco_path)
    os.makedirs(project_data_coco_path)
    logging.info(f"Processed data for Yolov7 will be stored at {project_data_coco_path}.")



def download_and_extract_datasets():
    """
    Downloads and extracts multiple datasets in parallel using their IDs.

    This function initiates parallel download and extraction tasks for a list of dataset IDs.
    It uses ThreadPoolExecutor to manage concurrent tasks. If any of the downloads fail, 
    the function logs an error and stops the execution of the program.

    It will wait for all tasks to complete before proceeding.

    Raises:
    SystemExit: If any of the dataset downloads fail, causing the program to exit.
    """
    logging.info(f"Downloading datasets.")
    # Start the progress printing thread
    progress_thread = threading.Thread(target=print_progress)
    progress_thread.start()
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(download_and_extract_dataset, dataset_id): dataset_id for dataset_id in dataset_ids}
        for future in as_completed(futures):  # wait for all futures to complete
            try:
                future.result()
            except DownloadError as e:
                logging.error(e)
                sys.exit(1)

    progress_thread.join()
    print()  # print a newline character to ensure that the next log will not appear on the same line


def process_dataset():
    logging.info("Processing dataset for YOLOv7 training...")
    dirs = setup_yolov7_data_directories(project_data_coco_path)
    image_dir = os.path.join(project_raw_data_path, "images")
    annotation_dir = os.path.join(project_raw_data_path, 'xmls')
    classes = extract_classes(annotation_dir)
    class_map = {cls: i for i, cls in enumerate(classes)}
    create_data_yaml_file(project_data_coco_path, classes)
    create_cfg_yaml_file(os.path.join(YOLOV7_PATH, 'cfg', 'training', 'yolov7-tiny.yaml'), 
                        os.path.join(project_data_coco_path, 'cfg.yaml'), classes)
    split_and_write_labels(image_dir, annotation_dir, dirs, class_map)


def run_yolov7_training():
    logging.info("Starting YOLOv7 training...")
    # Command construction
    command = [
        'python3', 'train.py',
        '--workers', str(yolov7_workers),
        '--device', str(yolov7_device),
        '--batch-size', str(yolov7_batch_size),
        '--data', os.path.join(project_data_coco_path, 'data.yaml'),
        '--img', str(yolov7_img_size), str(yolov7_img_size),
        '--cfg', os.path.join(project_data_coco_path, 'cfg.yaml'),
        '--weights', str(yolov7_weights),
        '--name', str(yolov7_name),
        '--epochs', str(yolov7_epochs),
        '--hyp', str(yolov7_hyper)
    ]
    try:
        logging.info(f"Executing {command}")
        subprocess.Popen(command, cwd=YOLOV7_PATH)
    except subprocess.CalledProcessError as e:
        logging.info(f"Error during YOLOv7 training: {e}")


def run_yolov7_convert():
    logging.info("Starting YOLOv7 ONNX model conversion...")
    # Command construction
    command = [
        'python3', 'export.py',
        '--weights', str(trained_best_model_path),
        '--img-size', str(yolov7_img_size), str(yolov7_img_size),
        '--max-wh', str(yolov7_img_size),
        '--iou-thres', str(yolov7_iou_thres),
        '--conf-thres', str(yolov7_conf_thres),
        '--topk-all', str(yolov7_topk_all)
    ]

    command.append('--grid')
    command.append('--end2end')
    command.append('--simplify')

    try:
        logging.info(f"Executing {command}")
        subprocess.run(command, check=True, cwd=YOLOV7_PATH)
        logging.info("YOLOv7 ONNX model conversion completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during YOLOv7 ONNX model conversion: {e}")


if __name__ == "__main__":
    parse_arguments()
    setup_project_directory_and_derived_variables()
    download_and_extract_datasets()
    process_dataset()
    run_yolov7_training()
    # Now instead of waiting for run_yolov7_training() to finish, use a pooling mechanism to check if training has finished.
    # Why do we need to do that? Because at the end of run_yolov7_training, wandb opens thread to upload result to Google which might fail due to network error.
    polling_interval = 5
    while True:
        # we determine training finished by checking results.txt & making sure best.pt model file exists
        if os.path.isfile(trained_best_model_path) and count_lines(train_result_path) == yolov7_epochs:
            logging.info(f"File found: {trained_best_model_path}.")
            run_yolov7_convert()
            sys.exit(0)
        time.sleep(polling_interval)
    
