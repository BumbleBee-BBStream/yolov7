import argparse
import logging
import requests
import os
import shutil
import yaml
import zipfile
import xml.etree.ElementTree as ET
import random
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## -- variables --
dataset_id = "667430116996943872"  # dataset id from http://www.deepvision-tech.cn:8090/Home/Index 
project_path = "/home/workspace/BatteryDataSet"
project_version = "14_defect_5_JiinChuan_auto_script_test"
training_data_percentage = 0.75  # how much % of data is used for training, with the rest as validation
yolov7_epochs = 100
yolov7_workers = 4
yolov7_device = 0
yolov7_batch_size = 32
yolov7_img_size = 640
yolov7_weights = "/home/workspace/yolo_weights/yolov7-tiny.pt"  # weights for the yolov7 model
yolov7_name = project_version
yolov7_hyper = "data/hyp.scratch.tiny.yaml"  # hyper parameters for the yolov7 model
yolov7_iou_thres = 0.65  # parameter for ONNX conversion
yolov7_conf_thres = 0.35
yolov7_topk_all = 100

## -- constants that are less likely to change --
YOLOV7_PATH = "/home/workspace/yolov7"  # yolov7 path, this is also the path under which this script resides
DATASRC_HOST = "http://www.deepvision-tech.cn:8090/"  # labeling server host address
DOWNLOAD_URI = DATASRC_HOST + "TaskManage/TaskData/ExportData"  # URI endpoint for dataset download

## -- derived variables (do not change manually) --
project_raw_data_path = os.path.join(project_path, "Data_xml", project_version)
project_data_coco_path = os.path.join(project_path, "DataCOCO", project_version)
trained_best_model_path = os.path.join(YOLOV7_PATH, "runs", "train", project_version, "weights", "best.pt")


## -- helper functions --
def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_length = int(r.headers.get('content-length', 0))
        chunk_size = 1024  # 1 Kibibyte
        downloaded = 0

        logging.info(f"Downloading {filename}")

        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    percentage = 100 * downloaded / total_length
                    print(f"\rProgress: {percentage:.2f}%", end='', flush=True)

        logging.info("\nDownload completed")

def unzip_and_organize(zip_file):
    # Create directories for images and xmls if they don't exist
    images_path = os.path.join(project_raw_data_path, 'images')
    xmls_path = os.path.join(project_raw_data_path, 'xmls')
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(xmls_path):
        os.makedirs(xmls_path)

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
    with open(filename, 'w') as f:
        for class_id, (xmin, xmax, ymin, ymax) in boxes:
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def split_and_write_labels(image_dir, annotation_dir, dirs, class_map):
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


## -- main functions --
def parse_arguments():
    global dataset_id, project_path, project_version, training_data_percentage
    global yolov7_epochs, yolov7_workers, yolov7_device, yolov7_batch_size, yolov7_img_size, yolov7_weights, yolov7_name, yolov7_hyper
    global yolov7_topk_all, yolov7_iou_thres, yolov7_conf_thres
    parser = argparse.ArgumentParser(description='Run YOLOv7 Training Script with Custom Dataset')
    parser.add_argument('--dataset_id', type=str, default=dataset_id, help='Dataset ID from the data source')
    parser.add_argument('--project_path', type=str, default=project_path, help='Base path for the project')
    parser.add_argument('--project_version', type=str, default=project_version, help='Version identifier for the dataset')
    parser.add_argument('--training_data_percentage', type=float, default=training_data_percentage, help='Percentage of data used for training (remainder used for validation)')
    
    # Arguments for YOLOv7 training
    parser.add_argument('--epochs', type=int, default=yolov7_epochs, help='Number of training epochs')
    parser.add_argument('--workers', type=int, default=yolov7_workers, help='Number of workers for data loading')
    parser.add_argument('--device', type=int, default=yolov7_device, help='Device ID for training (e.g., 0 for first GPU)')
    parser.add_argument('--batch-size', type=int, default=yolov7_batch_size, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=yolov7_img_size, help='Image size for training (width and height)')
    parser.add_argument('--weights', type=str, default=yolov7_weights, help='Path to the weights file for YOLOv7')
    parser.add_argument('--name', type=str, default=yolov7_name, help='Name for the training run')
    parser.add_argument('--hyp', type=str, default=yolov7_hyper, help='Path to the hyperparameter file')

    # Arguments for ONNX conversion
    parser.add_argument('--topk-all', type=int, default=yolov7_topk_all, help='Top-K detections to keep for all classes')
    parser.add_argument('--iou-thres', type=float, default=yolov7_iou_thres, help='IoU threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=yolov7_conf_thres, help='Confidence threshold for predictions')

    opt = parser.parse_args()

    dataset_id = opt.dataset_id
    project_path = opt.project_path
    project_version = opt.project_version
    training_data_percentage = opt.training_data_percentage
    yolov7_epochs = opt.epochs
    yolov7_workers = opt.workers
    yolov7_device = opt.device
    yolov7_batch_size = opt.batch_size
    yolov7_img_size = opt.img_size
    yolov7_weights = opt.weights
    yolov7_name = opt.name
    yolov7_hyper = opt.hyp
    yolov7_topk_all = opt.topk_all
    yolov7_iou_thres = opt.iou_thres
    yolov7_conf_thres = opt.conf_thres



def setup_project_directory():
    if os.path.exists(project_raw_data_path):
        shutil.rmtree(project_raw_data_path)
    os.makedirs(project_raw_data_path)
    logging.info(f"Raw data will be stored at {project_raw_data_path}.")
    if os.path.exists(project_data_coco_path):
        shutil.rmtree(project_data_coco_path)
    os.makedirs(project_data_coco_path)
    logging.info(f"Processed data for Yolov7 will be stored at {project_data_coco_path}.")


def download_and_extract_dataset():
    payload = {"id": dataset_id, "type": 1}
    response = requests.post(DOWNLOAD_URI, data=payload)
    if response.status_code == 200:
        response_json = response.json()
        file_url_suffix = response_json.get("Data")
        if file_url_suffix:
            file_url = DATASRC_HOST + file_url_suffix
            file_name = file_url_suffix.split('/')[-1]
            file_path = os.path.join(project_raw_data_path, file_name)
            download_file(file_url, file_path)
            unzip_and_organize(file_path)
        else:
            logging.error(f"Data field not found in the response. Check {DATASRC_HOST}.")
    else:
        logging.error(f"POST request failed. Check {DATASRC_HOST}.")


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
    # TODO: Handle the scenario where wandb uploading thread does not finish properly.
    logging.info("Starting YOLOv7 training...")
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
        subprocess.run(command, check=True, cwd=YOLOV7_PATH)
        logging.info("YOLOv7 training completed successfully.")
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
    setup_project_directory()
    download_and_extract_dataset()
    process_dataset()
    run_yolov7_training()
    run_yolov7_convert()



