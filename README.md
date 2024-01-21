# YOLOv7 Model Training

First ssh onto the GPU host (currently 122.9.48.13).

Example command to invoke training

```shell
docker exec -it yolov7_pytorch_devel python3 auto_train.py --dataset_id 667430116996943872 --project_path /home/workspace/BatteryDataSet --project_version 14_defect_5_JiinChuan_V7
```

and the trained model in ONNX will be available at e.g.
```
/home/workspace/yolov7/runs/train/14_defect_5_JiinChuan_auto_script_test/weights/best.onnx
```
The .onnx file location will also be displayed at command line.

For the complete set of parameters you can pass in, see below
```
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
```

## Where to get dataset_ids?
<img width="1413" alt="image" src="https://github.com/BumbleBee-BBStream/yolov7/assets/708337/8bc06c9d-31f3-495c-a0c8-e3edcde5d0dc">

