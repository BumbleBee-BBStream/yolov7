# YOLOv7 Model Training

First ssh onto the GPU host (currently 122.9.48.13).

Example command to invoke training

```shell
docker exec -it yolov7_pytorch_devel python3 auto_train.py --dataset-ids 667430116996943872 667397486096158720 --project-path /home/workspace/BatteryDataSet --project-version 14_defect_5_JinChuan_V7
```

You can pass in one or more dataset-ids. To find the trained model in ONNX, look for the following line from the command line output:
```
ONNX export success, saved as /home/workspace/yolov7/runs/train/14_defect_5_JinChuan_V7_256651/weights/best.onnx
```

For the complete set of accepted parameters, see below
```
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
```
Note that the `--name` parameter for YOLOv7 training is automatically set by the script. It is set by attaching random numbers on top of `project-version`, as a way to deduplicate in case `project-version` has been used before.

## Where to get dataset_ids?
<img width="1413" alt="image" src="https://github.com/BumbleBee-BBStream/yolov7/assets/708337/8bc06c9d-31f3-495c-a0c8-e3edcde5d0dc">

