## Overview: 
Computer vision models usually need a ton of labeled data, right? But creating those labels takes a lot of time and manual effort—which just makes the whole system heavily dependent on human input again. This project is my take on solving that issue.
Instead of relying fully on labeled data, I’ve used semi-supervised learning to train a model for multi-object detection in autonomous driving. The goal is to reduce the labeling workload while still getting solid results.

## Objective: 
The primary objective is to develop a semi-supervised multi-object detection system that can: <br/>
- Detect and classify four categories: Car, Truck, Van, and Miscellaneous  <br/>
- Utilize a hybrid learning strategy combining: <br/>
- Pseudo-labeling <br/>
- Mean Teacher consistency regularization <br/>
- Contrastive learning for feature representation <br/>
- Integrate spatial attention mechanisms for enhanced interpretability and transparency of model decisions <br/>
- Provide a scalable, generalizable framework for real-world autonomous driving applications <br/>

## Data Description: 
- Dataset Used: KITTI Dataset <br/>
- Categories: Car, Truck, Van, Miscellaneous <br/>
<br/>
Data Split: <br/>
- Labeled Subset: Used to initialize and guide training <br/>
- Unlabeled Subset: Used in semi-supervised learning strategies (e.g., pseudo-labelling) <br/>
 <br/>
Preprocessing includes: <br/>
- Conversion to YOLOv5-compatible annotation format <br/>
- Image resizing and augmentation <br/>
- Filtering of low-confidence pseudo-labels <br/>

## Dependencies: 
pip install -r requirements.txt

- Python 3.8+  <br/>
- PyTorch  <br/>
- YOLOv5 (Ultralytics)  <br/>
- OpenCV  <br/>
- Scikit-learn  <br/>
- Matplotlib  <br/>

Optional: Use a GPU with CUDA support for accelerated training

## Results:

YOLOv5 fine-tuned with pseudo-labels achieved an mAP@0.5 of 0.755. Strong performance was seen in detecting Cars and Trucks, while Van and Misc remained more challenging. Validation accuracy reached 0.993 mAP@0.5. Spatial attention maps improved interpretability, though technical issues limited the integration of contrastive and consistency regularization.

