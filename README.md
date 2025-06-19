## Realtime Traffic Sign Semantic Segmentation
### Lightweight Deep Learning Pipeline for Embedded Vision and Robotics

![Live Demo](output.gif)

### 1. Overview

This project performs real-time semantic segmentation of DIY traffic signs (made using Arduino), designed for embedded systems. It combines deep learning with classical image processing for fast and efficient results.

- Architecture: UNet-style autoencoder

- Backbone: Pretrained MobileNetV2

- Postprocessing: Classical methods (contours, thresholding, morphological ops)

- Target: Custom-made Arduino traffic signs

- Use Case: Robotics, embedded navigation, sign-aware behavior control

### 2. Model Architecture

- Encoder: MobileNetV2 pretrained on ImageNet

- Decoder: Custom upsampling path with skip connections (UNet-style)

- Output: Pixel-wise class map of traffic signs (semantic mask)

Input → MobileNetV2 → UNet Decoder → Segmentation Mask → Postprocessing → Result

### 3. Dataset

- Traffic Signs: crafted signs on small boards

- Captured: Using webcam 

- Classes: Red Light, Green Light, Yellow Light, Off Light, Turn Right, Turn Left.

- Annotation: Binary masks per class (labelme)
