# Sarcasm Detection in Vietnamese Social Media - UIT Data Science Challenge

## Overview
This repository contains my project for the **UIT Data Science Challenge**, where I worked on **sarcasm detection in Vietnamese social media** using the **ViMMSD dataset**. The goal was to develop a **multimodal learning approach** that detects sarcasm in both text and images.

## Dataset: ViMMSD

The dataset was initially provided via email during the competition and will be publicly available afterward.

## Task Description
The challenge was to classify social media posts into one of the following categories:
1. **text-sarcasm:** Sarcasm in text only
2. **image-sarcasm:** Sarcasm in images only
3. **multi-sarcasm:** Sarcasm in both text and images
4. **not-sarcasm:** No sarcasm

## Approach
### 1. Feature Extraction
- **OCR for Text from Images**: Used **VietOCR** to extract text from images.
- **Image Features**: Employed **CLIP (openai/clip-vit-base-patch16)** to encode image attributes.
- **Text Features**: Used **PhoBERT (vinai/phobert-base-v2)** for Vietnamese text representation.

### 2. Multimodal Model Architecture
We designed a multimodal sarcasm detection model by integrating information from text, images, and OCR:

- Encoders for each modality (text, image, OCR):
  * Text: Linear → GELU → Dropout → LayerNorm
  * Image: Linear → GELU → Dropout → LayerNorm
  * OCR: Linear → GELU → Dropout → LayerNorm

- Attention mechanisms for interaction between:
  * Text ↔ Image (Multihead Attention with 8 heads)
  * Text ↔ OCR (Multihead Attention with 8 heads)

- Feature Fusion: Processed features from all modalities were combined to make the final sarcasm prediction.
  
- Neural Network Components:
  * Fully connected layers
  * GELU activation function
  * Dropout for regularization
  * LayerNorm for data normalization

### 3. Training & Evaluation
- Optimized with **AdamW optimizer** and **cross-entropy loss**.
- Evaluated using **F1-score, accuracy, and recall**.
