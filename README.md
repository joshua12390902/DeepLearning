# Deep Learning Labs (Lab3-7)

NYCU AI College Deep Learning Course - Labs 3 through 7

## Folder Structure

```
DL/
├── Lab3/      Semi-Supervised Learning on Flower Classification
├── Lab4/      Semantic Segmentation on BCSS Dataset
├── Lab5/      Object Detection on PASCAL VOC
├── Lab6/      Seq2Seq Model Training on SQuAD
├── Lab7/      Deep Generative Image Synthesis
└── data/      Datasets
```

## Competition Rankings

| Lab | Topic | Rank | Kaggle Link |
|-----|-------|------|-------------|
| Lab3 | Flower Classification | 1/32 | [Competition](https://www.kaggle.com/competitions/lab-3-flower-classification/overview) |
| Lab4 | Semantic Segmentation on BCSS | 6/32 | [Competition](https://www.kaggle.com/competitions/lab-4-semantic-segmentation-on-bcss-639003) |
| Lab5 | Object Detection on Pascal VOC | 3/32 | [Competition](https://www.kaggle.com/competitions/lab-5-object-detection-on-pascal-voc-639003) |
| Lab6 | Text Summarization with Seq2Seq Model | 7/38 | [Competition](https://www.kaggle.com/competitions/lab-6-training-a-seq-2-seq-model-on-s-qu-ad-639003) |
| Lab7 | Image Generation | 15/32 | [Competition](https://www.kaggle.com/competitions/lab-7-deep-generative-image-synthesis-639003) |

## Lab Descriptions

### Lab3: Flower Classification
- **Topic**: Semi-Supervised Learning (Self-Training)
- **Model**: ResNet18 (implemented from scratch)
- **Dataset**: Flowers Recognition Dataset (5 classes, ~4262 images)
- **Method**: Supervised Learning + Self-Training (Pseudo-labeling)
- **Notebook**: [HW3_314831024.ipynb](Lab3/HW3_314831024.ipynb)

### Lab4: Semantic Segmentation on BCSS
- **Topic**: Semantic Segmentation
- **Model**: Segmentation Network Architecture
- **Dataset**: BCSS (Building Change Semantic Segmentation)
- **Notebook**: [HW4_314831024.ipynb](Lab4/HW4_314831024.ipynb)

### Lab5: Object Detection on Pascal VOC
- **Topic**: Object Detection
- **Model**: YOLO / Custom Detection Network
- **Dataset**: PASCAL VOC Dataset
- **Notebook**: [v3_mixup_CONVNEXT_large_multiscale_last.ipynb](Lab5/v3_mixup_CONVNEXT_large_multiscale_last.ipynb)

### Lab6: Text Summarization with Seq2Seq Model
- **Topic**: Sequence-to-Sequence Model
- **Model**: Transformer / Attention Mechanism
- **Dataset**: SQuAD (Stanford Question Answering Dataset)
- **Notebook**: [LAB6.ipynb](Lab6/LAB6.ipynb)

### Lab7: Image Generation
- **Topic**: Deep Generative Models (GAN / VAE)
- **Model**: Generative Adversarial Networks / Variational Autoencoder
- **Dataset**: Flowers 102 Dataset
- **Notebook**: [Lab7_314831024.ipynb](Lab7/Lab7_314831024.ipynb)

## Dataset Information

Datasets for each lab need to be downloaded from the respective Kaggle competition pages:

1. **Lab3**: Download from Lab3 Kaggle page, place in `Lab3/Lab3_data_flower_2025/`
2. **Lab4**: Download BCSS dataset from Lab4 Kaggle page
3. **Lab5**: Download PASCAL VOC dataset from Lab5 Kaggle page
4. **Lab6**: Download SQuAD dataset from Lab6 Kaggle page
5. **Lab7**: Download Flowers dataset from Lab7 Kaggle page


## Submission

All assignments are submitted to the corresponding Kaggle competitions. Please refer to each lab's notebook for detailed instructions.

## Key Technologies

- **Deep Learning Framework**: PyTorch
- **Model Architectures**: CNN (ResNet, ConvNeXt), Transformer, GAN, VAE
- **Training Techniques**: 
  - Data Augmentation
  - Mixup, CutMix
  - Learning Rate Scheduling
  - Self-Training for Semi-Supervised Learning

## Student Information

- **Student ID**: 314831024
- **Name**: Penyi Lee

---

**Last Updated**: February 27, 2026
