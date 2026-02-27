# Deep Learning Homework (Lab3-7)

課程作業紀錄 - 深度學習各項應用實踐

## 📁 資料夾結構

```
DL/
├── Lab3/      Semi-Supervised Learning on Flower Classification
├── Lab4/      Semantic Segmentation on BCSS Dataset
├── Lab5/      Object Detection on PASCAL VOC
├── Lab6/      Seq2Seq Model Training on SQuAD
├── Lab7/      Deep Generative Image Synthesis
└── data/      Datasets
```

## 📝 作業說明

### Lab3: Semi-Supervised Flower Classification
- **主題**: 半監督學習 (Self-Training)
- **模型**: ResNet18 (從零實作)
- **資料集**: Flowers Recognition Dataset (5 classes, ~4262 images)
- **方法**: 監督學習 + 自訓練 (Pseudo-labeling)
- **作業檔案**: [HW3_314831024.ipynb](Lab3/HW3_314831024.ipynb)
- **Kaggle**: https://www.kaggle.com/competitions/lab-3-flower-classification/overview

### Lab4: Semantic Segmentation on BCSS
- **主題**: 語義分割 (Semantic Segmentation)
- **模型**: 分割網路架構實作
- **資料集**: BCSS (Building Change Semantic Segmentation)
- **作業檔案**: [HW4_314831024.ipynb](Lab4/HW4_314831024.ipynb)
- **Kaggle**: https://www.kaggle.com/competitions/lab-4-semantic-segmentation-on-bcss-639003

### Lab5: Object Detection on PASCAL VOC
- **主題**: 物體偵測 (Object Detection)
- **模型**: YOLO / 自訓練檢測網路
- **資料集**: PASCAL VOC Dataset
- **作業檔案**: [v3_mixup_CONVNEXT_large_multiscale_last.ipynb](Lab5/v3_mixup_CONVNEXT_large_multiscale_last.ipynb)
- **Kaggle**: https://www.kaggle.com/competitions/lab-5-object-detection-on-pascal-voc-639003

### Lab6: Seq2Seq Model on SQuAD
- **主題**: 序列到序列模型 (Sequence-to-Sequence)
- **模型**: Transformer / 注意力機制
- **資料集**: SQuAD (Stanford Question Answering Dataset)
- **作業檔案**: [LAB6.ipynb](Lab6/LAB6.ipynb)
- **Kaggle**: https://www.kaggle.com/competitions/lab-6-training-a-seq-2-seq-model-on-s-qu-ad-639003

### Lab7: Deep Generative Image Synthesis
- **主題**: 深度生成模型 (GAN / VAE)
- **模型**: 生成對抗網路 / 變分自編碼器
- **資料集**: Flowers 102 Dataset
- **作業檔案**: [Lab7_314831024.ipynb](Lab7/Lab7_314831024.ipynb)
- **Kaggle**: https://www.kaggle.com/competitions/lab-7-deep-generative-image-synthesis-639003

## 📥 資料集說明

各個 Lab 所需的資料集通常需要自行從 Kaggle 競賽頁面下載：

1. **Lab3**: 在 Lab3 Kaggle 頁面下載，放入 `Lab3/Lab3_data_flower_2025/`
2. **Lab4**: 在 Lab4 Kaggle 頁面下載 BCSS 資料集
3. **Lab5**: 在 Lab5 Kaggle 頁面下載 PASCAL VOC 資料集
4. **Lab6**: 在 Lab6 Kaggle 頁面下載 SQuAD 資料集
5. **Lab7**: 在 Lab7 Kaggle 頁面下載花朵資料集


## 📊 作業提交

所有作業都在對應的 Kaggle 競賽中提交。詳細步驟請參考各 Lab 的 notebook 說明。

## 📚 主要技術

- **深度學習框架**: PyTorch
- **模型架構**: CNN (ResNet, ConvNeXt), Transformer, GAN, VAE
- **訓練技巧**: 
  - 數據增強 (Data Augmentation)
  - 混合訓練 (Mixup, CutMix)
  - 學習率調度 (Learning Rate Scheduling)
  - 自訓練 (Self-Training for Semi-Supervised Learning)

## 👤 學生資訊

- **學號**: 314831024
- **姓名**: 李朋逸

---

**最後更新**: 2026年2月27日
