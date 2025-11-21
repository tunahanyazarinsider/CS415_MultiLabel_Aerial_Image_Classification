# Related Work and References for Multi-Label Aerial Image Classification

This document provides a comprehensive list of relevant research papers organized by topic area, suitable for the Phase 2 report literature review section.

---

## 1. Foundational Dataset Papers

### 1.1 AID Dataset (Base Dataset)

**[REQUIRED - Dataset Foundation]**

**Title:** AID: A Benchmark Dataset for Performance Evaluation of Aerial Scene Classification

**Authors:** Xia, G.-S., Hu, J., Hu, F., Shi, B., Bai, X., Zhong, Y., Zhang, L., & Lu, X.

**Published:** IEEE Transactions on Geoscience and Remote Sensing, Vol. 55, No. 7, pp. 3965-3981, July 2017

**DOI/Link:**
- IEEE Xplore: https://ieeexplore.ieee.org/document/7907303/
- arXiv: https://arxiv.org/abs/1608.05167
- Official Website: https://captain-whu.github.io/AID/

**Key Contributions:**
- Introduced the Aerial Image Dataset (AID) with 10,000 images across 30 scene categories
- Images collected from Google Earth imagery at 600×600 pixel resolution
- Established benchmark for aerial scene classification tasks
- Provides diverse scenes including airports, residential areas, forests, industrial zones, etc.

**Why Important:** This is the foundational dataset from which the AID_MultiLabel dataset was derived. Essential to cite as the source of imagery.

---

### 1.2 AID Multi-Label Dataset (Your Dataset)

**[REQUIRED - Your Specific Task]**

**Title:** Relation Network for Multi-label Aerial Image Classification

**Authors:** Hua, Y., Mou, L., & Zhu, X. X.

**Published:** IEEE Transactions on Geoscience and Remote Sensing, Vol. 58, No. 7, pp. 4558-4572, July 2020

**DOI/Link:**
- arXiv: https://arxiv.org/abs/1907.07274
- IEEE Xplore: https://ieeexplore.ieee.org/document/8986556/
- Dataset GitHub: https://github.com/Hua-YS/AID-Multilabel-Dataset

**Key Contributions:**
- Created the AID multi-label dataset with 3,000 images and 17 object labels
- Proposed attention-aware label relational reasoning network
- Three modules: label-wise feature parcel learning, attentional region extraction, label relational inference
- Demonstrated importance of modeling label relationships in aerial images
- Made dataset publicly available

**Why Important:** This paper created the exact dataset you're using and proposed a state-of-the-art method for this task. MUST cite as the dataset source and as related work.

---

**Title:** Label Relation Inference for Multi-Label Aerial Image Classification

**Authors:** Hua, Y., Mou, L., & Zhu, X. X.

**Published:** IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2019

**DOI/Link:**
- IEEE Xplore: https://ieeexplore.ieee.org/document/8898934/
- PDF: https://elib.dlr.de/134105/1/2019_IGARSS.pdf

**Key Contributions:**
- Earlier conference version of the above work
- Introduced the concept of label relation inference for aerial images
- Evaluated on UCM and AID multi-label datasets

**Why Important:** Complements the journal paper above; shows progression of research.

---

## 2. Multi-Label Classification - General Methods

### 2.1 Deep Learning for Multi-Label Classification

**[RECOMMENDED - Survey Paper]**

**Title:** Deep Learning for Multi-Label Learning: A Comprehensive Survey

**Authors:** Multiple authors

**Published:** arXiv preprint, 2024

**Link:** https://arxiv.org/abs/2401.16549

**Key Contributions:**
- Comprehensive survey consolidating existing research in deep learning for multi-label classification
- Covers CNNs, transformers, autoencoders, and recurrent architectures
- Comparative analysis with insights for future research

**Why Important:** Excellent survey paper to establish context for multi-label classification landscape.

---

### 2.2 CNN-RNN Framework

**[RECOMMENDED]**

**Title:** CNN-RNN: A Unified Framework for Multi-label Image Classification

**Authors:** Wang, J., Yang, Y., Mao, J., Huang, Z., Huang, C., & Xu, W.

**Published:** IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

**DOI/Link:**
- IEEE Xplore: https://ieeexplore.ieee.org/document/7780620/
- PDF: https://openaccess.thecvf.com/content_cvpr_2016/papers/Wang_CNN-RNN_A_Unified_CVPR_2016_paper.pdf

**Key Contributions:**
- Unified framework combining CNN features with RNN for label dependency modeling
- Learns joint image-label embedding
- Characterizes semantic label dependency and image-label relevance
- End-to-end trainable from scratch

**Why Important:** Influential early work on modeling label dependencies using deep learning.

---

### 2.3 Graph-Based Multi-Label Classification

**[HIGHLY RECOMMENDED - State-of-the-Art]**

**Title:** Multi-Label Image Recognition with Graph Convolutional Networks

**Authors:** Chen, Z.-M., Wei, X.-S., Wang, P., & Guo, Y.

**Published:** IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019

**DOI/Link:**
- arXiv: https://arxiv.org/abs/1904.03582
- CVPR: https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Multi-Label_Image_Recognition_With_Graph_Convolutional_Networks_CVPR_2019_paper.pdf
- GitHub: https://github.com/megvii-research/ML-GCN

**Key Contributions:**
- Introduced ML-GCN (Multi-Label Graph Convolutional Network)
- Builds directed graph over object labels with word embeddings
- Novel re-weighted scheme for effective label correlation matrix
- GCN learns inter-dependent object classifiers
- Maintains meaningful semantic topology

**Why Important:** One of the most influential papers on using GCNs for multi-label classification. Excellent for "advanced techniques" section.

---

## 3. Transfer Learning and Pre-trained Models

### 3.1 ResNet Architecture

**[REQUIRED - Foundation Model]**

**Title:** Deep Residual Learning for Image Recognition

**Authors:** He, K., Zhang, X., Ren, S., & Sun, J.

**Published:** IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

**Key Contributions:**
- Introduced residual connections to enable very deep networks
- ResNet-50, ResNet-101, ResNet-152 architectures
- Won ILSVRC 2015 classification competition
- Enables training of networks with 100+ layers

**Why Important:** ResNet is your baseline backbone. MUST cite as the architecture foundation.

---

### 3.2 Transfer Learning for Remote Sensing

**[RECOMMENDED]**

**Title:** Satellite and Scene Image Classification Based on Transfer Learning and Fine Tuning of ResNet50

**Authors:** Shabbir, A., Ali, N., Ahmed, J., et al.

**Published:** Mathematical Problems in Engineering, 2021

**DOI/Link:** https://onlinelibrary.wiley.com/doi/10.1155/2021/5843816

**Key Contributions:**
- Fine-tuned ResNet50 for satellite and scene classification
- Network surgery and creation of network head
- Hyperparameter optimization for remote sensing

**Why Important:** Validates use of ResNet50 transfer learning specifically for satellite/aerial imagery.

---

**Title:** Deep Transfer Learning with ResNet for Remote Sensing Scene Classification

**Authors:** Multiple authors

**Published:** IEEE Conference Publication, 2022

**Link:** https://ieeexplore.ieee.org/document/9915967/

**Key Contributions:**
- Applied ResNet50 with multiple optimizers (Adam, SGDM, RMSProp)
- Achieved 95.8% accuracy on RSI-CB128 dataset
- Demonstrates effectiveness of transfer learning for remote sensing

**Why Important:** Recent validation of ResNet transfer learning for remote sensing.

---

**Title:** Transfer Learning in Environmental Remote Sensing

**Authors:** Multiple authors

**Published:** Remote Sensing of Environment, 2023

**Link:** https://www.sciencedirect.com/science/article/abs/pii/S0034425723004765

**Key Contributions:**
- Comprehensive review of transfer learning in remote sensing
- Discusses domain adaptation challenges
- Shows importance of fine-tuning for domain shift

**Why Important:** Addresses challenges of applying ImageNet pre-trained models to remote sensing data.

---

**Title:** Remote Sensing Image Classification: A Comprehensive Review and Applications

**Authors:** Mehmood, M., et al.

**Published:** Mathematical Problems in Engineering, 2022

**Link:** https://onlinelibrary.wiley.com/doi/10.1155/2022/5880959

**Key Contributions:**
- Comprehensive review of deep learning methods for remote sensing
- ResNet-50 most frequently used (15 times), followed by VGG-16 (12 times)
- Analysis of backbone selection trends

**Why Important:** Survey establishing ResNet as standard backbone for remote sensing.

---

## 4. Loss Functions for Multi-Label Classification

### 4.1 Binary Cross-Entropy Loss

**[REQUIRED - Your Loss Function]**

**General Reference:** Standard multi-label classification literature

**Key Points:**
- BCE treats each label as independent binary classification
- Combines sigmoid activation with cross-entropy
- Suitable for non-mutually exclusive labels
- Formula: L = -[y·log(p) + (1-y)·log(1-p)]

**Why Important:** This is the loss function you're using. Explain why it's appropriate for multi-label tasks.

---

### 4.2 Focal Loss for Class Imbalance

**[HIGHLY RECOMMENDED]**

**Title:** Focal Loss for Dense Object Detection

**Authors:** Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P.

**Published:** IEEE International Conference on Computer Vision (ICCV), 2017

**Key Contributions:**
- Introduced focal loss to address class imbalance
- Down-weights easy examples, focuses on hard examples
- Modulating factor: (1 - p_t)^γ
- Originally for object detection, widely adopted for classification

**Why Important:** Key paper for handling class imbalance. Recommended for future improvements.

---

**Title:** Focal Loss Improves the Model Performance on Multi-Label Image Classifications with Imbalanced Data

**Authors:** Multiple authors

**Published:** ACM Conference Proceedings, 2020

**Link:** https://dl.acm.org/doi/10.1145/3411016.3411020

**Key Contributions:**
- Validates focal loss effectiveness for multi-label classification
- Addresses extreme class imbalance
- Improves CNN performance on imbalanced datasets

**Why Important:** Directly applies focal loss to multi-label classification with imbalance.

---

## 5. Attention Mechanisms

### 5.1 Squeeze-and-Excitation Networks (SENet)

**[RECOMMENDED]**

**Title:** Squeeze-and-Excitation Networks

**Authors:** Hu, J., Shen, L., & Sun, G.

**Published:** IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018

**Key Contributions:**
- Introduced channel attention mechanism
- Squeeze: global information aggregation
- Excitation: channel-wise reweighting
- Won ILSVRC 2017 classification competition
- 2-3% accuracy improvement over vanilla ResNet

**Why Important:** Foundation of channel attention. Can be integrated into ResNet backbone.

---

### 5.2 CBAM (Convolutional Block Attention Module)

**[HIGHLY RECOMMENDED]**

**Title:** CBAM: Convolutional Block Attention Module

**Authors:** Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S.

**Published:** European Conference on Computer Vision (ECCV), 2018

**DOI/Link:** https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf

**Key Contributions:**
- Combines channel attention AND spatial attention
- Two sequential modules: channel attention → spatial attention
- Uses both GAP and GMP (global average/max pooling)
- Lightweight: only 15% additional computation
- 3% accuracy improvement over standalone CNN

**Why Important:** CBAM is more comprehensive than SENet (channel + spatial). Excellent for Approach 2 improvements.

---

## 6. Data Augmentation for Remote Sensing

### 6.1 Survey Paper

**[RECOMMENDED]**

**Title:** A Review of Data Augmentation Methods of Remote Sensing Image Target Recognition

**Authors:** Multiple authors

**Published:** Remote Sensing, Vol. 15, No. 3, 2023

**Link:** https://www.mdpi.com/2072-4292/15/3/827

**Key Contributions:**
- Comprehensive review of augmentation methods for remote sensing
- Traditional geometric transformations (rotation, flip)
- Noise-based augmentation (Gaussian, salt-and-pepper, speckle)
- Generative models (GANs, diffusion models)
- AutoAugment techniques

**Why Important:** Validates your augmentation strategy for aerial images.

---

**Title:** A Comparison of Data Augmentation Techniques in Training Deep Neural Networks for Satellite Image Classification

**Authors:** Multiple authors

**Published:** ResearchGate, 2020

**Link:** https://www.researchgate.net/publication/340294990

**Key Contributions:**
- Empirical comparison of augmentation techniques
- Rotation invariance crucial for aerial images
- Color jitter improves robustness

**Why Important:** Empirical validation of augmentation choices for satellite imagery.

---

**Title:** Image Augmentation for Satellite Images

**Authors:** Multiple authors

**Published:** arXiv, 2022

**Link:** https://arxiv.org/abs/2207.14580

**Key Contributions:**
- Specialized augmentation techniques for satellite imagery
- Addresses unique characteristics of overhead imagery
- Rotation-invariant transformations

**Why Important:** Specific to satellite/aerial image augmentation.

---

## 7. Evaluation Metrics for Multi-Label Classification

### 7.1 Comprehensive Metrics Guide

**General References:**

Multiple sources from scikit-learn documentation and academic papers establish standard metrics:

**Key Metrics:**

1. **Hamming Loss**
   - Fraction of wrong labels to total labels
   - Accounts for prediction error and missing error
   - Lower is better (ideal = 0)

2. **Subset Accuracy**
   - Exact match of all labels
   - Strictest metric
   - Percentage of samples with perfect predictions

3. **F1 Score (Micro/Macro/Weighted)**
   - Micro: Aggregate all classes, then compute (better for imbalanced)
   - Macro: Compute per-class, then average (treats all classes equally)
   - Weighted: Weight by class frequency

4. **Precision and Recall**
   - Micro/macro/weighted averaging
   - Per-class analysis

5. **Mean Average Precision (mAP)**
   - Standard for multi-label evaluation
   - Ranking-based metric

**Why Important:** Essential for explaining your evaluation methodology.

---

**Title:** Evaluating Multi-label Classifiers

**Published:** Towards Data Science (online resource)

**Link:** https://towardsdatascience.com/evaluating-multi-label-classifiers-a31be83da6ea/

**Key Contributions:**
- Practical guide to multi-label metrics
- Clear explanations with examples
- Comparison of different averaging strategies

**Why Important:** Accessible explanation of why accuracy is misleading for multi-label tasks.

---

## 8. Additional Relevant Topics

### 8.1 EfficientNet (Alternative Backbone)

**Title:** EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

**Authors:** Tan, M., & Le, Q. V.

**Published:** International Conference on Machine Learning (ICML), 2019

**Key Contributions:**
- Compound scaling method (width, depth, resolution)
- Better accuracy/efficiency trade-off than ResNet
- EfficientNet-B0 to B7 variants

**Why Important:** Potential improvement over ResNet baseline.

---

### 8.2 Vision Transformers

**Title:** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

**Authors:** Dosovitskiy, A., et al.

**Published:** International Conference on Learning Representations (ICLR), 2021

**Key Contributions:**
- Applied transformers to image classification
- State-of-the-art on many benchmarks
- Pre-trained on large datasets

**Why Important:** Cutting-edge alternative to CNNs.

---

### 8.3 Grad-CAM for Interpretability

**Title:** Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization

**Authors:** Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.

**Published:** IEEE International Conference on Computer Vision (ICCV), 2017

**Key Contributions:**
- Produces visual explanations for CNN decisions
- Gradient-weighted class activation mapping
- Highlights discriminative regions
- Works with any CNN architecture

**Why Important:** Essential for model interpretability and understanding predictions.

---

## 9. Suggested Citation Categorization for Your Report

### Essential Citations (MUST include):
1. Xia et al., 2017 - AID Dataset
2. Hua et al., 2019/2020 - AID Multi-Label Dataset and Relation Network
3. He et al., 2016 - ResNet Architecture

### Strongly Recommended (Core methodology):
4. Lin et al., 2017 - Focal Loss (if discussing class imbalance)
5. Chen et al., 2019 - ML-GCN (if discussing label correlations)
6. Woo et al., 2018 - CBAM (if using attention)
7. Multi-label metrics papers/resources

### Supporting Citations (Strengthen literature review):
8. Transfer learning for remote sensing papers
9. Data augmentation for aerial imagery papers
10. Multi-label classification survey papers
11. SENet, EfficientNet (if comparing architectures)

---

## 10. How to Organize Literature Review in Report

### Suggested Structure:

**Section 1: Problem Domain**
- Xia et al., 2017 (AID dataset)
- Hua et al., 2019/2020 (AID multi-label dataset)
- Importance of multi-label classification for aerial images

**Section 2: Multi-Label Classification Methods**
- General deep learning for multi-label (survey papers)
- CNN-RNN approaches (Wang et al., 2016)
- Graph-based methods (Chen et al., 2019 - ML-GCN)
- Label dependency modeling

**Section 3: Transfer Learning for Remote Sensing**
- He et al., 2016 (ResNet)
- Transfer learning papers (Shabbir et al., 2021)
- Domain adaptation challenges

**Section 4: Addressing Key Challenges**
- Class imbalance: Focal loss (Lin et al., 2017)
- Attention mechanisms: SENet, CBAM
- Data augmentation for aerial images

**Section 5: Evaluation Methodologies**
- Multi-label metrics (Hamming loss, F1, mAP)
- Why accuracy is insufficient

---

## 11. BibTeX Entries (Sample Format)

```bibtex
@article{xia2017aid,
  title={AID: A benchmark data set for performance evaluation of aerial scene classification},
  author={Xia, Gui-Song and Hu, Jingwen and Hu, Fan and Shi, Baoguang and Bai, Xiang and Zhong, Yanfei and Zhang, Liangpei and Lu, Xiaoqiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={55},
  number={7},
  pages={3965--3981},
  year={2017},
  publisher={IEEE}
}

@article{hua2020relation,
  title={Relation network for multilabel aerial image classification},
  author={Hua, Yuansheng and Mou, Lichao and Zhu, Xiao Xiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={58},
  number={7},
  pages={4558--4572},
  year={2020},
  publisher={IEEE}
}

@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}

@inproceedings{chen2019multi,
  title={Multi-label image recognition with graph convolutional networks},
  author={Chen, Zhao-Min and Wei, Xiu-Shen and Wang, Peng and Guo, Yanwen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5177--5186},
  year={2019}
}

@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2980--2988},
  year={2017}
}

@inproceedings{woo2018cbam,
  title={Cbam: Convolutional block attention module},
  author={Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and Kweon, In So},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={3--19},
  year={2018}
}
```

---

## 12. Quick Reference Summary

| Topic | Key Paper(s) | Why Cite |
|-------|-------------|----------|
| **Base Dataset** | Xia et al., 2017 | Source of AID imagery |
| **Your Dataset** | Hua et al., 2019/2020 | Multi-label dataset creation |
| **Architecture** | He et al., 2016 | ResNet foundation |
| **Multi-Label SOTA** | Chen et al., 2019 | ML-GCN with graphs |
| **Class Imbalance** | Lin et al., 2017 | Focal loss |
| **Attention** | Woo et al., 2018 | CBAM mechanism |
| **Transfer Learning** | Multiple RS papers | Validation for remote sensing |
| **Metrics** | Standard ML literature | Evaluation methodology |

---

**Total Recommended Papers:** 15-20 core papers + additional supporting references

**Report Strategy:**
- Introduction/Motivation: 3-4 papers
- Related Work section: 10-15 papers
- Methodology justification: 5-8 papers
- Throughout discussion: cite relevant papers

This gives you a strong, comprehensive literature review that positions your work within the current research landscape while justifying your methodological choices.
