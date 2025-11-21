# Strategy and Approach for Multi-Label Aerial Image Classification

## Problem Analysis

### Core Challenge
Multi-label classification of aerial images where each image can belong to multiple categories simultaneously (e.g., an airport image may contain: airplanes, buildings, pavement, cars).

### Key Differences from Single-Label Classification
1. **Non-mutually exclusive labels**: Multiple labels can be active
2. **Label correlations**: Some labels frequently co-occur (harbor + water + ships)
3. **Different evaluation metrics**: Precision, recall, F1 need per-label and overall strategies
4. **Loss function**: Binary Cross-Entropy instead of Categorical Cross-Entropy

---

## Strategic Approaches (Ranked by Complexity)

### Approach 1: Transfer Learning with Pre-trained CNN (BASELINE - Recommended Start)

**Description**: Use a pre-trained CNN (ResNet, EfficientNet, Vision Transformer) and replace the final layer with a multi-label head.

**Architecture**:
```
Input Image (600x600x3)
    ↓
Pre-trained Backbone (ResNet50/EfficientNet/ViT)
    ↓
Global Average Pooling
    ↓
Fully Connected Layer(s)
    ↓
Sigmoid Activation (17 outputs, one per class)
```

**Pros**:
- Quick to implement and train
- Leverages learned features from ImageNet
- Good baseline performance
- Well-documented and stable

**Cons**:
- Treats labels independently
- Doesn't model label relationships
- May struggle with rare classes

**Expected Performance**: 70-80% F1 score

**Recommended Models**:
1. **ResNet50**: Solid baseline, well-understood
2. **EfficientNet-B3/B4**: Better accuracy/efficiency trade-off
3. **Vision Transformer (ViT)**: State-of-the-art on many vision tasks

---

### Approach 2: Multi-Label with Attention Mechanisms

**Description**: Enhance the baseline with spatial attention to focus on relevant image regions for different labels.

**Architecture**:
```
Input Image (600x600x3)
    ↓
Pre-trained Backbone
    ↓
Spatial Attention Module (CBAM/SENet)
    ↓
Class-specific Attention Maps
    ↓
Multi-label Classification Head
    ↓
Sigmoid Activation (17 outputs)
```

**Pros**:
- Learns which regions matter for which labels
- More interpretable (can visualize attention)
- Better feature discrimination

**Cons**:
- More complex to implement
- Longer training time
- Requires more hyperparameter tuning

**Expected Performance**: 75-85% F1 score

**Key Techniques**:
- CBAM (Convolutional Block Attention Module)
- SENet (Squeeze-and-Excitation Networks)
- Multi-head attention for different label groups

---

### Approach 3: Label Correlation Modeling with Graph Neural Networks

**Description**: Explicitly model relationships between labels using GNNs where each label is a node.

**Architecture**:
```
Input Image (600x600x3)
    ↓
CNN Feature Extractor
    ↓
Initial Label Predictions (17D vector)
    ↓
Label Graph Construction (17 nodes)
    ↓
Graph Convolutional Network (2-3 layers)
    ↓
Refined Label Predictions
    ↓
Sigmoid Activation (17 outputs)
```

**Pros**:
- Captures label dependencies (harbor → water + ships)
- Can learn co-occurrence patterns
- State-of-the-art approach
- More contextually aware predictions

**Cons**:
- Most complex implementation
- Requires graph construction
- Longer training time
- May overfit with small datasets

**Expected Performance**: 80-90% F1 score

**Implementation Options**:
1. **ML-GCN** (Multi-Label Graph Convolutional Network)
2. **Relation Network** (from the paper that created this dataset)
3. **KGGR** (Knowledge Graph Guided Reasoning)

---

### Approach 4: Ensemble Methods

**Description**: Combine predictions from multiple models trained with different architectures or strategies.

**Strategy**:
```
Model 1: ResNet50 → Predictions
Model 2: EfficientNet-B4 → Predictions
Model 3: ViT-Base → Predictions
    ↓
Ensemble (Average/Weighted/Voting)
    ↓
Final Predictions
```

**Pros**:
- Often achieves best performance
- Reduces variance and overfitting
- Can combine complementary strengths

**Cons**:
- Computationally expensive (training + inference)
- More complex deployment
- Diminishing returns

**Expected Performance**: +2-5% over best single model

---

## Critical Technical Considerations

### 1. Handling Class Imbalance

**Problem**: Some classes (buildings, trees) appear much more frequently than others (dock, mobile home).

**Solutions** (in order of implementation difficulty):

#### A. Class-Weighted Loss
```python
# Weight rare classes more heavily
pos_weight = torch.tensor([class_frequencies])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

#### B. Focal Loss
```python
# Automatically focus on hard examples
# Down-weights easy examples, up-weights hard ones
FL(p_t) = -α_t(1-p_t)^γ * log(p_t)
```

#### C. Oversampling/Undersampling
- Duplicate images with rare labels
- Sample more frequently during training

**Recommendation**: Start with class-weighted BCE, then try Focal Loss if needed.

---

### 2. Evaluation Metrics

**Do NOT use simple accuracy** - it's misleading in multi-label scenarios.

**Essential Metrics**:

1. **Hamming Loss**: Fraction of wrong labels (lower is better)
   ```
   HL = (1/N*L) * Σ XOR(y_true, y_pred)
   ```

2. **Subset Accuracy**: Exact match of all labels (very strict)
   ```
   SA = (1/N) * Σ [y_true == y_pred]
   ```

3. **Per-Class Metrics**:
   - Precision, Recall, F1 for each of 17 classes
   - Identify which classes perform poorly

4. **Average Metrics**:
   - **Micro-average**: Aggregate all classes, then compute (better for imbalanced)
   - **Macro-average**: Compute per-class, then average (treats all classes equally)
   - **Weighted-average**: Weight by class frequency

5. **mAP (mean Average Precision)**: Standard for multi-label

**Recommendation**: Report micro-F1, macro-F1, per-class F1, and mAP.

---

### 3. Training Strategy

#### Phase 1: Baseline (Week 1-2)
- Implement ResNet50 with BCE loss
- Basic augmentation (flip, rotate, color jitter)
- Train for 20-30 epochs
- Establish baseline metrics
- **Goal**: Working pipeline + baseline results

#### Phase 2: Optimization (Week 2-3)
- Experiment with different backbones (EfficientNet, ViT)
- Implement class weighting/focal loss
- Advanced augmentation (mixup, cutmix)
- Hyperparameter tuning (learning rate, batch size)
- **Goal**: Improve baseline by 5-10%

#### Phase 3: Advanced Techniques (Week 3-4)
- Add attention mechanisms OR
- Implement label correlation modeling (GNN) OR
- Build ensemble
- Implement Grad-CAM for interpretability
- **Goal**: State-of-the-art results + insights

---

### 4. Data Augmentation

**Standard Augmentations** (aerial images are rotation-invariant):
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),  # Aerial images: any rotation valid
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Advanced Augmentations**:
- **Mixup**: Blend two images and their labels
- **CutMix**: Paste image patches with corresponding labels
- **AutoAugment**: Learn augmentation policies

---

### 5. Thresholding Strategy

**Problem**: Sigmoid outputs probabilities, need to convert to binary predictions.

**Strategies**:

1. **Fixed Threshold (0.5)**: Simple but suboptimal
2. **Per-Class Threshold**: Optimize threshold for each class on validation set
3. **Dynamic Threshold**: Adjust based on number of expected labels
4. **Top-K**: Select K labels with highest probabilities

**Recommendation**: Optimize per-class thresholds on validation set using F1 score.

---

## Recommended Implementation Plan

### Phase 1: Baseline Implementation (PRIORITY)

**Week 1 Tasks**:
1. ✅ Data loading and exploration (DONE)
2. Implement ResNet50 baseline
3. Training loop with BCE loss
4. Basic evaluation metrics
5. Visualization of predictions

**Deliverables**:
- Working training script
- Baseline results (micro-F1, macro-F1, per-class F1)
- Loss/accuracy curves

---

### Phase 2: Optimization & Analysis

**Week 2 Tasks**:
1. Implement class-weighted loss or Focal Loss
2. Experiment with EfficientNet-B3/B4
3. Advanced data augmentation
4. Per-class threshold optimization
5. Error analysis (confusion matrix, failure cases)

**Deliverables**:
- Improved results (target: +5-10% F1)
- Ablation study results
- Analysis of which classes are difficult

---

### Phase 3: Advanced Techniques (CHOOSE ONE)

**Option A: Attention Mechanisms** (Medium difficulty)
- Implement CBAM or spatial attention
- Visualize attention maps
- Compare with baseline

**Option B: Graph Neural Networks** (High difficulty)
- Implement ML-GCN or Relation Network
- Build label co-occurrence graph
- Compare with baseline

**Option C: Ensemble + Interpretability** (Medium difficulty)
- Train 2-3 different models
- Implement ensemble strategy
- Add Grad-CAM visualization
- Analysis of model decisions

**Recommendation for CS415**: **Option C** - provides good results + interpretability for report.

---

## Tools and Libraries

### Essential
- **PyTorch**: Deep learning framework
- **torchvision**: Pre-trained models and transforms
- **Hugging Face Datasets**: Data loading
- **scikit-learn**: Metrics and evaluation
- **matplotlib/seaborn**: Visualization

### Advanced (Optional)
- **pytorch-lightning**: Clean training code
- **wandb**: Experiment tracking
- **timm**: More pre-trained models (EfficientNet, ViT)
- **torch-geometric**: For GNN implementation
- **grad-cam**: For interpretability

---

## Expected Results Comparison

| Approach | Expected Micro-F1 | Expected Macro-F1 | Implementation Time | Difficulty |
|----------|------------------|------------------|---------------------|------------|
| ResNet50 Baseline | 0.72-0.78 | 0.65-0.72 | 1 week | Easy |
| + Class Weighting | 0.75-0.80 | 0.70-0.76 | +2 days | Easy |
| + Better Backbone | 0.78-0.83 | 0.73-0.79 | +3 days | Easy |
| + Attention | 0.80-0.85 | 0.75-0.82 | +1 week | Medium |
| + GNN | 0.82-0.88 | 0.78-0.85 | +2 weeks | Hard |
| Ensemble | 0.84-0.90 | 0.80-0.87 | +1 week | Medium |

---

## Risk Mitigation

### Common Pitfalls

1. **Overfitting on small dataset (3000 images)**
   - **Solution**: Strong augmentation, dropout, early stopping

2. **Poor performance on rare classes**
   - **Solution**: Class weighting, focal loss, oversample rare classes

3. **Incorrect evaluation metrics**
   - **Solution**: Use multi-label specific metrics (micro/macro F1, mAP)

4. **Memory issues with large models**
   - **Solution**: Gradient accumulation, smaller batch size, mixed precision training

5. **Long training times**
   - **Solution**: Use GPU (Colab/Kaggle), smaller models first, reduce image size

---

## Final Recommendation for CS415 Project

### Optimal Strategy (Best Results + Report Quality):

**Phase 1 (Week 1)**:
- ResNet50 baseline with BCE loss
- Basic metrics and visualization
- **Target**: 72-78% micro-F1

**Phase 2 (Week 2)**:
- Add class-weighted loss
- Try EfficientNet-B3
- Optimize thresholds
- **Target**: 78-82% micro-F1

**Phase 3 (Week 3)**:
- Train 2-3 best models
- Implement ensemble
- Add Grad-CAM visualization
- Error analysis and insights
- **Target**: 82-85% micro-F1

**Phase 4 (Week 4)**:
- Write comprehensive report
- Generate all figures and tables
- Compare with related work
- Document findings

This approach balances:
- Strong technical implementation
- Good experimental results
- Rich analysis for report
- Interpretability and insights
- Manageable complexity

---

## Questions to Consider

Before implementation, decide:

1. **Which backbone?** ResNet50 (safe) vs EfficientNet (better) vs ViT (cutting-edge)?
2. **How to handle imbalance?** Class weights vs Focal Loss?
3. **Advanced technique?** Attention vs GNN vs Ensemble?
4. **Evaluation focus?** Macro-F1 (all classes equal) vs Micro-F1 (overall performance)?
5. **Report emphasis?** Technical depth vs Experimental breadth vs Interpretability?

---

## Next Steps

1. Review this strategy document
2. Decide on approach (recommend: Baseline → Optimization → Ensemble+Grad-CAM)
3. Set up training infrastructure (GPU, experiment tracking)
4. Implement baseline model
5. Establish evaluation pipeline
6. Iterate and improve

**Ready to start coding when you are!**
