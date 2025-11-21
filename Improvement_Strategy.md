# Model Improvement Strategy

## Current Performance Analysis

**Current Metrics:**
- Macro F1: **0.75** (good baseline)
- Micro F1: **0.91** (excellent overall)
- **Gap: 0.16** (indicates imbalance issues)

**What This Tells Us:**
- ✅ Model performs excellently on frequent classes (0.91 micro)
- ⚠️ Model struggles with rare classes (pulling macro down to 0.75)
- 🎯 **Target:** Improve macro F1 to 0.80-0.85 while maintaining micro F1 > 0.88

---

## Improvement Roadmap (Ranked by Impact)

### Priority 1: Address Class Imbalance (Expected: +3-5% macro F1)

These techniques directly target your macro-micro gap.

---

#### 1.1 Class-Weighted BCE Loss ⭐ **[EASIEST & HIGH IMPACT]**

**Why:** Penalizes mistakes on rare classes more heavily.

**Implementation:**

```python
# Calculate positive class weights (inverse frequency)
import numpy as np

def calculate_pos_weights(train_labels):
    """
    Calculate positive weights for BCEWithLogitsLoss.
    Higher weight = rarer class.
    """
    num_samples = len(train_labels)
    pos_counts = train_labels.sum(axis=0)  # Count positives per class
    neg_counts = num_samples - pos_counts

    # pos_weight = neg_count / pos_count
    pos_weights = neg_counts / (pos_counts + 1e-6)  # Avoid division by zero

    return torch.FloatTensor(pos_weights)

# In your training setup:
pos_weights = calculate_pos_weights(np.array(y_train))
pos_weights = pos_weights.to(device)

# Use weighted loss
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
```

**Expected Improvement:**
- Macro F1: 0.75 → **0.78-0.80**
- Micro F1: 0.91 → **0.89-0.90** (slight decrease acceptable)
- Gap: 0.16 → **0.09-0.11**

**Pros:**
- 5 lines of code change
- No architecture modification
- Proven effective for imbalance

**Cons:**
- May slightly hurt micro F1
- Needs hyperparameter tuning (can scale weights)

---

#### 1.2 Focal Loss ⭐⭐ **[BEST FOR SEVERE IMBALANCE]**

**Why:** Automatically focuses on hard examples, down-weights easy ones.

**Implementation:**

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for multi-label classification.

        Args:
            alpha: Weighting factor (0-1), balance positive/negative
            gamma: Focusing parameter (>=0), higher = more focus on hard examples
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits (before sigmoid)
        # targets: binary labels

        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        pt = torch.exp(-BCE_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Usage:
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

**Hyperparameter Tuning:**
- `gamma=0` → standard BCE
- `gamma=1` → mild focusing
- `gamma=2` → moderate focusing (recommended start)
- `gamma=3-5` → strong focusing

Try: `gamma in [1.0, 2.0, 3.0]` and `alpha in [0.25, 0.5, 0.75]`

**Expected Improvement:**
- Macro F1: 0.75 → **0.80-0.82**
- Micro F1: 0.91 → **0.89-0.91** (usually maintains)
- Gap: 0.16 → **0.07-0.11**

**Pros:**
- State-of-the-art for imbalance
- No oversampling needed
- Adaptive (focuses where needed)

**Cons:**
- Requires hyperparameter search
- Slightly more complex

---

#### 1.3 Oversampling Rare Classes

**Why:** Balance training data distribution.

**Implementation:**

```python
from torch.utils.data import WeightedRandomSampler

def get_sample_weights(labels, num_classes):
    """
    Calculate sampling weights based on label rarity.
    Images with rare labels get sampled more often.
    """
    # Count how many rare labels each sample has
    label_counts = np.array([labels[i].sum() for i in range(len(labels))])

    # Alternative: weight by rarest label in image
    pos_counts = np.array(labels).sum(axis=0)
    label_weights = 1.0 / (pos_counts + 1)  # Inverse frequency

    sample_weights = []
    for label_vec in labels:
        # Weight = average weight of all positive labels
        positive_indices = [i for i in label_vec]
        if len(positive_indices) > 0:
            weight = np.mean([label_weights[i] for i in positive_indices])
        else:
            weight = 1.0
        sample_weights.append(weight)

    return torch.DoubleTensor(sample_weights)

# Create weighted sampler
sample_weights = get_sample_weights(y_train, num_classes)
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Use in DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,  # Instead of shuffle=True
    num_workers=2
)
```

**Expected Improvement:**
- Macro F1: 0.75 → **0.77-0.79**
- Micro F1: 0.91 → **0.90-0.91**

**Pros:**
- Works with any loss function
- Can combine with focal loss

**Cons:**
- Longer epochs (see duplicate samples)
- May overfit rare classes

---

### Priority 2: Better Backbone Architecture (Expected: +2-4% overall)

Replace ResNet50 with more powerful models.

---

#### 2.1 EfficientNet-B3/B4 ⭐⭐ **[BEST BALANCE]**

**Why:** Better accuracy-efficiency trade-off than ResNet.

**Implementation:**

```python
import timm  # pip install timm

class MultiLabelEfficientNet(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b3', pretrained=True, dropout=0.5):
        super().__init__()

        # Load EfficientNet from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )

        # Get feature dimension
        num_features = self.backbone.num_features

        # Custom head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.flatten(1)

        # Classify
        logits = self.classifier(features)
        return logits

# Usage:
model = MultiLabelEfficientNet(
    num_classes=17,
    model_name='efficientnet_b3',  # or 'efficientnet_b4'
    pretrained=True,
    dropout=0.5
)
```

**Model Options:**
- `efficientnet_b0`: Lightweight, faster training
- `efficientnet_b3`: **Recommended** - good balance
- `efficientnet_b4`: Higher accuracy, slower
- `efficientnet_b5`: Best accuracy, much slower

**Expected Improvement:**
- Macro F1: 0.75 → **0.78-0.81**
- Micro F1: 0.91 → **0.92-0.93**

**Pros:**
- Better than ResNet50 on most benchmarks
- More efficient (fewer params for same accuracy)
- Compound scaling

**Cons:**
- Requires `timm` library
- Slightly longer training time than ResNet34

---

#### 2.2 ResNet101 ⭐ **[SIMPLE UPGRADE]**

**Why:** Deeper = more capacity, same architecture.

**Implementation:**

```python
# Just change one line in your existing code:
model = MultiLabelCNN(
    num_classes=num_classes,
    backbone='resnet101',  # Instead of 'resnet50'
    pretrained=True,
    dropout=0.5
)
```

**Expected Improvement:**
- Macro F1: 0.75 → **0.76-0.78**
- Micro F1: 0.91 → **0.91-0.92**

**Pros:**
- No code changes needed (already supported in your notebook!)
- Proven architecture

**Cons:**
- More parameters (slower training)
- Diminishing returns vs ResNet50

---

### Priority 3: Advanced Data Augmentation (Expected: +1-3% overall)

Your current augmentation is good, but can be enhanced.

---

#### 3.1 Mixup Augmentation ⭐⭐

**Why:** Blends images and labels, improves generalization.

**Implementation:**

```python
def mixup_data(x, y, alpha=0.2):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute loss for mixup
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# In training loop:
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)

    # Apply mixup
    images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)

    # Forward
    outputs = model(images)
    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Hyperparameter:**
- `alpha=0.2`: Mild mixing (recommended)
- `alpha=0.4`: Moderate mixing
- `alpha=1.0`: Strong mixing

**Expected Improvement:**
- Macro F1: 0.75 → **0.76-0.78**
- Micro F1: 0.91 → **0.91-0.92**
- Better generalization

**Pros:**
- Proven regularization technique
- Works well with multi-label
- Simple to implement

**Cons:**
- Training takes longer (slower convergence)

---

#### 3.2 CutMix Augmentation

**Why:** Pastes image regions with corresponding labels.

**Implementation:**

```python
def cutmix_data(x, y, alpha=1.0):
    """
    Returns cutmix inputs and labels
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Generate random box
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Adjust lambda based on actual box area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    # Apply cutmix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # For multi-label: combine labels (union)
    y_mixed = torch.max(y, y[index])

    return x, y_mixed

# In training loop (50% probability):
if np.random.rand() < 0.5:
    images, labels = cutmix_data(images, labels, alpha=1.0)
```

**Expected Improvement:**
- Macro F1: 0.75 → **0.76-0.77**
- Micro F1: 0.91 → **0.91-0.92**

---

#### 3.3 Enhanced Color Augmentation

**Why:** Aerial images have varying lighting/weather conditions.

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),

    # Enhanced color augmentation
    transforms.ColorJitter(
        brightness=0.3,  # Increased from 0.2
        contrast=0.3,
        saturation=0.3,
        hue=0.15  # Added hue variation
    ),
    transforms.RandomGrayscale(p=0.1),  # Occasionally grayscale

    # Optional: Gaussian blur
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ], p=0.2),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

### Priority 4: Attention Mechanisms (Expected: +2-4% overall)

Add spatial/channel attention to focus on relevant regions.

---

#### 4.1 CBAM (Convolutional Block Attention Module) ⭐⭐⭐

**Why:** Learns what (channel) and where (spatial) to focus.

**Implementation:**

```python
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # Average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average and max along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and convolve
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x

# Integrate into ResNet:
class MultiLabelCNN_CBAM(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', pretrained=True, dropout=0.5):
        super().__init__()

        # Load ResNet
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)

        # Remove final FC and avgpool
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Add CBAM after layer4
        self.cbam = CBAM(2048, reduction=16)  # 2048 for ResNet50 layer4

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Apply CBAM
        x = self.cbam(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
```

**Expected Improvement:**
- Macro F1: 0.75 → **0.78-0.80**
- Micro F1: 0.91 → **0.92-0.93**

**Pros:**
- Learns to focus on relevant regions
- Interpretable (can visualize attention)
- Only 15% more computation

**Cons:**
- More complex architecture
- Requires careful integration

---

### Priority 5: Per-Class Threshold Optimization (Expected: +1-2%)

**Why:** 0.5 threshold may not be optimal for all classes.

**Implementation:**

```python
from sklearn.metrics import f1_score

def optimize_thresholds(y_true, y_pred_probs, metric='f1'):
    """
    Find optimal threshold for each class independently.
    """
    num_classes = y_true.shape[1]
    optimal_thresholds = []

    for i in range(num_classes):
        best_threshold = 0.5
        best_score = 0

        # Try thresholds from 0.1 to 0.9
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred_class = (y_pred_probs[:, i] > threshold).astype(int)
            score = f1_score(y_true[:, i], y_pred_class, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        optimal_thresholds.append(best_threshold)

    return np.array(optimal_thresholds)

# On validation set:
_, _, val_labels, val_probs = evaluate(model, val_loader, criterion, device, threshold=0.5)
optimal_thresholds = optimize_thresholds(val_labels, val_probs)

print("Optimal thresholds per class:")
for i, (class_name, thresh) in enumerate(zip(class_names, optimal_thresholds)):
    print(f"  {class_name}: {thresh:.2f}")

# Apply to test set:
test_preds_optimized = np.zeros_like(test_probs)
for i in range(num_classes):
    test_preds_optimized[:, i] = (test_probs[:, i] > optimal_thresholds[i]).astype(int)
```

**Expected Improvement:**
- Macro F1: 0.75 → **0.76-0.77**
- Micro F1: 0.91 → **0.91-0.92**

---

### Priority 6: Ensemble Methods (Expected: +2-3%)

**Why:** Combine strengths of multiple models.

**Simple Ensemble:**

```python
# Train 3 different models
models = [
    MultiLabelCNN(num_classes, backbone='resnet50'),
    MultiLabelCNN(num_classes, backbone='resnet101'),
    MultiLabelEfficientNet(num_classes, model_name='efficientnet_b3')
]

# Train each model separately (already done)

# Ensemble prediction (average probabilities):
def ensemble_predict(models, dataloader, device):
    all_probs = []

    for model in models:
        model.eval()
        model_probs = []

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                model_probs.append(probs.cpu().numpy())

        all_probs.append(np.vstack(model_probs))

    # Average predictions
    ensemble_probs = np.mean(all_probs, axis=0)
    return ensemble_probs

# Use ensemble:
ensemble_probs = ensemble_predict(models, test_loader, device)
ensemble_preds = (ensemble_probs > 0.5).astype(int)
```

**Expected Improvement:**
- Macro F1: 0.75 → **0.77-0.80**
- Micro F1: 0.91 → **0.92-0.94**

---

## Recommended Implementation Order

### Week 1: Quick Wins (Expected: 0.75 → 0.78-0.80 macro F1)

1. ✅ **Implement class-weighted BCE loss** (30 min)
2. ✅ **Try Focal Loss** (1 hour)
3. ✅ **Per-class threshold optimization** (30 min)
4. ✅ **Enhanced augmentation** (30 min)

**Time:** 3 hours
**Expected Macro F1:** 0.78-0.80

---

### Week 2: Architecture Improvements (Expected: 0.78 → 0.80-0.82)

1. ✅ **Try EfficientNet-B3** (1 hour setup + training)
2. ✅ **Add Mixup augmentation** (1 hour)
3. ✅ **Compare ResNet50 vs ResNet101 vs EfficientNet** (training time)

**Time:** 1 day training + analysis
**Expected Macro F1:** 0.80-0.82

---

### Week 3: Advanced Techniques (Expected: 0.80 → 0.82-0.84)

1. ✅ **Implement CBAM attention** (2-3 hours)
2. ✅ **Train best model + CBAM** (training time)
3. ✅ **Ensemble top 3 models** (1 hour)

**Time:** 1 day training + analysis
**Expected Macro F1:** 0.82-0.84

---

## Summary: Impact vs Effort

| Technique | Effort | Expected Macro F1 Gain | Micro F1 Impact | Priority |
|-----------|--------|----------------------|-----------------|----------|
| **Class-weighted BCE** | Low | +3-5% | -1-2% | ⭐⭐⭐ |
| **Focal Loss** | Low | +5-7% | 0-1% | ⭐⭐⭐ |
| **EfficientNet-B3** | Medium | +3-6% | +1-2% | ⭐⭐⭐ |
| **CBAM Attention** | Medium | +3-5% | +1-2% | ⭐⭐ |
| **Mixup** | Low | +1-3% | 0-1% | ⭐⭐ |
| **Threshold Opt** | Low | +1-2% | 0-1% | ⭐⭐ |
| **Oversampling** | Low | +2-4% | -1% | ⭐ |
| **ResNet101** | Very Low | +1-3% | 0-1% | ⭐ |
| **Ensemble** | High | +2-3% | +1-2% | ⭐ |
| **CutMix** | Low | +1-2% | 0-1% | ⭐ |

---

## Realistic Target Performance

**Conservative Estimate (Week 1-2):**
- Macro F1: **0.80-0.82**
- Micro F1: **0.90-0.92**

**Optimistic Estimate (Week 3 with ensemble):**
- Macro F1: **0.83-0.86**
- Micro F1: **0.91-0.93**

**Top Performance (All techniques):**
- Macro F1: **0.85-0.88**
- Micro F1: **0.92-0.94**

---

## Next Steps

1. **Analyze per-class performance** - identify which classes are worst
2. **Start with Focal Loss** - biggest single improvement
3. **Try EfficientNet-B3** - better backbone
4. **Add CBAM if time permits** - interpretability bonus
5. **Ensemble for final boost** - competition-winning technique

Would you like me to:
1. Create a new notebook implementing Focal Loss?
2. Update existing notebook with class-weighted BCE?
3. Implement CBAM-enhanced ResNet?
4. Create ensemble prediction script?
