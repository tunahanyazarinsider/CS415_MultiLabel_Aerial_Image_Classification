### Why Binary Cross-Entropy (BCE) is Chosen

**1. It Aligns Perfectly with the Sigmoid Activation Function**

As we discussed, the final layer of our network uses a sigmoid function to treat each label as an independent "Yes/No" question. Binary Cross-Entropy (BCE) is the loss function that is mathematically designed to handle exactly this kind of binary prediction task.

For each label, BCE measures the "distance" between the true answer (1 if the label is present, 0 if it's not) and the probability that our model predicted (the output of the sigmoid, e.g., 0.95).

**2. It Decomposes the Problem into Multiple Binary Tasks**

The core idea is that we are not solving one big, complex multi-label problem. Instead, we are solving many small, simple binary classification problems in parallel—one for each label.

BCE loss allows us to do this. It calculates the error for each label independently, as if it were its own separate task. This is fundamentally different from a loss function like Categorical Cross-Entropy (used with softmax), which assumes only one class can be correct and calculates a single loss for the entire set of classes.

### How Binary Cross-Entropy (BCE) is Used

Here’s a step-by-step look at how the loss is calculated for a single image during training:

Let's assume our project has four possible labels: `building`, `road`, `tree`, and `water`.

**Step 1: Define the True Labels**

For a given training image, we know the ground truth. Let's say the image contains a `building` and a `tree`, but no `road` or `water`. We represent this as a binary vector:

*   **True Labels:** `[1, 0, 1, 0]`
    *   (1 for `building`, 0 for `road`, 1 for `tree`, 0 for `water`)

**Step 2: Get the Model's Predictions**

The image is passed through our network, and the final sigmoid layer outputs a probability for each class. Let's say the model predicts:

*   **Predicted Probabilities:** `[0.9, 0.2, 0.8, 0.1]`
    *   (90% chance of `building`, 20% chance of `road`, 80% chance of `tree`, 10% chance of `water`)

**Step 3: Calculate the BCE Loss for Each Label Independently**

The BCE loss is now calculated for each label one by one:

*   **Loss for `building`:** The model was very confident (0.9) and it was right (true label is 1). **The loss will be very low.**
*   **Loss for `road`:** The model was not confident (0.2) and it was right (true label is 0). **The loss will be very low.**
*   **Loss for `tree`:** The model was confident (0.8) and it was right (true label is 1). **The loss will be very low.**
*   **Loss for `water`:** The model was not confident (0.1) and it was right (true label is 0). **The loss will be very low.**

In this case, the model did a good job, so the individual losses are all small.

**What if the model was wrong?**

Let's say for `tree`, the model predicted `0.1` (not confident) but the true label was `1` (it was present). The BCE loss for `tree` would be **very high**, heavily penalizing the model for this mistake.

**Step 4: Aggregate the Final Loss**

The final loss for the image is simply the **sum (or average) of all the individual losses** calculated in Step 3.

This single, aggregated loss value is what the optimizer uses. During backpropagation, the model adjusts its internal weights with the goal of minimizing this total loss. By doing so, it learns to make better predictions for all labels simultaneously.

In summary, **BCE is chosen because it allows us to train a single network to perform multiple, independent binary classification tasks at once**, which is exactly what multi-label classification is.