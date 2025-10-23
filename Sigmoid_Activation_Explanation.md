### What is the Sigmoid Function?

The sigmoid function is a mathematical function that takes any real-valued number as input and "squashes" it into a range between 0 and 1.

No matter how large or small the input number is, the output of the sigmoid function will always be between 0 and 1. Because of this property, its output can be easily interpreted as a **probability**. A value close to 1 indicates a high probability, while a value close to 0 indicates a low probability.

### Why Sigmoid is Chosen for Multi-Label Classification

**1. It Enables Independent Predictions for Each Class**

The fundamental difference between single-label and multi-label classification is how we treat the classes.

*   In **single-label** classification, we assume an image can only belong to *one* class.
*   In **multi-label** classification, we assume an image can belong to *multiple* classes at the same time.

To handle this, we treat the prediction for each label as a separate, independent "Yes/No" question. The sigmoid function is the perfect tool for this. When applied to the final layer of a neural network, it gives us an independent probability for each class, answering questions like:

*   What is the probability that a 'building' is in this image? (e.g., 0.95)
*   What is the probability that a 'road' is in this image? (e.g., 0.88)
*   What is the probability that a 'water' body is in this image? (e.g., 0.12)

Notice that these probabilities do not need to add up to 1. They are independent assessments.

**2. The Critical Difference from Softmax**

To understand this better, it's helpful to contrast sigmoid with the **softmax** function, which is used for single-label classification.

Softmax also outputs probabilities, but it forces the sum of all probabilities across all classes to be equal to 1. This makes the classes compete with each other. Softmax is designed to answer the question: "Which *single* class is the most likely for this image?"

*   **Softmax is for a "multiple-choice" question:** You must pick only one answer.
*   **Sigmoid is for a "check all that apply" question:** You can pick as many answers as are relevant.

Since an aerial image can contain both a 'building' and a 'road', we need our model to be able to say "Yes" to both. Sigmoid allows this, while softmax would force a choice between them.

**3. How it is Used in Practice**

After the final layer applies the sigmoid function to produce a probability for each class, we use a **threshold** (commonly 0.5) to make the final classification decision.

For each label, the rule is simple:
*   If the predicted probability is **greater than the threshold**, we assign that label to the image.
*   If the predicted probability is **less than or equal to the threshold**, we do not assign it.

**Example:**
*   Model's Probabilities: `[building: 0.95, road: 0.88, water: 0.12]`
*   Threshold: `0.5`
*   Final Labels: `['building', 'road']`

In summary, the sigmoid function is chosen because it correctly frames the multi-label task as a series of independent binary decisions, allowing the model to identify any number of relevant objects or themes within a single image.