CIFAR-10 Shortcut Learning Experiment

Investigating whether Convolutional Neural Networks (CNNs) rely on artificial shortcuts rather than learning robust object representations.

Overview

Deep learning models can achieve high accuracy by exploiting simple patterns that correlate with class labels, even when those patterns are not the meaningful features we want them to learn. This behavior is known as shortcut learning.

This project adapts the central idea of the "Shortcut Perspective repository" (https://github.com/rgeirhos/shortcut-perspective) to CIFAR-10 by introducing class-correlated color patches into images and evaluating model performance under different test conditions.

Research question: Does a CNN rely on an artificial visual shortcut, and how does its performance change when that shortcut is removed or contradicts the true label?

«This is an experimental extension inspired by the original repository, not a direct reproduction of its original synthetic experiment.»

Objectives

- Implement a shortcut-learning experiment using CIFAR-10.
- Introduce artificial, class-correlated visual features.
- Train a CNN using multiple random seeds.
- Evaluate performance under different shortcut conditions.
- Measure variability across five independent training runs.
- Investigate evidence of shortcut reliance through accuracy differences.

Dataset

The experiment uses "CIFAR-10" (https://www.cs.toronto.edu/~kriz/cifar.html), a dataset of 60,000 color images across 10 classes.

Split| Number of images
Training| 45,000
Validation| 5,000
Test| 10,000
Total| 60,000

Each image has a resolution of 32 × 32 pixels and three color channels (RGB).

The official test set is kept separate from training and validation.

Methodology

1. Shortcut Injection

A 5 × 5 colored patch is placed in the top-right corner of each image. The patch color is associated with the image's class during training.

This creates an artificial predictive signal that a CNN may learn to exploit instead of relying on the object's visual features.

2. Model Architecture

A lightweight CNN is trained from scratch using:

- Three convolutional layers
- ReLU activation functions
- Max pooling
- Adaptive average pooling
- A fully connected layer with 10 output classes

3. Training Configuration

Parameter| Configuration
Dataset| CIFAR-10
Optimizer| Adam
Learning rate| 0.001
Loss function| Cross-entropy
Batch size| 128
Default epochs| 10
Random seeds| 42, 123, 2026, 7, 99
Framework| PyTorch
Execution environment| Google Colab

The same data split and model configuration are used across runs. Model initialization and training-data shuffling vary by random seed.

Evaluation Protocol

The trained model is evaluated under three conditions.

A. IID Evaluation

The patch color matches the class-to-color mapping used during training.

Purpose: Measure performance when the learned shortcut remains available.

B. Patch-Free Evaluation

The colored patch is removed from test images.

Purpose: Measure performance when the artificial shortcut is absent.

C. Conflicting-Shortcut Evaluation

The patch indicates a different class from the image's true label.

Purpose: Investigate whether a contradictory shortcut influences predictions.

Evaluation Metrics

The following metrics are recorded for each run:

- Validation accuracy: Accuracy on the validation set.
- IID test accuracy: Accuracy with the matching shortcut.
- Patch-free test accuracy: Accuracy without the patch.
- Conflicting-shortcut accuracy: Accuracy when the patch contradicts the true label.
- Clean accuracy gap: IID accuracy minus patch-free accuracy.
- Conflict accuracy gap: IID accuracy minus conflicting-shortcut accuracy.
- Training time: Duration of each run.

Accuracy gaps are reported in percentage points.

Repeated Experiments

Five training runs are performed using the following random seeds:

"42, 123, 2026, 7, 99"

For each metric, the mean and sample standard deviation are calculated across completed runs.

These repeated runs help assess the stability of the results under different initializations and data-order shuffles. They use the same fixed data split and therefore do not represent five independent datasets.

Results

Results will be added after all five experiments have completed successfully.

Metric| Mean| Standard deviation
IID test accuracy| Pending| Pending
Patch-free test accuracy| Pending| Pending
Conflicting-shortcut accuracy| Pending| Pending
Clean accuracy gap| Pending| Pending
Conflict accuracy gap| Pending| Pending

No experimental values are reported until they have been measured.

Interpretation

A substantial decrease in accuracy when the patch is removed or contradicted may provide evidence consistent with shortcut reliance.

However, accuracy differences alone do not establish that the model relies exclusively on the patch. Results should be interpreted alongside the experimental design, training behavior, and limitations.

Running the Experiment

1. Clone the Repository

git clone https://github.com/rgeirhos/shortcut-perspective.git
cd shortcut-perspective

2. Configure Google Colab

Open Google Colab and select a GPU runtime if one is available.

Check GPU availability:

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

3. Run a Smoke Test

After placing "cifar10_shortcut.py" in the project directory, run:

EPOCHS=1 TRAIN_LIMIT=1000 SEEDS=42 python cifar10_shortcut.py

This checks that dataset loading, training, evaluation, and result logging work correctly.

4. Run All Five Experiments

EPOCHS=10 TRAIN_LIMIT=0 SEEDS=42,123,2026,7,99 python cifar10_shortcut.py

The script downloads CIFAR-10 if necessary and saves the results and model checkpoints.

Output Files

The experiment generates the following files:

results/
├── results.csv
├── summary.csv
├── model_seed_42.pt
├── model_seed_123.pt
├── model_seed_2026.pt
├── model_seed_7.pt
└── model_seed_99.pt

- "results.csv" contains per-run metrics.
- "summary.csv" contains the mean and sample standard deviation.
- "model_seed_*.pt" contains saved model weights for each completed run.

Limitations

- The artificial shortcut is deliberately strong and perfectly class-correlated during training.
- The experiment uses a lightweight CNN and a fixed training configuration.
- Five runs assess variability across training runs, not across independently sampled datasets.
- The current implementation evaluates the final training epoch rather than selecting a checkpoint using validation accuracy.
- GPU availability and execution time depend on the Google Colab runtime.

References

- "Shortcut Perspective — Original Repository" (https://github.com/rgeirhos/shortcut-perspective)
- "CIFAR-10 Dataset" (https://www.cs.toronto.edu/~kriz/cifar.html)
- "PyTorch Documentation" (https://pytorch.org/docs/stable/index.html)

---

Project status: Implementation and experimental evaluation in progress.
