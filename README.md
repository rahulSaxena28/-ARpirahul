CIFAR-10 Shortcut Learning Experiment
Overview
This project adapts the core idea of the Shortcut Perspective repository to a CIFAR-10 image-classification experiment.
Research question: Does a CNN rely on an easy, class-correlated visual shortcut rather than robust object features?
The experiment adds a small colored patch to each image. The patch color is associated with the image's class during training. We then evaluate the model under three conditions to investigate whether performance changes when the shortcut is removed or contradicts the true label.
This is an experimental extension inspired by the original repository, not a direct reproduction of its original synthetic stars-and-moons experiment.
Dataset
CIFAR-10 contains 60,000 color images of size 32×32 across 10 classes:
50,000 training images
10,000 test images
The script uses a fixed split of the official training set:
45,000 training examples
5,000 validation examples
10,000 official test examples
The test set is reserved for final evaluation.
Shortcut design
A 5×5 colored patch is placed in the top-right corner of each image.
The experiment evaluates:
IID: the patch color matches the true class's assigned color.
Clean: the patch is absent.
Conflict: the patch indicates a different class (the next class modulo 10), while the true label stays unchanged.
A large IID-to-clean or IID-to-conflict accuracy gap can be evidence consistent with shortcut reliance. It does not, by itself, prove that the model relies exclusively on the shortcut.
Model and training
The implementation uses a small CNN with three convolutional layers, ReLU activations, max pooling, adaptive average pooling, and a 10-class linear output layer.
Optimizer: Adam
Learning rate: 0.001
Loss: cross-entropy
Default epochs: 10
Default batch size: 128
Seeds: 42, 123, 2026, 7, 99
Each run initializes a new model. The data split, model architecture, and training configuration remain fixed; initialization and training-data shuffle vary by seed.
Running in Google Colab
Open Google Colab and create a notebook.
If available, choose Runtime → Change runtime type → T4 GPU.
Clone the repository and place cifar10_shortcut.py in /content/shortcut-perspective/.
Check whether CUDA is available:
import torch
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
Run a small smoke test first:
!EPOCHS=1 TRAIN_LIMIT=1000 SEEDS=42 python /content/shortcut-perspective/cifar10_shortcut.py
If the smoke test succeeds, run the full five-seed experiment:
!EPOCHS=10 TRAIN_LIMIT=0 SEEDS=42,123,2026,7,99 python /content/shortcut-perspective/cifar10_shortcut.py
The script downloads CIFAR-10 if needed. GPU availability is not guaranteed by Colab; CPU runs may be slower.
Outputs
The script saves outputs under results/:
results.csv: one row per completed seed
summary.csv: mean and sample standard deviation across completed runs
model_seed_<seed>.pt: model checkpoint for each completed run
Metrics recorded per seed:
Validation accuracy
IID test accuracy
Clean test accuracy
Conflict test accuracy
IID-minus-clean accuracy gap (percentage points)
IID-minus-conflict accuracy gap (percentage points)
Training duration in seconds
Accuracy values are percentages. Accuracy gaps are in percentage points.
Interpretation
High IID, lower clean accuracy: performance depends on the presence of the shortcut or on other differences between the conditions.
High IID, lower conflict accuracy: a contradictory shortcut may be influencing predictions.
Small gaps: no large performance change was observed under these tests, but this alone does not prove that the model ignored the shortcut.
Report the mean and sample standard deviation across the five seeds. Do not invent or fill in results before the runs finish. Keep the configuration and software versions with the results.
Limitations and reproducibility notes
This is a deliberately strong synthetic shortcut: all training images receive a class-correlated patch.
Five seeds represent repeated training runs on the same fixed split, not five independent datasets.
The model is a lightweight baseline, not an exhaustive architecture study.
The current script evaluates the final epoch rather than selecting a checkpoint by best validation accuracy.
Record Python, PyTorch, Torchvision versions, runtime type, training duration, and any code changes.
Verify the notebook's actual code and outputs before reporting final results; saved notebook outputs may be stale.
References
Original repository: https://github.com/rgeirhos/shortcut-perspective
CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
