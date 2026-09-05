🧠 CIFAR-10 Image Generation using DCGAN (PyTorch)

This project builds a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch to generate synthetic CIFAR-10 images. A Generator learns to create realistic images from random noise while a Discriminator learns to tell real images from fake ones, with both improving through adversarial training.

📌 Table of Contents

* [Project Overview](#project-overview)
* [Technologies Used](#technologies-used)
* [Dataset](#dataset)
* [Data Preprocessing](#data-preprocessing)
* [Model Architecture](#model-architecture)
* [Training Process](#training-process)
* [Experimentation](#experimentation)
* [Loss Function & Optimizers](#loss-function--optimizers)
* [Results](#results)
* [Project Structure](#project-structure)
* [How to Run](#how-to-run)
* [Future Improvements](#future-improvements)

📊 Project Overview

The goal is to train a DCGAN to generate realistic 32x32 RGB images similar to those in the CIFAR-10 dataset. The Generator and Discriminator are trained simultaneously in an adversarial setup: the Generator tries to fool the Discriminator, while the Discriminator tries to correctly classify real vs. fake images.

🛠 Technologies Used

* Python
* PyTorch
* Torchvision
* Pandas
* Matplotlib
* PIL (Pillow)
* CUDA (GPU acceleration)

🗂 Dataset

* CIFAR-10 dataset
* Image size: 32x32 RGB
* 50,000 training images used (unlabeled for GAN training)
* Loaded and automatically downloaded via `torchvision.datasets.CIFAR10`

🔄 Data Preprocessing

* Resize to (32, 32)
* RandomHorizontalFlip
* ToTensor
* Normalization: mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]
* Batch size: 128
* Shuffling enabled

🧠 Model Architecture

**Discriminator**

* Conv2D (3 → 64), LeakyReLU
* Conv2D (64 → 128), BatchNorm, LeakyReLU
* Conv2D (128 → 256), BatchNorm, LeakyReLU
* Conv2D (256 → 1), Sigmoid
* Outputs a probability of the image being real

**Generator**

* ConvTranspose2D (100 → 256), BatchNorm, ReLU
* ConvTranspose2D (256 → 128), BatchNorm, ReLU
* ConvTranspose2D (128 → 64), BatchNorm, ReLU
* ConvTranspose2D (64 → 32), BatchNorm, ReLU
* Conv2D (32 → 3), Tanh
* Takes a 100-dimensional random noise vector as input and outputs a 32x32x3 image

🚀 Training Process

* Loss Function: Binary Cross-Entropy Loss (BCELoss)
* Optimizer: Adam (lr = 0.0002, betas = (0.5, 0.999)) for both Generator and Discriminator
* Batch Size: 128
* Epochs: 150
* Latent Vector Size: 100
* Device: CUDA (NVIDIA GeForce RTX 5060 Laptop GPU)

Each training step alternates between:
1. Training the Discriminator on real images (label = 1) and fake images (label = 0)
2. Training the Generator to fool the Discriminator (target label = 1 for fake images)

Generated sample images are saved every 5 epochs to track visual progress.

🧪 Experimentation

Multiple training runs were done to find the best configuration:

* Started with **50 epochs** — generated images were blurry and lacked clear structure
* Increased to **100 epochs** — noticeable improvement in image sharpness and object shapes
* Finally trained for **150 epochs** with tuned **batch size** and **learning rate** — this gave the most stable training and the clearest, most realistic generated images

Adjusting batch size and learning rate alongside the epoch count helped balance the Generator and Discriminator learning speed, reducing mode collapse and leading to steadily improving image quality as training progressed.

⚖️ Loss Function & Optimizers

* **Discriminator Loss** = Real Loss + Fake Loss (BCELoss)
* **Generator Loss** = BCELoss between Discriminator's output on fake images and real labels
* Both networks use separate Adam optimizers to allow independent learning rates and momentum behavior

📈 Results

* Discriminator loss generally decreased over training, showing improved ability to detect fake images
* Generator loss fluctuated as it adapted to a strengthening Discriminator, a typical pattern in adversarial training
* Visual quality of generated images improved noticeably from early epochs to epoch 150
* Sample grids were saved at epochs 15, 30, 45, 60, 75, 90, 105, 120, 135, and 150 for comparison

📂 Project Structure

```
GAN-Based-Image-Generation
│
├── data/
│   └── (CIFAR-10 dataset, auto-downloaded)
│
├── generated_images_DCGAN/
│   ├── epoch_5.png
│   ├── epoch_10.png
│   └── ... (every 5 epochs up to epoch_150.png)
│
├── .gitignore
├── DCGAN_CIFAR10.ipynb
└── README.md
```

▶️ How to Run

Clone Repository
```
git clone https://github.com/sushil0126/GAN-Based-Image-Generation.git 
```

Install Dependencies
```
pip install torch torchvision pandas matplotlib pillow
```

Run Training
```
jupyter notebook
```

🔮 Future Improvements

* Add label smoothing and noisy labels to stabilize adversarial training
* Experiment with Wasserstein loss (WGAN-GP) for more stable convergence
* Increase image resolution using progressive growing techniques
* Track and plot Generator/Discriminator loss curves over epochs
* Compute FID (Fréchet Inception Distance) score to quantitatively evaluate image quality
* Deploy a demo using Streamlit or Gradio to generate images interactively