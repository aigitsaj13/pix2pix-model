# pix2pix-model
Implementing the pix2pix model to convert segmented images into real images of the environment
This repository contains an implementation of the Pix2Pix model, a generative adversarial network (GAN) designed for paired image-to-image translation tasks. The Pix2Pix framework is commonly used for tasks such as image colorization, style transfer, and semantic image segmentation.
Key Features:

    Custom Dataset Class: The Pix2PixDataset class loads and preprocesses images for training and validation. Each input image is split into two halves: the left half as the target image and the right half as the input image.
    Generator Architecture: Implements a U-Net-like architecture with downsampling (encoder) and upsampling (decoder) layers. Skip connections are used to improve the generator's ability to reconstruct high-frequency details.
    Discriminator Architecture: Uses a PatchGAN discriminator to evaluate the quality of generated images. The model compares the generated images with real ones to provide adversarial feedback to the generator.
    Custom Loss Function: A combination of adversarial loss (using binary cross-entropy) and L1 loss is used to optimize the generator. The balance between these losses is controlled by a configurable hyperparameter lambda_L1.
    Visualization During Training: Periodically displays side-by-side comparisons of input images, generated images, and target images to monitor training progress.
    Loss Monitoring: Tracks and plots the generator and discriminator losses during training for performance analysis.

Code Structure:

    Data Loading and Preprocessing: The Pix2PixDataset class and PyTorch's DataLoader manage image loading and augmentation.
    Model Architectures:
        Generator: A U-Net-based model with downsampling and upsampling layers.
        Discriminator: A PatchGAN model for evaluating image quality at the patch level.
    Training Loop: Implements a robust training pipeline with loss computation, gradient updates, and periodic evaluation of results.
    Hyperparameters: Configurable settings for learning rates, batch size, number of epochs, and the L1 loss weight (lambda_L1).

Requirements:

    Python 3.7+
    PyTorch 1.10+
    torchvision
    matplotlib
    PIL (Pillow)

How to Use:

    Clone the repository and install the required dependencies.
    Download and prepare your dataset. Ensure the images are structured such that the input and target images are concatenated side-by-side.
    Update the dataset_path variable with the location of your dataset.
    Run the training script to train the Pix2Pix model.
    Visualize results and evaluate the model's performance using the generated images.

Results:

During training, the generator progressively improves its ability to synthesize realistic target images from the input images. The final results include:

    Realistic image-to-image translation results.
    Loss curves for both the generator and discriminator.

Applications:

    Semantic segmentation to RGB image generation.
    Black-and-white image colorization.
    Style transfer.
    Image inpainting.


The following hyperparameters are used in the code, along with their values:
1. Generator and Discriminator Optimizers

    Learning Rate (lr): 0.0002
        Controls the step size during parameter updates.
    Beta1 (betas[0]): 0.5
        Decay rate for the first moment of the gradient in Adam optimizer.
    Beta2 (betas[1]): 0.999
        Decay rate for the second moment of the gradient in Adam optimizer.

2. Batch Size

    Value: 128
        Number of image pairs processed in one forward/backward pass.

3. Image Size

    Value: (256, 256)
        Images are resized to this dimension during preprocessing using the transforms.Resize function.

4. Normalization

    Mean: (0.5, 0.5, 0.5)
        Applied to normalize each channel of the images.
    Standard Deviation (std): (0.5, 0.5, 0.5)
        Used for scaling the normalized images.

5. L1 Loss Weight (lambda_L1)

    Value: 100
        Balances the adversarial loss and L1 loss in the generator's total loss function.

6. PatchGAN Discriminator

    Kernel Size: 4
        Used in convolutional layers of the discriminator.
    Stride: 2 (for downsampling layers) and 1 (for the final layers).
    Padding: 1
        Ensures proper alignment of feature maps.

7. Training Epochs

    Value: 50
        Total number of iterations over the dataset.

8. Dropout Probability in Generator's UpSampling Layers

    Value: 0.5
        Applied conditionally in upsampling layers to prevent overfitting.

9. Activation Functions

    LeakyReLU:
        Negative slope: 0.2 (used in DownSample layers).
    ReLU:
        Used in UpSample layers.
    Tanh:
        Used as the final activation in the generator to scale output values to [-1, 1].

10. Dataset Parameters

    Dataset images must have the input and target images concatenated horizontally (side-by-side) for proper loading and splitting in the Pix2PixDataset class.



Result: We see that in the initial stages, the quality of the reconstructed image is very low, but the more epochs we do, the better the quality of the output images becomes. If we want the quality to be better than this, we can go up to 100 epochs, but due to memory limitations, it was not possible for me to do this or change the hyperparameters.
#According to the loss graph, we see that the loss for the Generator is decreasing with increasing epochs, but it takes a long time for the loss to decrease and the training process is not nearly stable.
