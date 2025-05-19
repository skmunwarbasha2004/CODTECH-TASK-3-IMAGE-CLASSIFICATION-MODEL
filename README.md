# CODTECH-TASK-3-IMAGE-CLASSIFICATION-MODEL
COMPANY : CODTECH IT SOLUTION

NAME : SHAIK MUNWAR BASHA

INTERN ID : CT06DM431

DOMAIN : Machine Learning

MENTOR : Neela Santosh

DURATION : 6 weeks

Building a Convolutional Neural Network (CNN) for Image Classification Using TensorFlow<br/>

Image classification is a computer vision task where a model identifies the category of an image from a set of predefined labels. In this case, the CIFAR-10 dataset contains 60,000 32x32 color images across 10 classes (e.g., airplane, automobile, bird). The goal is to train a model to classify these images accurately. The code uses a Convolutional Neural Network (CNN) built with TensorFlow, a popular deep learning framework, to achieve this.<br/>

Procedure and Code Explanation<br/>
1.Loading and Preparing the Dataset: The CIFAR-10 dataset is loaded using tensorflow.keras.datasets.cifar10.load_data(), providing 50,000 training images (X_train) and 10,000 test images (X_test), each of size 32x32x3 (height, width, RGB channels). The labels (y_train, y_test) are integers from 0 to 9, corresponding to the 10 classes. The labels are reshaped from a 2D array (e.g., (50000, 1)) to a 1D array (e.g., (50000,)) for compatibility with the model. A plot_sample function visualizes sample images, labeling them with their class (e.g., "airplane" for y_train[0]=6).<br/>
2.Normalization: Pixel values in the images range from 0 to 255. Dividing X_train and X_test by 255 normalizes them to the range [0, 1], which helps the model converge faster during training by standardizing input values.<br/>
3.Initial Attempt with an Artificial Neural Network (ANN): An ANN is first built using models.Sequential with a Flatten layer to convert the 32x32x3 images into a 1D vector (3072 elements), followed by two dense layers (3000 and 1000 neurons with ReLU activation), and a final dense layer with 10 neurons (one per class) using softmax activation for probability outputs. The model is compiled with the SGD optimizer, sparse_categorical_crossentropy loss (suitable for integer labels), and accuracy as the metric. After 5 epochs, the ANN achieves ~49% accuracy, indicating it struggles with image data due to its inability to capture spatial patterns<br/>.
4.Building the CNN: A CNN is better suited for images as it preserves spatial relationships. The CNN model is built with:
Two Conv2D layers (32 and 64 filters, 3x3 kernels, ReLU activation) to extract features like edges and textures.
Two MaxPooling2D layers (2x2) to reduce spatial dimensions (e.g., from 32x32 to 16x16, then 8x8), lowering computation while retaining key features.
A Flatten layer to convert the 2D feature maps into a 1D vector.
A Dense layer (64 neurons, ReLU) for high-level reasoning, and a final Dense layer (10 neurons, softmax) for classification. The model is compiled with the Adam optimizer (faster convergence than SGD) and the same loss and metrics. Training for 10 epochs yields a significant improvement, reaching ~78% accuracy on the training set and ~70% on the test set (cnn.evaluate(X_test, y_test)).<br/>
5.Prediction and Evaluation: The CNN predicts probabilities for each test image (y_pred), which are converted to class labels using np.argmax (e.g., [3, 8, 8, 8, 4] for the first 5 images). Comparing these to the true labels (y_test), we see some correct predictions (e.g., index 0: predicted 3, actual 3) and errors (e.g., index 3: predicted 8, actual 0). The plot_sample function visualizes these predictions, and classes[y_classes[3]] ("airplane") shows the predicted class for the fourth test image.<br/>
How TensorFlow Facilitates This<br/>
TensorFlow, with its Keras API, simplifies building and training neural networks. It provides high-level abstractions like Sequential for stacking layers, Conv2D and MaxPooling2D for CNN-specific operations, and utilities like datasets for easy data loading. TensorFlow handles the underlying computations (e.g., backpropagation, optimization) efficiently, supporting both CPU and GPU acceleration, making it ideal for image classification tasks like this.


SAMPLES CIFAR-10 DATASET:

![Image](https://github.com/user-attachments/assets/8044b6f0-9868-44ae-9084-bf37fda38268)




#OUTPUT

![Image](https://github.com/user-attachments/assets/c9ef285f-10c3-402f-9553-78b79b535c76)

![Image](https://github.com/user-attachments/assets/d121be37-b5c2-4c84-89af-675e5aeee0a8)
