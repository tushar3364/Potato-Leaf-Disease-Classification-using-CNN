# Potato-Leaf-Disease-Classification-using-CNN
### Problem Statement :
Farmers every year face economic loss and crop waste due to various diseases in potato plants. Early blight and Late blight are major disease of potato leaf. It is estimated that the major loss occurred in potato yield is due to these diseases. Thus the images are classified into 3 classes :

Potato Late Blight,                                                            
Potato Early Blight,                                                        
Potato Healthy Leaf.      

The process involves several steps, including importing libraries, loading and viewing data, splitting the data into training and validation sets, data pre-processing, building the CNN model, training the model, and analyzing the model's performance.


Dataset is available at Kaggle : https://www.kaggle.com/datasets/arjuntejaswi/plant-village

### Step 1: Importing Libraries and Data : In this step, we import the necessary libraries such as Pandas, NumPy, and TensorFlow. We also load the potato leaf images from the PlantVillage directory using tf.keras.preprocessing.image_dataset_from_directory. This function automatically loads the images from the directory and creates an image dataset along with their corresponding labels.

### Step 2: Viewing Data Images : We visualize and inspect a few samples of the loaded data to ensure that the images are correctly loaded and to get an understanding of the dataset's structure.

### Step 3: Splitting the Data : We split the dataset into training and validation sets. The training set is used to train the CNN model, while the validation set is used to evaluate the model's performance on unseen data.

### Step 4: Data Pre-processing : Data pre-processing involves several steps to prepare the images for training. These steps include resizing and rescaling the images to a consistent size, data augmentation to increase the dataset's diversity and generalization, and other normalization techniques.

### Step 5: Model Building : In this step, we build the CNN model using TensorFlow's Keras API. The model consists of Convolutional layers (Conv2D), MaxPooling layers to reduce the spatial dimensions, and other necessary layers. We optimize the model using the Adam optimizer and use Sparse Categorical Crossentropy as the loss function, as we have multiple classes and integer-encoded labels. We also define accuracy metrics to monitor the model's performance during training.

### Step 6: Model Training and Analysis : We train the CNN model on the training set and evaluate its performance on the validation set. During training, we monitor metrics such as training accuracy, validation accuracy, training loss, and validation loss to analyze the model's behavior and identify potential overfitting.

### Step 7: Predicting on New Images (Unknown Leaf) : Finally, we plot and analyze images from a new directory called "Unknown Leaf," which contains leaf images downloaded from the internet. We use the trained CNN model to predict the class (Potato Late Blight, Potato Early Blight, or Potato Healthy Leaf) of each unknown leaf image.

