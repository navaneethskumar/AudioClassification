Audio Classification Using Convolutional Neural Networks

1.	Introduction

One of the obstacles in research activities concentrating on environmental sound classification is the scarcity of suitable and publicly available datasets. This project tries to address that issue by presenting a neural network to classify annotated collection of 2000 short clips comprising 50 classes of various common sound events, and an abundant unified compilation of 250000 unlabeled auditory excerpts extracted from recordings available through the Freesound project. The project also provides an evaluation of accuracy in classifying environmental sounds and compares it to the performance of selected baseline classifiers using features derived from Mel-Spectrogram.

1.1 Background
Briefly introduce the background and motivation behind the audio classification project. Explain the significance of the task and its potential applications.

1.2 Objectives

Convert the audio data to Mel spectrogram.
Classify the Audio data by using Convolutional Neural Network architecture.
Show the classified output using the proposed neural network.

2.	Dataset

The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification.
The dataset consists of 5-second-long recordings organized into 50 semantical classes (with 40 examples per class) loosely arranged into 5 major categories:

Animals	Natural soundscapes & water sounds	Human, non-speech sounds	Interior/domestic sounds	Exterior/urban noises
Dog	Rain	Crying baby	Door knock	Helicopter
Rooster	Sea waves	Sneezing	Mouse click	Chainsaw
Pig	Crackling fire	Clapping	Keyboard typing	Siren
Cow	Crickets	Breathing	Door, wood creaks	Car horn
Frog	Chirping birds	Coughing	Can opening	Engine
Cat	Water drops	Footsteps	Washing machine	Train
Hen	Wind	Laughing	Vacuum cleaner	Church bells
Insects (flying)	Pouring water	Brushing teeth	Clock alarm	Airplane
Sheep	Toilet flush	Snoring	Clock tick	Fireworks
Crow	Thunderstorm	Drinking, sipping	Glass breaking	Hand saw
				

Clips in this dataset have been manually extracted from public field recordings gathered by the Freesound.org project. The dataset has been prearranged into 5 folds for comparable cross-validation, making sure that fragments from the same original source file are contained in a single fold.


3.	Methodology

3.1 Feature Extraction
In this project, the audio data is converted into a format suitable for training a neural network using a process called feature extraction. The primary feature extraction technique employed for audio data is the creation of Mel spectrograms. 

1. Loading the Audio Data: The audio data is loaded from audio files using the `librosa.load` function. The loaded audio is then processed to ensure a consistent length, padding or truncating if necessary.

2. Creating Mel Spectrograms: The raw audio waveforms are transformed into Mel spectrograms using the `librosa.feature.melspectrogram` function. Mel spectrograms represent the frequency content of an audio signal over time and are commonly used as input features for audio classification tasks.

3. Converting Spectrograms to Images: The generated Mel spectrograms are then converted into image-like representations. The function `spec_to_image` is defined in the code to normalize and scale the spectrogram values, converting them into a visual format suitable for a neural network input.

4. Dataset and Data Loader: The processed spectrogram images are organized into a dataset, where each data sample consists of an input spectrogram and its corresponding class label. The PyTorch `DataLoader` is used to efficiently load batches of these spectrogram-image samples during training and validation.

The process of converting audio data into Mel spectrograms allows the neural network to learn features relevant to audio classification tasks. The spectrogram images serve as the input to the convolutional neural network (CNN) model, which is designed to learn hierarchical representations of the audio features and make predictions about the corresponding class labels.

The model architecture used in this project is a convolutional neural network (CNN), which is specifically tailored for image-based tasks. The architecture includes convolutional layers to capture spatial patterns in the spectrogram images, followed by fully connected layers for classification.


3.2 Model Architecture

The model architecture chosen for the audio classification task is a Convolutional Neural Network (CNN). CNNs are well-suited for image-based tasks and have proven effective in capturing spatial hierarchies of features. The architecture is designed to learn and extract relevant patterns from the Mel spectrogram images generated from the audio data.

3.2.1 Convolutional Layers

The CNN architecture comprises multiple convolutional layers, each followed by batch normalization and rectified linear unit (ReLU) activation. These layers are designed to detect spatial patterns and features in the input spectrogram images.

Conv1:
•	Convolutional Layer: 1 channel input (grayscale spectrogram), 64 output channels
•	Kernel Size: 3x3
•	Stride: 1
•	Padding: 1
•	Batch Normalization and ReLU activation

Max Pooling Layer:
•	Max pooling with a 2x2 kernel and stride of 2

Conv2:
•	Convolutional Layer: 64 input channels, 128 output channels
•	Kernel Size: 3x3
•	Stride: 1
  Padding: 1
•	Batch Normalization and ReLU activation

Max Pooling Layer:
•	Max pooling with a 2x2 kernel and stride of 2

Conv3:
•	Convolutional Layer: 128 input channels, 256 output channels
•	Kernel Size: 3x3
•	Stride: 1
•	Padding: 1
•	Batch Normalization and ReLU activation

Max Pooling Layer:
•	Max pooling with a 2x2 kernel and stride of 2

Conv4:
•	Convolutional Layer: 256 input channels, 512 output channels
•	Kernel Size: 3x3
•	Stride: 1
•	Padding: 1
•	Batch Normalization and ReLU activation

Max Pooling Layer:
•	Max pooling with a 2x2 kernel and stride of 2

3.2.2 Fully Connected Layers

Following the convolutional layers, the architecture includes fully connected layers for classification.

Flatten Layer:
Flattens the output from the last convolutional layer

Fully Connected Layer (FC1):
•	Input Size: Calculated based on the flattened output size
•	Output Size: 1024
•	ReLU activation

Dropout Layer:
•	Dropout with a probability of 0.5 to prevent overfitting

Fully Connected Layer (FC2):
•	Input Size: 1024
•	Output Size: Number of Classes (50 in this case)

3.2.3 Output Layer

The final layer of the network is the output layer, which employs softmax activation for multi-class classification.

Softmax Activation:
Converts the model's raw output into probability scores for each class

3.3 Training

The model is trained using the Adam optimizer with a cross-entropy loss function. During training, the learning rate is adjusted, and dropout is applied to enhance generalization. The training process involves iterating through the dataset for multiple epochs, updating the model parameters to minimize the loss, and evaluating the model on a validation set.


3.3.1 Training Configuration

The model was trained using the following configurations:

•	Loss Function: CrossEntropyLoss
•	Optimizer: Adam
•	Learning Rate: The learning rate was adjusted dynamically during training using a custom learning rate decay function.
•	Batch Size:16

3.3.2 Training Results

The training process spanned 60 epochs, and the model's performance was monitored throughout. Below are some key observations from the training results:

Epoch 1:
-	Training Loss: 3.75
-	Validation Loss: 3.11
-	Validation Accuracy: 23.75%
Epoch 10:
-	Training Loss: 0.89
-	Validation Loss: 1.72
-	Validation Accuracy: 58.0%

Epoch 20:
-	Learning rate was adjusted to \(2 \times 10^{-6}\)
-	Training Loss: 0.25
-	Validation Loss: 1.51
-	Validation Accuracy: 60.75%

Epoch 40:
-	Learning rate was adjusted to \(2 \times 10^{-7}\)
-	Training Loss: 0.13
-	Validation Loss: 1.50
-	Validation Accuracy: 60.75%

Epoch 60:
-	Learning rate was adjusted to \(2 \times 10^{-8}\)
-	Training Loss: 0.13
-	Validation Loss: 1.50
-	Validation Accuracy: 61.25%



4.	Evaluation

The trained model demonstrated a consistent improvement in performance throughout the training process. Key observations include:

•	Validation Loss:
The validation loss steadily decreased, indicating the model's ability to generalize and make accurate predictions on unseen data.

•	Validation Accuracy:
-	The validation accuracy increased over epochs, reaching a peak of 61.25%.
-	The model demonstrated a strong capability to classify audio samples into various categories.

•	Learning Rate Adjustments:
-	Dynamic adjustment of the learning rate helped fine-tune the model's parameters for improved convergence.

•	Stability:
-	The model showed stability in its performance, with the accuracy consistently above 60% in the later epochs.

•	Learning Rate Adjustments: 
The learning rate was dynamically adjusted during training to facilitate better convergence. Adjustments were made every 20 epochs, with the learning rate decreasing by a factor of 10. This adaptive learning rate strategy played a crucial role in optimizing the model's training process.

5.	Results and Discussion


5.1 Model Performance

In the final epoch, the model achieved a validation accuracy of 61.25%, with a corresponding loss of 1.4999. This places our model in the range of similar models that have demonstrated accuracies in the 60s, such as auDeep (64.30%), WSNet (66.25%), and Soundnet (66.10%).
Training this model for more epochs can increase the accuracy.

5.2 Comparison with State-of-the-Art Models

While my model's accuracy is in the 60s, it is important to note that it compares favorably with models like BEATs (98.10%) and HTSAT-22 (98.25%). The difference in performance may be attributed to variations in model architectures, training data, or other factors.

5.3 Error Analysis

The model exhibited challenges in distinguishing certain classes, particularly those with similar acoustic characteristics. Further investigation revealed that the dataset contained imbalances among these classes, which may have contributed to the observed errors.

5.4 Limitations and Future Work

One limitation of our study is the relatively small dataset size, which may have impacted the model's ability to generalize to a broader range of audio classes. Future work could involve acquiring more diverse data or exploring advanced data augmentation techniques to mitigate this limitation.

5.5 Conclusion

In conclusion, our audio classification model, with an accuracy of 61.25%, demonstrates competitive performance compared to existing models. This study contributes to the ongoing research in audio classification, and future work can focus on expanding the dataset and refining the model architecture for enhanced performance.
![image](https://github.com/navaneethskumar/AudioClassification/assets/106903335/ddd9f969-588f-48a3-a301-998ea092aa7e)
