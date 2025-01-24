# NLP_a2-Language-Modeling

## Overview

This assignment developed a language model using a text dataset of choice, which is the Star Wars - Thrawn Trilogy 01: Heir to the Empire by Timothy Zahn. The model is based on LSTM (Long Short-Term Memory) architecture, and it is capable of generating coherent and contextually relevant text based on a given input. Additionally, a web application was developed to allow users to interact with the trained model.

Task Breakdown
Task 1: Dataset Acquisition
The dataset chosen is Star Wars - Thrawn Trilogy 01: Heir to the Empire by Timothy Zahn. It is a text-rich dataset suitable for language modeling tasks due to its narrative text, dialogue, and descriptions.

Dataset Source: Hugging Face datasets library
Name: myothiha/starwars
Dataset Details:
Features: Contains a single text feature.
Train Split: 7,860 rows
Validation Split: 8,101 rows
Test Split: 9,236 rows
Task 2: Model Training
The LSTM-based language model is designed to generate text based on the input prompt. The model architecture consists of several layers:

Embedding Layer: Converts tokens to dense vectors.
LSTM Layer: Processes sequences and captures long-term dependencies.
Dropout Layer: Regularizes the model to prevent overfitting.
Fully Connected Layer: Predicts the next word in the sequence.
Model Architecture:
Input: Tokens converted into indices (numericalization).
Output: A prediction of the next word based on the input sequence.
Training Process: The model is trained using the Adam optimizer and CrossEntropyLoss. A ReduceLROnPlateau scheduler adjusts the learning rate based on validation performance.
Task 3: Text Generation Web Application
Features:
Input Box: Users can type a text prompt.
Temperature: A dropdown allows users to set the "temperature" value, which controls the creativity/randomness of the generated text. The values range from 0.5 to 1.0.
Generated Text: The model generates text based on the prompt provided by the user.
![image](https://github.com/user-attachments/assets/2da800d1-9818-44dc-ad17-4328f4f48f4d)
