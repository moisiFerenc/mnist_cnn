import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the MNIST dataset from a CSV file
mnist_data = pd.read_csv('train_data/mnist_train.csv')
mnist_data = mnist_data.sample(frac=1).reset_index(drop=True)  # Shuffle the data
X = mnist_data.iloc[:, 1:].values
Y = mnist_data.iloc[:, 0].values
X = X / 255.0  # Normalize pixel values to the range [0, 1]

# Split the data into training and validation sets
split_ratio = 0.9
split_index = int(split_ratio * len(X))
X_train, X_dev = X[:split_index], X[split_index:]
Y_train, Y_dev = Y[:split_index], Y[split_index:]

# Reshape the data for convolutional network
X_train = X_train.reshape(-1, 28, 28, 1)
X_dev = X_dev.reshape(-1, 28, 28, 1)

# Define a simple CNN model
def create_cnn_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training function using the CNN
def train_cnn(X_train, Y_train, epochs):
    model = create_cnn_model()
    model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_dev, Y_dev))
    return model

# Function to evaluate the accuracy of the CNN
def evaluate_cnn(model, X, Y):
    _, accuracy = model.evaluate(X, Y, verbose=0)
    return accuracy

# Function to make predictions using the CNN
def test_cnn_prediction(index, model):
    current_image = X_dev[index].reshape(28, 28) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

    predictions = model.predict(X_dev[index:index + 1])
    predicted_label = np.argmax(predictions)
    true_label = Y_dev[index]
    print("Prediction:", predicted_label)
    print("True Label:", true_label)

# Modify the 'menu' function to train and test the CNN
def menu_cnn():
    model = None
    while True:
        print("*********Menu*********\n")
        print("1. Train CNN\n")
        print("2. Test CNN\n")
        print("3. Accuracy\n")
        print("4. Exit\n")
        print("Select an option:\n")

        choice = input()

        if choice == '1':
            print("Option 1. \n Input epochs:")
            epochs = int(input())
            model = train_cnn(X_train, Y_train, epochs)

        elif choice == '2':
            if model is not None:
                print("Option 2. Testing CNN")
                index = random.randint(0, len(X_dev) - 1)
                test_cnn_prediction(index, model)
            else:
                print("Please train the CNN first.")

        elif choice == '3':
            if model is not None:
                print("Option 3. Accuracy:")
                accuracy = evaluate_cnn(model, X_dev, Y_dev)
                print("Accuracy: {:.2f}%".format(accuracy * 100))
            else:
                print("Please train the CNN first.")

        elif choice == '4':
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please select a valid option.")

# Call the modified menu function
menu_cnn()
