# ...

# Importáljuk a szükséges könyvtárakat és modulokat
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Az MNIST adathalmaz betöltése egy CSV fájlból
mnist_data = pd.read_csv('train_data/mnist_train.csv')

# Az adatokat véletlenszerűen összekeverjük, hogy ne legyen minta sorrendi hatás
mnist_data = mnist_data.sample(frac=1).reset_index(drop=True)

# Az X változóba helyezzük az adatok képeket (pixel értékeket)
X = mnist_data.iloc[:, 1:].values

# Az Y változóba helyezzük az adatok címkéit (osztályokat)
Y = mnist_data.iloc[:, 0].values

# A pixel értékeket normalizáljuk, hogy a tartomány [0, 1] legyen
X = X / 255.0

# Az adatokat felosztjuk tréning és validációs halmazra
split_ratio = 0.9  # Az adatok 90%-a tréninghalmazban lesz
split_index = int(split_ratio * len(X))
X_train, X_dev = X[:split_index], X[split_index:]
Y_train, Y_dev = Y[:split_index], Y[split_index:]

# Az adatokat átalakítjuk, hogy megfeleljenek a konvolúciós hálózat számára
X_train = X_train.reshape(-1, 28, 28, 1)
X_dev = X_dev.reshape(-1, 28, 28, 1)

# Egyszerű konvolúciós modell létrehozása
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

# Az adatok tréningje a konvolúciós hálózaton
def train_cnn(X_train, Y_train, epochs):
    model = create_cnn_model()
    model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_dev, Y_dev))
    return model

# A konvolúciós modell pontosságának kiértékelése
def evaluate_cnn(model, X, Y):
    _, accuracy = model.evaluate(X, Y, verbose=0)
    return accuracy

# Az eredmények megjelenítésére és tesztelésre szolgáló funkció
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

# A menü funkció módosítása a konvolúciós hálózat tréningjére és tesztelésére
def menu_cnn():
    model = None
    while True:
        print("*********Menü*********\n")
        print("1. CNN tréning\n")
        print("2. CNN teszt\n")
        print("3. Pontosság\n")
        print("4. Kilépés\n")
        print("Válasszon egy lehetőséget:\n")

        choice = input()

        if choice == '1':
            print("1. lehetőség. Kérem adja meg az epoch-ok számát:")
            epochs = int(input())
            model = train_cnn(X_train, Y_train, epochs)

        elif choice == '2':
            if model is not None:
                print("2. lehetőség. CNN tesztelése")
                index = random.randint(0, len(X_dev) - 1)
                test_cnn_prediction(index, model)
            else:
                print("Kérem először tréningezze a CNN-t.")

        elif choice == '3':
            if model is not None:
                print("3. lehetőség. Pontosság kiértékelése:")
                accuracy = evaluate_cnn(model, X_dev, Y_dev)
                print("Pontosság: {:.2f}%".format(accuracy * 100))
            else:
                print("Kérem először tréningezze a CNN-t.")

        elif choice == '4':
            print("Kilépés a programból.")
            break

        else:
            print("Érvénytelen választás. Kérem válasszon egy érvényes lehetőséget.")

# Hívja meg a módosított menü funkciót
menu_cnn()
