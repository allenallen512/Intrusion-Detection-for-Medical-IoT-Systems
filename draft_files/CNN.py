import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# important sources:
# https://www.tensorflow.org/tutorials/images/cnn

def CNN():
    data = pd.read_csv("data.csv")

    # drop non numeric columns and label
    X = data.select_dtypes(include=['float64', 'int64']).drop('Label', axis=1)
    y = data['Label']

    le = LabelEncoder()
    y = le.fit_transform(y)
    y_categorical = to_categorical(y)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)


    num_features = X_scaled.shape[1]
    grid_size = int(np.ceil(np.sqrt(num_features)))
    X_padded = np.zeros((X_scaled.shape[0], grid_size * grid_size))
    X_padded[:, :num_features] = X_scaled
    X_reshaped = X_padded.reshape(-1, grid_size, grid_size, 1)

    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.3, random_state=42)



    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(grid_size, grid_size, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_categorical.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    input_layer = Input(shape=(grid_size, grid_size, 1))
    x = Conv2D(32, kernel_size=(3,3), activation='relu')(input_layer)
    print("After first Conv2D:", x.shape)

    x = MaxPooling2D(pool_size=(2,2))(x)
    print("After first MaxPooling2D:", x.shape)

    x = Conv2D(64, kernel_size=(2,2), activation='relu')(x)  # Adjusted kernel size
    print("After second Conv2D:", x.shape)

    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

CNN()