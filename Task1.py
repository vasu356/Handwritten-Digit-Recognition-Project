import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.models import load_model
import tkinter as tk
import scipy
from tkinter import Canvas, Button, Label
from PIL import ImageGrab, Image

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_test = to_categorical(y_test, num_classes=10)

# Create a more complex CNN model with Batch Normalization
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

# Compile the model with a lower initial learning rate and a learning rate scheduler
def lr_scheduler(epoch):
    if epoch < 5:
        return 0.001  # High initial learning rate
    else:
        return 0.0001  # Lower learning rate

lr_callback = LearningRateScheduler(lr_scheduler)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

datagen.fit(x_train)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with augmented data, learning rate scheduler, and early stopping
model.fit(datagen.flow(x_train, y_train, batch_size=128),
          epochs=20,
          validation_data=(x_test, y_test),
          callbacks=[lr_callback, early_stopping])

# Save the model
model.save('mnist_model.h5')

# Load the model for handwriting recognition
handwriting_model = load_model('mnist_model.h5')

class DigitRecognition:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition")
        self.canvas = Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.pack()
        self.label = Label(self.root, text="Draw a digit and click predict")
        self.label.pack()
        self.clear_button = Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()
        self.predict_button = Button(self.root, text="Predict", command=self.predict_digit)
        self.predict_button.pack()
        self.image = np.zeros((280, 280), dtype=np.uint8)
        self.drawing = False
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill='black', width=10)
            self.image = cv2.line(self.image, (self.last_x, self.last_y), (x, y), 255, 10)
            self.last_x, self.last_y = x, y

    def stop_draw(self, event):
        self.drawing = False

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = np.zeros((280, 280), dtype=np.uint8)

    def preprocess_image(self):
        resized_image = cv2.resize(self.image, (28, 28))
        return resized_image.reshape(1, 28, 28, 1).astype('float32') / 255.0

    def predict_digit(self):
        preprocessed_image = self.preprocess_image()
        prediction = handwriting_model.predict(preprocessed_image)
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit] * 100
        self.label.config(text=f"Predicted Digit : {predicted_digit}\nConfidence : {confidence:.2f}%")

if __name__ == "__main__":
    import cv2
    root = tk.Tk()
    app = DigitRecognition(root)
    root.mainloop()
