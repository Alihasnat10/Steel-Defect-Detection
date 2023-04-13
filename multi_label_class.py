from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics 
from utils import plot_loss_curves


class TrainClassifier:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.class_weights = {0: 6.23,
                1: 23.87,
                2: 1.06,
                3: 6.73}
    
    def generate_data(self, batch_size, target_size):
        self.df['ClassId'] = self.df['ClassId'].astype(str)

        train_datagen = ImageDataGenerator(rescale=1/255.,
                                        validation_split = 0.2)
                                        
        train_data = train_datagen.flow_from_dataframe(dataframe=self.df,
                                                    directory='data/train_images',
                                                    target_size=target_size, x_col='ImageId',
                                                    y_col='ClassId', batch_size=batch_size,
                                                    class_mode='categorical', subset = "training",
                                                    shuffle=True)
        valid_data = train_datagen.flow_from_dataframe(dataframe=self.df,
                                                    directory='data/train_images',
                                                    target_size=target_size, x_col='ImageId',
                                                    y_col='ClassId', batch_size=batch_size,
                                                    class_mode='categorical', subset = "validation",
                                                    shuffle=True)
        return train_data, valid_data

    def define_model(self, input_shape):
        model = Sequential([
            Conv2D(10, 3, activation='relu', input_shape=input_shape),
            Conv2D(10, 3, activation='relu'),
            MaxPool2D(),
            Conv2D(10, 3, activation='relu'),
            Conv2D(10, 3, activation='relu'),
            MaxPool2D(),
            Flatten(),
            Dense(4, activation='softmax') 
        ])

        # Compile the model
        model.compile(loss="categorical_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])
        return model
    
    def train(self, train_data, valid_data, epochs, input_shape, save=True):
        model = self.define_model(input_shape)
        history = model.fit(train_data,
                    epochs=epochs,
                    steps_per_epoch=len(train_data),
                    validation_data=valid_data,
                    validation_steps=len(valid_data),
                    class_weight=self.class_weights)

        if save:
            model.save("models/defect_classify")
        return history, model
    
    def evaluate(self, data, model):
        actual = data.classes
        predicted = model.predict(data)

        predicted_new = []
        for i in range(len(predicted)):
            predicted_new.append(np.argmax(predicted[i]))

        confusion_matrix = metrics.confusion_matrix(actual, predicted_new)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1, 2, 3])


        cm_display.plot()
        plt.show() 
        return confusion_matrix

if __name__ == "__main__":

    classifier_config = {
        "epochs": 5,
        "batch_size": 16,
        "target_size": (128, 800),
        "model_input_shape": (128, 800, 3)
    }

    classifier = TrainClassifier(csv_path='data/train.csv')

    train_data, valid_data = classifier.generate_data(classifier_config["batch_size"], classifier_config["target_size"])
    
    history, model = classifier.train(train_data, valid_data, classifier_config["epochs"], classifier_config["model_input_shape"])
    
    plot_loss_curves(history)

    _ = classifier.evaluate(valid_data, model)
    