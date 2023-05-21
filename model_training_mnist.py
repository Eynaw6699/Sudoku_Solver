import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import joblib
import logging
from config import EPOCH, BATCH_SIZE, LEARNING_RATE

LOGGER = logging.getLogger(__name__)


# Set up model configurations
def model_config():
    LOGGER.info("initializing model configurations")
    inp_shape = (28, 28, 1)

    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=inp_shape, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))
    return model


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Reshape training and testing sets
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # Convert to float
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    # Set up model configurations
    model = model_config()

    E = EPOCH
    BS = BATCH_SIZE
    LR = LEARNING_RATE
    model.compile(optimizer=Adam(learning_rate=LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=E, batch_size=BS, verbose=1, validation_data=(x_test, y_test))
    model.save('model/model_mnist')
    LOGGER.info("Model saved")

    LOGGER.info("Plotting accuracy graph")
    plotting_data_dict = history.history

    test_loss = plotting_data_dict['val_loss']
    training_loss = plotting_data_dict['loss']
    test_accuracy = plotting_data_dict['val_accuracy']
    training_accuracy = plotting_data_dict['accuracy']

    epochs = range(1, len(test_loss) + 1)

    # Plot accuracy 
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.plot(epochs, test_loss, label='test_loss')
    plt.plot(epochs, training_loss, label='training_loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.plot(epochs, training_accuracy, label='training_accuracy')

    plt.legend()
    plt.savefig('accuracy.png', dpi=300)
    LOGGER.info("-----END training-----")


if __name__ == '__main__':
    main()
