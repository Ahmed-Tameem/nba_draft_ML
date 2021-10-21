import matplotlib.pyplot as plt
from scrape import scrape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.initializers import he_normal
from tensorflow.keras import optimizers

def plot_training_history(history, title_str=''):
    """ Plot Keras training history

    Inputs:
        history: History object
    
    Returns:
        None
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy ' + title_str)

    ymin = min([min(acc), min(val_acc)])

    axes = plt.gca()
    axes.set_xlim([0, len(acc)])
    axes.set_ylim([ymin, 1])

    plt.legend(loc=0)
    plt.figure()
    plt.show
    
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, loss, 'b', label='Validation loss')
    plt.title('Training and validation loss ' + title_str)

    ymin = min([min(loss), min(val_loss)])

    axes = plt.gca()
    axes.set_xlim([0, len(loss)])
    axes.set_ylim([ymin, 1])

    plt.legend(loc=0)
    plt.show()


X, Y = scrape()

X_train = X.T[:456].T
X_test = X.T[456:].T

Y_train = Y[:456]
Y_test = Y[456:]

model = Sequential([
    Input(X.shape[0]),
    Dense(10, kernel_initializer=he_normal()),
    Activation("relu"),
    Dense(100, kernel_initializer=he_normal()),
    Activation("relu"),
    Dense(80, kernel_initializer=he_normal()),
    Activation("relu"),
    Dense(60, kernel_initializer=he_normal()),
    Activation("relu"),
    Dense(40, kernel_initializer=he_normal()),
    Activation("relu"),
    Dense(20, kernel_initializer=he_normal()),
    Activation("relu"),
    Dense(1, kernel_initializer=he_normal()),
    Activation("sigmoid"),])

optimizer = optimizers.SGD(lr=0.0008)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the model
history = model.fit(X_train.T, 
                    Y_train.T, 
                    validation_data=(X_train.T, Y_train.T), 
                    epochs=100)

plot_training_history(history)
