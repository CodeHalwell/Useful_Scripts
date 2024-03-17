################################################################################################
#                                                                                              #
#                               TensorFlow Tabular Data                                        #
#                                                                                              #
################################################################################################

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import tensorflow.keras.activations as activations

tf_activation_functions = {
    'sigmoid': activations.sigmoid,
    'tanh': activations.tanh,
    'relu': activations.relu,
    'softmax': activations.softmax,
    'softplus': activations.softplus,
    'softsign': activations.softsign,
    'selu': activations.selu,
    'elu': activations.elu,
    'exponential': activations.exponential,
    'hard_sigmoid': activations.hard_sigmoid,
    'linear': activations.linear,
    'swish': activations.swish,
    'relu6': activations.relu6,
    'leaky_relu': activations.leaky_relu, # Note: requires an alpha value when called
    'prelu': activations.prelu, # Note: PReLU requires learnable parameters
    'gelu': activations.gelu
}

def create_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        # Assume a regression task; change the last layer accordingly for classification
        Dense(1)
    ])
    return model

# Example: Creating a model for input features of shape 10
model = create_model(input_shape=10)

# For regression tasks
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

# For classification tasks (binary classification example)
# model.compile(optimizer=Adam(learning_rate=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# Example parameters
epochs = 100
batch_size = 32

# Fit the model on the training data
history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val))

# The `history` object holds a record of the loss values and metric values during training
test_loss, test_metric = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Metric: {test_metric}')


################################################################################################
#                                                                                              #
#                               TensorFlow CNN                                                 #
#                                                                                              #
################################################################################################


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        # Convolutional layer 1
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        # Convolutional layer 2
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Flatten the output to feed into the dense layers
        Flatten(),

        # Dense layer 1
        Dense(128, activation='relu'),
        Dropout(0.5),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    return model


# Assuming you have your data ready in x_train, y_train, x_val, y_val, x_test, y_test
# For demonstration, these variables will be placeholders. Replace them with your actual data.

# Load and prepare your data here
# x_train, y_train = load_your_data()
# x_val, y_val = load_your_validation_data()
# x_test, y_test = load_your_test_data()

# Example: Create a CNN model for 28x28 grayscale images (1 channel) and 10 classes
model = create_cnn_model(input_shape=(28, 28, 1), num_classes=10)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use 'binary_crossentropy' for binary classification
              metrics=['accuracy'])

# Assuming your labels are integers, convert them to one-hot encoded vectors
# y_train = to_categorical(y_train, 10)
# y_val = to_categorical(y_val, 10)
# y_test = to_categorical(y_test, 10)

# Train the model
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=10,
                    validation_data=(x_val, y_val))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

