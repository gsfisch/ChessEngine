import tensorflow as tf
from tensorflow import keras


def get_neural_network():
    model = keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='linear', input_shape=(8, 8, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='linear', input_shape=(8, 8, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='linear', input_shape=(8, 8, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='linear', input_shape=(8, 8, 1)),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(256, activation='linear'),
        #tf.keras.layers.Dense(256, activation='linear'),
        #tf.keras.layers.Dense(256, activation='linear'),
        #tf.keras.layers.Dense(256, activation='linear'),
        #tf.keras.layers.Dense(256, activation='linear'),
        #tf.keras.layers.Dense(256, activation='linear'),
        #tf.keras.layers.Dense(256, activation='linear'),
        #tf.keras.layers.Dense(256, activation='linear'),
        tf.keras.layers.Dense(2048, activation='linear'),
        tf.keras.layers.Dense(2048, activation='linear'),
        tf.keras.layers.Dense(2048, activation='linear'),
        tf.keras.layers.Dense(2048, activation='linear'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(#optimizer='adam',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    model.summary()

    return model
