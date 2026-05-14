import tensorflow as tf
from keras import layers, models, regularizers


def get_data_augmentation():
    """Reusable data augmentation block for training."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")


def create_cnn_backbone(inputs):
    """Optimized CNN backbone with Batch Normalization and L2 Regularization."""
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Flatten()(x)
    
    # Dense Block
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    
    return x


def create_cnn():
    inputs = layers.Input(shape=(128, 128, 1))
    
    # Apply data augmentation
    x = get_data_augmentation()(inputs)
    
    # Pass augmented input to backbone
    features = create_cnn_backbone(x)
    outputs = layers.Dense(1, activation="sigmoid")(features)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_liveness")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = create_cnn()
    model.summary()
