from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from src.cnn_model import create_cnn_backbone, get_data_augmentation
from src.vit_model import PatchEncoder, Patches, create_vit_backbone


def create_hybrid_model():
    inputs = layers.Input(shape=(128, 128, 1))

    # Apply data augmentation
    augmented_inputs = get_data_augmentation()(inputs)

    cnn_features = create_cnn_backbone(augmented_inputs)
    vit_features = create_vit_backbone(augmented_inputs)

    fused = layers.Concatenate(name="feature_fusion")([cnn_features, vit_features])
    fused = layers.Dense(128, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)
    outputs = layers.Dense(1, activation="sigmoid")(fused)

    model = models.Model(inputs=inputs, outputs=outputs, name="hybrid_cnn_vit_liveness")
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_hybrid_model(model_path, compile=True):
    return models.load_model(
        model_path,
        compile=compile,
        custom_objects={
            "Patches": Patches,
            "PatchEncoder": PatchEncoder,
        },
    )
