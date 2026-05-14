from pathlib import Path
import tempfile
from keras import layers, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Imports for both model architectures
from src.hybrid_model import create_hybrid_model
from src.vit_model import create_vit_backbone

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, epochs=30, batch_size=32, model_type="hybrid"):
    
    # 1. Select and Build the Model Architecture
    if model_type.lower() == "vit":
        print("[*] Building Pure ViT Model...", flush=True)
        inputs = layers.Input(shape=(128, 128, 1))
        representation = create_vit_backbone(inputs)
        outputs = layers.Dense(1, activation="sigmoid", name="vit_liveness")(representation)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        model_path = Path("models/liveness_vit.keras")
    else:
        print("[*] Building Hybrid CNN-ViT Model...", flush=True)
        model = create_hybrid_model()

        # FIX 1: Hybrid model was likely compiled with learning_rate=1e-3 inside
        # create_hybrid_model(). We recompile here with a lower rate to prevent
        # the model from overfitting in the first epoch.
        model.compile(
            optimizer=Adam(learning_rate=1e-4),  # lowered from 1e-3
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        model_path = Path("models/liveness_hybrid.keras")

    print(f"Train samples: {len(X_train)}", flush=True)
    print(f"Validation samples: {len(X_val)}", flush=True)
    print(f"Test samples: {len(X_test)}", flush=True)

    # 2. Setup Directory
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # FIX 2: Added ReduceLROnPlateau alongside EarlyStopping.
    # When validation loss stops improving, this halves the learning rate
    # instead of immediately stopping — giving the model more chances to recover
    # rather than quitting after 6 flat epochs.
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=7,                  # FIX 3: increased from 6 to 7
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,                  # halve the LR when stuck
            patience=3,                  # try for 3 epochs before reducing
            min_lr=1e-6,                 # don't go below this
            verbose=1,
        )
    ]
    
    print(f"[*] Training {model_type.upper()} Model:")
    model.summary()
    
    # 3. Class Weights — FIX 4: more aggressive boosting of live class
    # Your dataset has ~1142 spoof vs ~393 live samples in test alone,
    # meaning spoof is ~3x more frequent. 'balanced' handles this,
    # but we multiply the live weight by 3.0 instead of 2.0 to push
    # the model harder toward correctly classifying live faces.
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    
    for key in list(class_weight_dict.keys()):
        if int(key) == 1:
            class_weight_dict[key] *= 3.0  # FIX 4: increased from 2.0 to 3.0
            
    print(f"[*] Computed class weights: {class_weight_dict}")

    # FIX 5: Reduced batch size from 32 to 16.
    # Smaller batches mean more gradient updates per epoch and expose
    # the model to more varied live/spoof combinations per pass,
    # which helps with the class imbalance problem.
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,                   # FIX 5: reduced from 32 to 16
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    # 4. Evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    # 5. Serialization Workaround
    temp_dir = Path(".tmp/keras_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temp_dir.resolve())

    # 6. Save the model
    model.save(model_path)
    
    print(f"\n--- {model_type.upper()} Results ---")
    print(f"Validation Accuracy: {val_accuracy:.4f}", flush=True)
    print(f"Test Accuracy:       {test_accuracy:.4f}", flush=True)
    print(f"Model saved to:      {model_path}", flush=True)

    return model, history