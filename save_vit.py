import tensorflow as tf
from src.vit_model import create_vit_backbone # Import from your src folder
from keras import layers, Model

def build_and_save_vit():
    # 1. Define Input
    inputs = layers.Input(shape=(128, 128, 1)) # Match your CNN/Hybrid input
    
    # 2. Create the backbone using your vit_model.py script
    representation = create_vit_backbone(inputs)
    
    # 3. Add the final classification head
    outputs = layers.Dense(1, activation="sigmoid", name="liveness_output")(representation)
    
    # 4. Construct the Model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Optional: Load your trained weights if you have a .h5 or checkpoint file
    # model.load_weights("path_to_weights.h5") 
    
    # 5. Save it to the models folder
    model.save("models/liveness_vit.keras")
    print("Success: ViT model saved to models/liveness_vit.keras")

if __name__ == "__main__":
    build_and_save_vit()