import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.face_detection import detect_face
from src.vit_model import Patches, PatchEncoder

def get_attention_map(model, img_path, target_size=(128, 128)):
    # 1. Load and Preprocess Image
    image = cv2.imread(img_path)
    if image is None: 
        print(f"Error: Could not find image at {img_path}")
        return None, None, None 
    
    face = detect_face(image)
    if face is None: 
        print("Error: No face detected in the image.")
        return None, None, None
    
    # Handle Channels (BGR to Gray if necessary)
    if len(face.shape) == 3 and face.shape[2] == 3:
        processed_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    else:
        processed_face = face
        
    processed_face = cv2.resize(processed_face, target_size)
    input_tensor = processed_face.astype("float32") / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=(0, -1)) # Shape: (1, 128, 128, 1)

    # 2. Extract Attention Weights
    try:
        # Check layer name - standard is 'multi_head_attention'
        # If your model summary shows a different name, change this string
        attn_layer = model.get_layer("multi_head_attention") 
        
        intermediate_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.output, attn_layer.output]
        )
        
        prediction, weights = intermediate_model.predict(input_tensor, verbose=0)
        
        # 3. Safe Weight Processing
        weights = np.array(weights) # Convert to numpy
        
        # Handle (Batch, Heads, Seq, Seq)
        if weights.ndim == 4:
            weights = np.mean(weights, axis=1)[0] # Average heads, remove batch
        elif weights.ndim == 3:
            weights = weights[0] # Just remove batch
            
        # Now weights should be (Seq, Seq). 
        # For ViT, row 0 is usually the [CLS] token's attention to all other patches.
        if weights.ndim == 2:
            # We take index 0 (CLS) and everything from 1 onwards (the patches)
            cls_attention = weights[0, 1:] 
        else:
            # Fallback if the layer returned a flattened sequence
            cls_attention = weights.flatten()[1:]

        # 4. Reshape to Grid
        num_patches = cls_attention.shape[0]
        grid_size = int(np.sqrt(num_patches))
        
        # Take only what fits in a square and reshape
        clean_attn = cls_attention[:grid_size*grid_size]
        heatmap = clean_attn.reshape((grid_size, grid_size))
        
        # 5. Resize and Normalize
        heatmap = cv2.resize(heatmap, target_size)
        
        # Normalize 0-255 with epsilon to avoid division by zero
        denom = (heatmap.max() - heatmap.min()) + 1e-8
        heatmap = (heatmap - heatmap.min()) / denom
        heatmap = np.uint8(255 * heatmap)
        
        return face, heatmap, prediction[0][0]

    except Exception as e:
        print(f"Error extracting attention: {e}")
        return None, None, None

def main():
    model_path = "models/liveness_vit.keras"
    # Using the path from your error log
    img_path = r"dataset/CASIA/Test/Colour/2_5.avi_100_fake.jpg"
    
    # Load model with custom layers
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={
            "Patches": Patches, 
            "PatchEncoder": PatchEncoder
        })
    except Exception as e:
        print(f"Model Load Error: {e}")
        return

    original_face, heatmap, score = get_attention_map(model, img_path)
    
    if original_face is not None:
        # Create visual overlay
        color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original_face_resized = cv2.resize(original_face, (128, 128))
        
        # If original_face_resized is grayscale, make it BGR for the overlay
        if len(original_face_resized.shape) == 2:
            original_face_resized = cv2.cvtColor(original_face_resized, cv2.COLOR_GRAY2BGR)
            
        overlay = cv2.addWeighted(original_face_resized, 0.6, color_heatmap, 0.4, 0)

        # Plotting
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.title("Detected Face")
        plt.imshow(cv2.cvtColor(original_face_resized, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title(f"ViT Attention Map\nLiveness Score: {score:.4f}")
        plt.imshow(heatmap, cmap='jet')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("XAI Overlay")
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()