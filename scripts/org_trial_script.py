import tensorflow.compat.v1 as tf
import numpy as np
import cv2
from PIL import Image
import os

# --- FILE PATHS (CORRECTED) ---
# 1. Path to your downloaded frozen graph file (.pb)
# Corrected path to look inside the nested folder.
MODEL_PATH = "../models/20180402-114759/20180402-114759.pb"

# 2. Path to your image (Confirmed to be sample.png)
IMAGE_PATH = "../images/sample.png" 

# --- MODEL CONSTANTS ---
INPUT_TENSOR_NAME = 'input:0'
OUTPUT_TENSOR_NAME = 'embeddings:0'
PHASE_TRAIN_TENSOR_NAME = 'phase_train:0'
IMAGE_SIZE = 160 # The model expects 160x160 input

# --- Helper Function: Image Pre-processing ---
def pre_process_image(image_path, size=IMAGE_SIZE):
    """
    Loads an image and preprocesses it for FaceNet (resizing and pre-whitening).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # 1. Load and Resize
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    
    # 2. Convert to Array and Normalize (Pre-whitening)
    image_array = np.array(img, dtype=np.float32)
    
    mean = np.mean(image_array)
    std = np.std(image_array)
    std_adj = np.maximum(std, 1.0 / np.sqrt(image_array.size))
    prewhitened_image = (image_array - mean) / std_adj
    
    # 3. Add Batch Dimension (Model expects [1, 160, 160, 3])
    input_tensor = np.expand_dims(prewhitened_image, axis=0)
    
    return input_tensor

# --- Main Script: Load Graph and Run Inference ---
def load_model_and_get_embedding(model_path, image_path):
    """
    Loads the frozen graph and runs a forward pass to get the embedding.
    """
    tf.disable_eager_execution()
    
    print(f"Loading model from: {model_path}")
    print(f"Processing image: {image_path}\n")

    try:
        # Load the graph definition from the .pb file
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Import the graph into the session
        tf.import_graph_def(graph_def, name='')

        # Prepare the input image
        input_image = pre_process_image(image_path)

        # Start a TensorFlow Session
        with tf.Session() as sess:
            # Get the necessary tensors by name
            images_placeholder = tf.get_default_graph().get_tensor_by_name(INPUT_TENSOR_NAME)
            embeddings = tf.get_default_graph().get_tensor_by_name(OUTPUT_TENSOR_NAME)
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name(PHASE_TRAIN_TENSOR_NAME)
            
            # Run the forward pass (Inference)
            feed_dict = {
                images_placeholder: input_image,
                phase_train_placeholder: False
            }
            
            embedding_vector = sess.run(embeddings, feed_dict=feed_dict)
            
            return embedding_vector[0] # Return the 512-D vector

    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Please check the MODEL_PATH variable.")
        return None
    except Exception as e:
        print(f"A graph loading or execution error occurred: {e}")
        print("\n*TIPS: This often means the input tensor names or shapes are slightly wrong, or the wrong TensorFlow version is installed.*")
        return None

if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"🚨 ERROR: Cannot find sample image at '{IMAGE_PATH}'.")
        print("Please ensure your file is named 'sample.png' and is in the same folder.")
    else:
        # 1. Run the test
        final_embedding = load_model_and_get_embedding(MODEL_PATH, IMAGE_PATH)
        
        # 2. Print the results
        if final_embedding is not None:
            print("\n-----------------------------------------------------")
            print("✅ TEST SUCCESSFUL! FaceNet Embedding Generated.")
            print("-----------------------------------------------------")
            print(f"1. Model Type: Frozen TensorFlow Graph (.pb)")
            print(f"2. Output Dimension: {final_embedding.shape[0]}D (Your 512-D Face Template)")
            print(f"3. First 5 Embedding Values:\n   {final_embedding[:5]}")
            
            # Print the full 512-D embedding vector
            print("\n4. FULL 512-D EMBEDDING VECTOR (THE FACE MAP):")
            # Set print options to ensure the entire array is displayed without truncation
            np.set_printoptions(threshold=np.inf, linewidth=np.inf) 
            print(final_embedding)
            np.set_printoptions(threshold=1000, linewidth=75) 

            print("\n🔥 NEXT STEP: Now, this float array must be converted to a small, fast TFLite file.")
