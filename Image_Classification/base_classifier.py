"""MobileNetV2 image classifier with Grad-CAM.

Part 1 of the AIE image classification project. The classifier is the
course starter; the Grad-CAM section was added so the program can show which
regions of the image drove the top prediction.

Usage: run the file, then type an image filename at the prompt. For each
image it prints the top-3 ImageNet predictions and writes
<name>_gradcam.png next to the input.
"""
import os

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")           # write files; no display needed under WSL
import matplotlib.pyplot as plt

# Load MobileNetV2 with ImageNet weights once at startup (1000 classes).
model = MobileNetV2(weights="imagenet")

# The last convolutional layer in MobileNetV2. Grad-CAM reads its 7x7
# feature maps, because they are the last point where the network still
# knows "where" something is before pooling collapses the image to a vector.
LAST_CONV_LAYER = "Conv_1"


def load_and_preprocess(image_path):
    """Resize to 224x224, scale pixels the way MobileNetV2 expects, add batch axis."""
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


def make_gradcam_heatmap(img_array, class_index=None):
    """Return a 7x7 heatmap in [0, 1] for the predicted (or given) class.

    Grad-CAM in four steps:
      1. Run the image through the network, keeping both the last conv
         feature maps and the final class scores.
      2. Take the gradient of the chosen class score with respect to those
         feature maps: how much each map's activation pushes that score up.
      3. Average each map's gradient to get one importance weight per map,
         then combine the maps with those weights.
      4. Keep only positive evidence (ReLU) and scale to 0..1.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_outputs)          # d score / d feature maps
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))      # one weight per feature map
    conv_outputs = conv_outputs[0]                            # 7 x 7 x 1280
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]    # weighted sum of maps
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), class_index


def save_gradcam_overlay(image_path, heatmap, output_path, alpha=0.5, gamma=0.6):
    """Blend the upscaled heatmap over the original image and save it.

    gamma < 1 lifts the mid-range so weaker regions stay visible; the
    hottest region is still the hottest, only the contrast changes.
    """
    original = image.load_img(image_path)
    original = image.img_to_array(original).astype("uint8")
    height, width = original.shape[:2]
    heat = np.uint8(255 * np.power(heatmap, gamma))                     # 7 x 7, values 0..255
    heat = Image.fromarray(heat, mode="L").resize((width, height), Image.BILINEAR)
    heat = np.asarray(heat, dtype=np.float32)                           # height x width
    assert heat.shape == (height, width), heat.shape                    # one heat value per pixel
    colored = plt.get_cmap("jet")(heat / 255.0)[..., :3] * 255           # red = important
    overlay = (colored * alpha + original * (1 - alpha)).astype("uint8")
    image.array_to_img(overlay).save(output_path)


def classify_image(image_path):
    try:
        img_array = load_and_preprocess(image_path)

        predictions = model.predict(img_array, verbose=0)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        print("\nTop-3 Predictions for", image_path)
        for i, (_, label, score) in enumerate(decoded_predictions):
            print(f"  {i + 1}: {label} ({score:.2f})")

        # Grad-CAM for the top prediction.
        heatmap, class_index = make_gradcam_heatmap(img_array)
        base, _ = os.path.splitext(image_path)
        output_path = f"{base}_gradcam.png"
        save_gradcam_overlay(image_path, heatmap, output_path)
        print(f"  Grad-CAM for '{decoded_predictions[0][1]}' saved as '{output_path}'")
    except Exception as e:
        print(f"Error processing '{image_path}': {e}")


if __name__ == "__main__":
    print("Image Classifier with Grad-CAM (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename: ").strip()
        if image_path.lower() == "exit":
            print("Goodbye!")
            break
        classify_image(image_path)
