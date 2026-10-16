# Line-by-line explanations

These are the AI's explanations of the two course starter programs and of the Grad-CAM algorithm, as the assignment asks. My observations from running the code are in `README.md`.

## base_classifier.py (course starter)

| Line | What it does |
|---|---|
| `import tensorflow as tf` | Loads TensorFlow, the library that runs the neural network. |
| `tf.get_logger().setLevel('ERROR')` | Hides TensorFlow's informational messages so only errors print. |
| `from tensorflow.keras.applications import MobileNetV2` | Imports MobileNetV2, a small image classification network built for phones. |
| `from ...mobilenet_v2 import preprocess_input, decode_predictions` | Imports the two helpers made for this network: one prepares pixels the way it was trained, one turns its output into readable labels. |
| `from tensorflow.keras.preprocessing import image` | Imports helpers for loading an image file into an array. |
| `import numpy as np` | Imports NumPy for array operations. |
| `model = MobileNetV2(weights="imagenet")` | Builds the network and downloads weights already trained on ImageNet's 1,000 classes. Runs once at startup. |
| `def classify_image(image_path):` | Defines the function that classifies one file. |
| `try:` | Starts a block whose errors are caught and printed instead of crashing the program. |
| `img = image.load_img(image_path, target_size=(224, 224))` | Opens the file and resizes it to 224 x 224, the input size the network expects. It stretches the image to a square rather than cropping it. |
| `img_array = image.img_to_array(img)` | Converts the image into a 224 x 224 x 3 array of red, green, and blue values. |
| `img_array = preprocess_input(img_array)` | Rescales the pixel values from 0 to 255 into the range -1 to 1 that MobileNetV2 was trained on. |
| `img_array = np.expand_dims(img_array, axis=0)` | Adds a batch dimension, making the shape 1 x 224 x 224 x 3, because the model always takes a batch of images. |
| `predictions = model.predict(img_array)` | Runs the image through the network and returns 1,000 scores, one per ImageNet class. |
| `decoded_predictions = decode_predictions(predictions, top=3)[0]` | Matches the scores to class names and keeps the three highest, for the first (only) image in the batch. |
| `print(...)` and the `for` loop | Prints the rank, class name, and score of each of the three predictions. |
| `except Exception as e:` | Prints the error if anything above failed, for example a missing file. |
| `if __name__ == "__main__":` | Runs the following only when the file is run directly, not when imported. |
| the `while True:` loop | Keeps asking for a filename and classifying it until the user types `exit`. |

## basic_filter.py (course starter)

| Line | What it does |
|---|---|
| `from PIL import Image, ImageFilter` | Imports Pillow for opening images and applying filters. |
| `import matplotlib.pyplot as plt` | Imports matplotlib, here used to save the result. |
| `import os` | Imports file path helpers. |
| `def apply_blur_filter(image_path, output_path="blurred_image.png"):` | Defines the function that blurs one image and saves it. |
| `img = Image.open(image_path)` | Opens the image file. |
| `img_resized = img.resize((128, 128))` | Shrinks the image to 128 x 128, ignoring its original proportions. |
| `img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))` | Blurs it: each pixel becomes a weighted average of its neighbors, with closer pixels weighted more. Radius 2 sets how far the averaging reaches. |
| `plt.imshow(img_blurred)` | Draws the blurred image on a matplotlib figure. |
| `plt.axis('off')` | Hides the axes and tick marks. |
| `plt.savefig(output_path, bbox_inches='tight', pad_inches=0)` | Saves the figure to a file, trimmed to the image. Because this saves the drawn figure rather than the image itself, the file comes out at the figure's resolution, not 128 x 128. |
| `plt.close()` | Closes the figure so figures do not pile up in memory. |
| `print(...)` | Reports where the file was saved. |
| `except Exception as e:` | Prints any error. |
| the main loop | Asks for filenames, skips ones that do not exist, builds an output name ending in `_blurred`, and calls the function, until the user types `exit`. |

## How Grad-CAM works

Grad-CAM answers one question about a prediction: which parts of the image raised the score for this class?

1. A convolutional network keeps a grid of where things are until near the end. In MobileNetV2 the last convolutional layer, `Conv_1`, outputs 1,280 feature maps, each 7 x 7. Each map responds to some pattern, and its 7 x 7 grid says where in the image that pattern showed up.
2. Run the image forward and keep two things: those feature maps and the score for the class you care about.
3. Compute the gradient of that class score with respect to every value in every feature map. A large positive gradient means "more of this pattern here would raise the score."
4. Average each map's gradients over its 7 x 7 grid. That gives one importance weight per map: how much this pattern, wherever it appears, matters for this class.
5. Multiply each map by its weight and add all 1,280 maps together. The result is a single 7 x 7 map.
6. Keep only the positive values (ReLU), because the question is what supported the class, not what argued against it, and scale so the strongest cell is 1. Because of that scaling, every map has a hottest cell, even for a class the model barely considered. The map says where the evidence was, not how much there was; the class score says how much.
7. Stretch the 7 x 7 map to the size of the original image and color it, red for strong, blue for none, over the photo.

Because it uses the class score, you can run it for any class, not just the top prediction. That is how the README compares the Eskimo dog, Siberian husky, and malamute maps for the same photo. Grad-CAM shows where the evidence came from. It does not show why the network learned to treat that region as evidence.
