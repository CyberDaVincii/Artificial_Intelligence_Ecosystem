# Image Classification and Processing

Angela Reinhold · The Artificial Intelligence Ecosystem, Week 3 · project 3.3

Forked from the course repository. This project lives at https://github.com/CyberDaVincii/Artificial_Intelligence_Ecosystem/tree/main/Image_Classification. My files are `base_classifier.py` (the classifier with Grad-CAM added), `new_filter.py` (my filter), and the images in `images/`. `basic_filter.py` and `requirements.txt` are the course starters, unchanged.

## How to run

Inside WSL, in a Python 3.13 virtual environment (see the reflection for why 3.13 and not the system Python):

```
pip install -r requirements.txt
python base_classifier.py     # type images/apollo.jpg at the prompt
python basic_filter.py        # the course blur
python new_filter.py          # the hologram filter
```

## Part 1: the classifier and Grad-CAM

**The image.** My dog Apollo, a black and white Siberian husky, photographed at home (`images/apollo.jpg`, 1536 x 2048). He has one brown eye and one blue eye, which mattered more than I expected.

**Top-3 predictions** from the unchanged `base_classifier.py`:

| Rank | ImageNet class | Confidence |
|---|---|---|
| 1 | Eskimo_dog | 0.46 |
| 2 | Siberian_husky | 0.32 |
| 3 | malamute | 0.03 |

Apollo is a Siberian husky, so the correct breed came second. ImageNet keeps Eskimo dog (class 248) and Siberian husky (class 250) as separate classes for dogs that look nearly the same, so this split is the kind of mistake I expected to see if the model was looking at the right things. The heatmaps below say it was.

**What the code does.** The line-by-line explanation is in `EXPLANATIONS.md`. The detail that mattered most for Apollo is that `load_img(..., target_size=(224, 224))` squeezes the photo to a square instead of cropping it. His photo is 3:4, so the network saw him slightly squashed. `preprocess_input` scales the pixels to the range MobileNetV2 was trained on, `expand_dims` adds the batch dimension the model expects, and `decode_predictions` turns the 1,000 output scores into labels and keeps the top three.

**Grad-CAM.** Added to `base_classifier.py`. It reads the last convolutional layer, `Conv_1`, which still keeps a 7 x 7 map of where things are in the image (1,280 feature maps). It takes the gradient of the predicted class score with respect to those maps, averages each map's gradient into one importance weight, combines the maps with those weights, keeps only positive evidence, and scales the result to 0 to 1. The 7 x 7 result is stretched over the photo as a color overlay: red is where the evidence for the class was strongest, blue is where there was none. The program saves it as `<name>_gradcam.png`.

**Heatmap analysis.** I looked at the overlays and also printed the 7 x 7 values so the analysis would not rest on colors alone.

- **Eskimo_dog, the top prediction** (`images/apollo_gradcam.png`). The hottest cell is the left side of his muzzle, with the right cheek close behind at 0.94 of the peak. The band across his muzzle, mouth, and dark neck ruff averages 0.70. The upper third of the frame, his ears and eyes and the table behind them, averages 0.39. His legs and the floor around them average 0.14, and the outer columns, where the wall and storage bins are, 0.02. The model was looking at the dog, and mostly at his lower face.
- **Siberian_husky, the correct breed** (`images/apollo_gradcam_class250.png`). I ran Grad-CAM a second time for this class. The hottest cell moves up to the center of his face, between the eyes, and the upper third (0.51) now outweighs the muzzle and neck (0.42). His legs drop to zero.
- **The difference.** The two maps are similar overall (correlation 0.77) but not the same. The Eskimo dog evidence sits on the muzzle and ruff. The Siberian husky evidence sits on the facial mask: the cell between his eyes scores 1.00, the cell with his brown eye 0.95, and the cell with his blue eye 0.91. I expected his blue eye to stand out, since blue eyes are so tied to huskies, but the two eye cells score almost the same. That points at the mask pattern around the eyes mattering more than eye color, though each cell covers an eye and the fur around it, so a map this coarse cannot isolate eye color on its own. A heatmap shows where the evidence came from, not why the model learned it.
- **Malamute, at 0.03** (`images/apollo_gradcam_class249.png`). This map puts almost no heat on Apollo. Its hottest cell is the floor in the bottom right corner, and the image edges carry more heat (0.23) than his face does. One caution: the code scales every class map so its own strongest cell is 1.0, so even a class the model barely considered gets a bright spot somewhere. The map shows that what little malamute evidence there was sat on the floor, not on the dog. It cannot say how much evidence that was; the 0.03 score says that.

What I learned about the classifier's behavior: it does not decide "husky" from the whole dog. It separates two nearly identical classes by which part of the face it reads, and its confidence of 0.46 is a split decision, not certainty. Without the second and third heatmaps, I would only have known it was wrong. With them, I know how it was wrong.

## Part 2: filters

**The course blur** (`basic_filter.py`, output `images/apollo_blurred.jpg`). It opens the image, resizes it to 128 x 128, applies a Gaussian blur with radius 2, then saves it by drawing it with matplotlib (`imshow` and `savefig`) instead of saving it directly. Two things I noticed from running it on Apollo. The resize ignores the aspect ratio, so he comes out squashed into a square, the same way the classifier sees him. And because `savefig` re-renders the 128-pixel image at the figure's resolution, the saved file is 369 x 369, blurred and also low resolution. Saving with Pillow directly would keep it at exactly 128 x 128.

**My filter: hologram** (`new_filter.py`, output `images/apollo_hologram.png`). I picked a projected-hologram look because holograms are something I want to build, and I wanted to see Apollo as one. The filter runs in five steps, each with its knob at the top of the file:

1. Raise contrast by 25 percent.
2. Blend each pixel 45 percent toward cyan, scaled by the pixel's brightness, so light areas glow and dark areas stay dark.
3. Shift the red channel 3 pixels left and the blue channel 3 pixels right, like a projector slightly out of alignment.
4. Darken every fourth row to 55 percent brightness for scanlines.
5. Add light grain (fixed random seed, so the output repeats), then lay a blurred copy (radius 6) back over the image at about 28 percent strength for a glow.

**Its effect on Apollo.** His black and white coat turns cyan and silver, and the white of his face and chest blooms. The scanlines show on the white fur and the light floor and nearly vanish on his black fur: on the darkened rows, bright areas drop about 75 levels of brightness and dark fur only about 7, because darkening something already dark changes little. Around his eyes and the white patches above them, the channel shift leaves thin red and blue fringes, which is what makes him look projected rather than just tinted. His blue eye turns into a bright ring.

**Running it on other images.** I also ran it on a 224 x 224 drawing and a 1000 x 1000 painting. On Apollo's large photo the scanlines are fine and read as texture. On the small drawing, one darkened row out of every four is a much bigger share of the picture, and the lines dominate. The scanline spacing is fixed in pixels and does not scale with image size. That is the next change I would make: set the spacing as a fraction of the image height.

## Working with the AI

I worked with Claude through the whole project, from setup to the final run. Four moments stand out.

1. **The setup did not work as written.** The Ubuntu in my WSL is version 26.04, which ships Python 3.14, and TensorFlow's current release only publishes Linux packages up to Python 3.13. Following the instructions exactly would have failed at `pip install -r requirements.txt`. The AI caught this by checking which Python versions TensorFlow publishes for, installed Python 3.13 inside WSL with the `uv` tool, and built the virtual environment from that.
2. **A bug that looked right.** The first Grad-CAM version ran on two square test images and produced believable colored overlays, and the AI described them to me as results. When I gave it Apollo's photo, which is not square, it crashed with a shape error. That error showed the overlay had been wrong the whole time: it took a single column of the heatmap and smeared it across the image as vertical stripes. On square images the shapes happened to line up, so nothing failed and the stripes passed as a heatmap. The fix stretches the 7 x 7 heatmap to the photo's exact width and height and checks the shape before drawing. The lesson I took from it is the same one the Week 3 lecture opened with: a result that looks plausible is not a result that has been checked. We now print the numbers behind every heatmap and compare them with the picture.
3. **Making weaker regions readable.** Raising the heat values to the power 0.6 before coloring them lifts the middle range so secondary regions stay visible, without changing which region is hottest.
4. **Choosing what to publish.** One of my test images was another artist's watermarked fan art. The AI flagged that it should not go in a public repository, so this repository contains only my own photo of Apollo and the images made from it.

What worked best was treating the AI's explanations as something to check rather than something to accept. The line-by-line explanations are in `EXPLANATIONS.md`, and the parts I learned the most from were the ones we verified by running the code and measuring the output: the squashed input, the 369-pixel blur, the scanlines that vanish on black fur, and the striped heatmap.
