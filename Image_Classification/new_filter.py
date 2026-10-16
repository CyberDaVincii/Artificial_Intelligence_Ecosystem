"""Hologram filter.

Part 2 of the AIE image classification project. Starts from the course's
basic blur filter and turns it into a projected-hologram look: a cyan cast,
horizontal scanlines, a slight red/blue channel offset, a soft glow, and a
little noise, like a figure projected in a dark room.

Usage: run the file, then type an image filename at the prompt. The result
is saved as <name>_hologram.png next to the input. The knobs at the top are
the parameters to adjust.
"""
import os

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Tuning knobs
TINT = (90, 220, 255)      # cyan the image is pushed toward (R, G, B)
TINT_STRENGTH = 0.45       # 0 = original colors, 1 = pure tint
SCANLINE_PERIOD = 4        # every Nth row is darkened
SCANLINE_DARKNESS = 0.55   # 0 = black lines, 1 = invisible
CHANNEL_SHIFT = 3          # pixels of red/blue misalignment
GLOW_RADIUS = 6            # blur radius of the glow layer
GLOW_STRENGTH = 0.35       # how much glow is added back
NOISE = 12                 # amplitude of grain, 0 to disable


def apply_hologram_filter(image_path, output_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = ImageEnhance.Contrast(img).enhance(1.25)

        # 1. Cyan tint: blend each pixel toward the tint color.
        arr = np.asarray(img).astype(np.float32)
        tint = np.array(TINT, dtype=np.float32)
        arr = arr * (1 - TINT_STRENGTH) + tint * (arr.mean(axis=2, keepdims=True) / 255.0) * TINT_STRENGTH

        # 2. Channel offset: shift red left and blue right, like a projector
        #    that is slightly out of register.
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        r = np.roll(r, -CHANNEL_SHIFT, axis=1)
        b = np.roll(b, CHANNEL_SHIFT, axis=1)
        arr = np.stack([r, g, b], axis=2)

        # 3. Scanlines: darken every Nth row.
        arr[::SCANLINE_PERIOD, :, :] *= SCANLINE_DARKNESS

        # 4. Grain: small random noise so the projection looks alive.
        if NOISE:
            rng = np.random.default_rng(7)
            arr += rng.normal(0, NOISE, arr.shape)

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        holo = Image.fromarray(arr)

        # 5. Glow: blur a copy and screen it back over the image.
        glow = holo.filter(ImageFilter.GaussianBlur(radius=GLOW_RADIUS))
        holo = Image.blend(holo, Image.fromarray(
            np.clip(np.asarray(holo).astype(np.float32) + np.asarray(glow).astype(np.float32) * GLOW_STRENGTH, 0, 255).astype(np.uint8)
        ), 0.8)

        holo.save(output_path)
        print(f"Processed image saved as '{output_path}'.")
    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    print("Hologram Filter (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        if image_path.lower() == "exit":
            print("Goodbye!")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue
        base, _ = os.path.splitext(image_path)
        apply_hologram_filter(image_path, f"{base}_hologram.png")
