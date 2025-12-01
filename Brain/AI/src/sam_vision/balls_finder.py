from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import argparse
device = "cuda" if torch.cuda.is_available() else "cpu"
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, type=str, help="Path to input image")
    args = parser.parse_args()
    return args.path

def load_SAM3():
    SAM_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    SAM_processor = Sam3Processor.from_pretrained("facebook/sam3")
    return SAM_model, SAM_processor

MODEL,PROCESSOR=load_SAM3()


def load_image(path):
    image = Image.open(path).convert("RGB")
    return image

def detect_balls(image_path,model=None, processor=None ):
    if model == None:
        model, processor = MODEL,PROCESSOR
    image = load_image(image_path)
    # for draughts
    text_prompts = ["draughts"]
    inputs = processor(images=image, text=text_prompts, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    masks = results["masks"]
    print(f"Found {len(results['masks'])} objects")
    # If no masks → skip drawing
    if len(masks) == 0:
        print("No 'draughts' detected.")
        masks = []
        text_prompts = ["ball OR coloured ball"]
        inputs = processor(images=image, text=text_prompts, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Post-process results
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.4,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        masks = results["masks"]
        print(f"Found {len(results['masks'])} balls")

    else:
        print("'draughts' detected.")
        # ---- Find largest mask by area ----
        areas = masks.view(masks.shape[0], -1).sum(dim=1)   # tensor areas
        largest_idx = areas.argmax().item()
        masks = [masks[largest_idx]]   # <-- keep only the largest mask
        
        largest_mask = masks[largest_idx]
        largest_mask_np = largest_mask.cpu().numpy().astype(bool)
        ys, xs = np.where(largest_mask_np)

        if len(xs) > 0:
            mask_width = xs.max() - xs.min() + 1
            mask_height = ys.max() - ys.min() + 1
            print(f"Largest mask size → width: {mask_width}, height: {mask_height}")   
            if (mask_width - mask_height) > 20:
                print ("no draughts")
                masks = []
                text_prompts = ["ball OR coloured ball"]
                inputs = processor(images=image, text=text_prompts, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                # Post-process results
                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.4,
                    mask_threshold=0.5,
                    target_sizes=inputs.get("original_sizes").tolist()
                )[0]
                masks = results["masks"]
                print(f"Found {len(results['masks'])} balls")
            else:
                print("yes, draughts")
                text_prompts = ["draughts piece"]
                inputs = processor(images=image, text=text_prompts, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                # Post-process results
                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=inputs.get("original_sizes").tolist()
                )[0]
                masks = results["masks"]
                print(f"Found {len(results['masks'])} draught pieces")
                
                
                
    img_np = np.array(image)
    overlay = img_np.copy()
    colors = np.random.randint(0, 255, (len(["masks"]), 3))
    alpha = 0.5
    centers_radii = []
    for mask in masks:
        # Convert tensor → numpy
        mask = mask.cpu().numpy()
        m = mask.astype(bool)

        color = np.array([0, 200, 0])  # or any color you like
        overlay[m] = (overlay[m] * (1 - alpha) + color * alpha).astype(np.uint8)

        ys, xs = np.where(m)
        if len(xs) == 0:
            continue

        cx = int(xs.mean())
        cy = int(ys.mean())
        area = len(xs)                    # number of pixels in the mask
        radius = np.sqrt(area / np.pi)    # equivalent circle radius
        centers_radii.append((cx, cy, radius))
        cv2.circle(overlay, (cx, cy), int(radius), (255, 255, 255), 4)  # white outer
        cv2.circle(overlay, (cx, cy), 10, (0, 0, 0), 2)        # black inner
        text = f"({cx}, {cy})"

        cv2.putText(
            overlay,
            text,
            (cx, cy),               # position slightly offset
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),             # white text
            2,
            cv2.LINE_AA
        )
        


        # write radius instead of (cx, cy)
        # text = f"r={radius:.1f}px"

        # cv2.putText(
        #     overlay,
        #     text,
        #     (cx-10, cy-10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1.0,
        #     (255, 255, 255),
        #     2,
        #     cv2.LINE_AA,
        # )
    '''
    base, _ = os.path.splitext(save_path)
    txt_path = base + ".txt"

    with open(txt_path, "w") as f:
        for cx, cy, r in centers_radii:
            f.write(f"{cx} {cy} {r:.2f}\n")
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()
    '''
    return centers_radii,overlay


if __name__ == "__main__":
    image_path = arg_parser()
    model, processor = load_SAM3()
    image = load_image(image_path)
    detect_balls(model, processor, image, image_path)