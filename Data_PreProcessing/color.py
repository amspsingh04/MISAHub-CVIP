import cv2
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor


# Define the directory path
input_1= '../Dataset/training'
input_2='../Dataset/validation'
output_1 = '../Dataset_new/training'
output_2 = '../Dataset_new/validation'

if not (os.path.exists(output_1) and os.path.exists(output_2)):
    os.makedirs(output_1)
    os.makedirs(output_2)

def sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    return sepia_image

def sharpenn(image):
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

def adjust_brightness(image, brightness=30):
    bright_image = cv2.convertScaleAbs(image, beta=brightness)
    return bright_image

def adjust_contrast(image, contrast=1.5):
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    return adjusted_image

def apply_filter(image):
    #image = sepia(image)
    image = sharpenn(image)
    image = adjust_brightness(image, brightness=50)    
    image = adjust_contrast(image, contrast=1.3)
    return image

def process_image(filepath):
    image = cv2.imread(filepath)
    if image is None:
        print(f"Error loading image {filepath}, skipping...")
        return
    filtered_image = apply_filter(image)
    relative_path = os.path.relpath(filepath, input_directory)
    output_path = os.path.join(output_directory, relative_path)
    output_subfolder = os.path.dirname(output_path)
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    success = cv2.imwrite(output_path, filtered_image)
    
    if success:
        print(f"Processed and saved: {output_path}")
    else:
        print(f"Failed to save: {output_path}")

def process_images_in_parallel():
    image_files = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_files.append((os.path.join(root, file), root))  # Append tuple of (file path, root)

    image_paths = [item[0] for item in image_files]
    with ProcessPoolExecutor() as executor:
        executor.map(process_image, image_paths)
    print(image_paths)
if __name__ == "__main__":
    process_images_in_parallel()
    print("Processing complete!")