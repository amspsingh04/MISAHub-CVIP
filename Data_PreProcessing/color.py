import cv2
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing  

# Define the directory path
input_dirs = ['../Dataset/training', '../Dataset/validation']
output_dirs = ['../Dataset_new/training', '../Dataset_new/validation']


for output_dir in output_dirs:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

#def sepia(image):
#   sepia_filter = np.array([[0.272, 0.534, 0.131],
#                           [0.349, 0.686, 0.168],
#                          [0.393, 0.769, 0.189]])
# sepia_image = cv2.transform(image, sepia_filter)
# sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
# return sepia_image
#We use the above one to sepia our images
#I do not think we need to sepia our images, but we can use the above template to provide a color filter over the images (we can use it to grayscale images)

def sharpenn(image):
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image
#check sharpened image example here: '..\examples\sharpen.jpg' and decide whether to use

def adjust_brightness(image, brightness=30):
    bright_image = cv2.convertScaleAbs(image, beta=brightness)
    return bright_image
#check brightness example here: '..\examples\brightness.jpg' and toggle around to decide on parameters

def adjust_contrast(image, contrast=1.5):
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    return adjusted_image
#check contrast toggling here: ''..\examples\contrast.jpg'' and decide on parameters

def gaussian_blur_correction(image):
    kernel_size = (5, 5)  
    sigma = 10e-5
    kernel = cv2.getGaussianKernel(ksize=1, sigma=sigma)
    kernel = np.outer(kernel, kernel.T)
    corrected_image = cv2.filter2D(image, -1, kernel)
    return corrected_image
#check GBC toggling here: ''..\examples\gaussian_blur.jpg'' and decide on parameters

def apply_filter(image):
    #image = sepia(image)
    image = sharpenn(image)
    #image = adjust_brightness(image, brightness=50)    
    image = adjust_contrast(image, contrast=1.4)
    image = gaussian_blur_correction(image)
    return image

def process_image(filepath, output_directory):
    image = cv2.imread(filepath)
    print(filepath)
    if image is None:
        print(f"Error loading image {filepath}, skipping...")
        return
    
    filtered_image = apply_filter(image)
    
    relative_path = os.path.relpath(filepath, os.path.dirname(output_directory))
    output_path = os.path.join(output_directory, relative_path)
    output_subfolder = os.path.dirname(output_path)
    
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    success = cv2.imwrite(output_path, filtered_image)
    
    if success:
        print(f"Processed and saved: {output_path}")
    else:
        print(f"Failed to save: {output_path}")

def process_images_in_parallel(input_directory, output_directory):
    image_files = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, file))  
    num_cores = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        executor.map(process_image, image_files, [output_directory] * len(image_files))

if __name__ == "__main__":
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        print(input_dir, output_dir)
        process_images_in_parallel(input_dir, output_dir)
    print("Processing complete!")