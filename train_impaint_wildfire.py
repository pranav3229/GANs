import cv2
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Path to the directory containing original and corrupted images
original_dir = '/home/postman/dl_project_tanna/train+val/train/Wildfire/'
corrupted_dir = '/home/postman/dl_project_tanna/train+val/train/Wildfire_paired/'

# Path to save the processed images
output_dir = '/home/postman/dl_project_tanna/impainted_dataset/train/wildfire/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all files in the original directory
file_list = os.listdir(original_dir)

# Initialize lists to store SSIM and PSNR values
ssim_values = []
psnr_values = []

for file in file_list:
    original_path = os.path.join(original_dir, file)
    corrupted_path = os.path.join(corrupted_dir, file)

    if os.path.isfile(original_path) and os.path.isfile(corrupted_path):
        original_image = cv2.imread(original_path)
        corrupted_image = cv2.imread(corrupted_path)
        # print(original_image.shape)
        # print(corrupted_image.shape)
        if original_image is not None and corrupted_image is not None:
            # Resize images to 256x256
            original_image = cv2.resize(original_image, (256, 256))
            corrupted_image = cv2.resize(corrupted_image, (256, 256))

            # Convert images to the required format (BGR to RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            corrupted_image = cv2.cvtColor(corrupted_image, cv2.COLOR_BGR2RGB)

            # Mask the black lines in the corrupted image
            mask = cv2.inRange(corrupted_image, (0, 0, 0), (10, 10, 10))  # Define the threshold for black
            corrupted_image[mask != 0] = original_image[mask != 0]  # Replace the black areas with original image data

            # Use DeepFill v1 model to inpaint the corrupted areas #this model internally uses GANs
            result = cv2.inpaint(corrupted_image, (mask/255).astype('uint8'), inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            # Calculate SSIM
            ssim_score = ssim(cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY),
                              cv2.cvtColor(result, cv2.COLOR_RGB2GRAY))
            ssim_values.append(ssim_score)

            # Calculate PSNR
            psnr_score = psnr(original_image, result)
            psnr_values.append(psnr_score)

            # Save the resulting image
            output_path = os.path.join(output_dir, file)
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

# Calculate average SSIM and PSNR values
avg_ssim = sum(ssim_values) / len(ssim_values)
avg_psnr = sum(psnr_values) / len(psnr_values)

print("Average SSIM:", avg_ssim)
print("Average PSNR:", avg_psnr)
print("Image processing complete. Check the output directory for the processed images.")



