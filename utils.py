import cv2
import numpy as np
from scipy.spatial import distance as dist

def order_points(pts):
    """
    Orders the four points of a quadrilateral in top-left, top-right,
    bottom-right, bottom-left order. This is crucial for perspective transform.
    """
    # Sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # Grab the left-most and right-most points from the sorted x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # Now, sort the left-most coordinates according to their y-coordinates
    # so we can grab the top-left and bottom-left points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # Now that we have the top-left coordinate, use it as an anchor to
    # calculate the Euclidean distance between the top-left and right-most
    # points; the point with the largest distance will be the bottom-right
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # Return the coordinates in top-left, top-right, bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def deskew_and_clean_plate(plate_img):
    """
    This is the ADVANCED core function for improving OCR accuracy.
    It takes a cropped license plate image and performs:
    1. Perspective transformation (de-skewing) to make the plate look flat.
    2. Grayscaling.
    3. Contrast Enhancement using CLAHE.
    4. Thresholding to create a clean binary image.
    5. Morphological operations to clean up noise.
    6. Resizing for OCR consistency.
    """
    # --- 1. De-skew the image ---
    # First, we need to find the plate within the (potentially loose) crop
    # and warp it to be a flat, rectangular image.
    
    # Grayscale for contour detection
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter for noise reduction while keeping edges sharp
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive thresholding to find contours
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, we can't de-skew. Fall back to simpler processing.
    if not contours:
        # If de-skewing fails, use the original crop for the next steps
        warped = plate_img
    else:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = box.astype(int)

        ordered_box = order_points(box)
        (tl, tr, br, bl) = ordered_box

        # Calculate the width and height of the new de-skewed image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Prevent errors from very small or malformed contours
        if maxWidth <= 0 or maxHeight <= 0:
            warped = plate_img # Fallback if contour is invalid
        else:
            dst = np.array([
                [0, 0], [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
            
            M = cv2.getPerspectiveTransform(ordered_box, dst)
            warped = cv2.warpPerspective(plate_img, M, (maxWidth, maxHeight))

    # --- Now apply the rest of the pre-processing steps to the `warped` image ---
    
    # --- 2. Convert to Grayscale ---
    processed_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # --- 3. Contrast Enhancement (CLAHE) ---
    # This is excellent for handling shadows and uneven lighting.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_contrast = clahe.apply(processed_gray)

    # --- 4. Thresholding (Otsu's Binarization) ---
    # This converts the image to pure black and white, isolating the text.
    # Otsu's method automatically finds the best threshold value.
    _, thresholded = cv2.threshold(enhanced_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- 5. Morphological Operations (Optional but helpful) ---
    # Use a small kernel to close gaps in characters without merging them.
    kernel = np.ones((2, 1), np.uint8)
    morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    # --- 6. Resizing for Better OCR ---
    # Resize to a consistent, larger height for the OCR engine.
    # Using a high-quality interpolation method is important.
    final_height = 60
    aspect_ratio = morphed.shape[1] / morphed.shape[0]
    final_width = int(final_height * aspect_ratio)
    final_image = cv2.resize(morphed, (final_width, final_height), interpolation=cv2.INTER_CUBIC)

    # Invert image if text is white on black background (common after thresholding)
    # EasyOCR often works better with black text on a white background.
    mean_pixel_value = cv2.mean(final_image)[0]
    if mean_pixel_value < 127: # If the image is mostly black
        final_image = cv2.bitwise_not(final_image) # Invert it

    # Return the clean image, ready for OCR
    return final_image

def clean_plate_text(raw_text):
    """
    Cleans the raw OCR output.
    - Converts to uppercase
    - Removes all non-alphanumeric characters
    """
    if not raw_text:
        return ""
    return ''.join(char for char in raw_text if char.isalnum()).upper()