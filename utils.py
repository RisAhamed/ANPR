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
    This is the core function for improving OCR accuracy.
    It takes a cropped license plate image and performs:
    1. Grayscaling and contrast enhancement.
    2. Contour detection to find the actual plate within the crop.
    3. Perspective transformation (de-skewing) to make the plate look flat.
    4. Resizing for consistency.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Apply a bilateral filter to reduce noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Use adaptive thresholding to create a binary image
    # This helps in finding contours of the plate characters and border
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour which is likely the plate itself
    if not contours:
        # If no contours, fall back to a simple resize and grayscale
        return cv2.cvtColor(cv2.resize(gray, (200, 60), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_GRAY2BGR)

    # Get the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the minimum area rectangle for the largest contour
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    
    # --- THIS IS THE FIX ---
    # OLD LINE: box = np.int0(box)
    # NEW LINE:
    box = box.astype(int)
    # --- END OF FIX ---

    # Order the points of the rectangle
    ordered_box = order_points(box)
    (tl, tr, br, bl) = ordered_box

    # Calculate the width and height of the new de-skewed image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Define the destination points for the de-skewed image
    # We create a new "flat" rectangle of the calculated max width and height
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Get the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(ordered_box, dst)
    warped = cv2.warpPerspective(plate_img, M, (maxWidth, maxHeight))

    # Final resize for OCR consistency
    warped = cv2.resize(warped, (200, 60), interpolation=cv2.INTER_LANCZOS4)
    
    # Return the processed plate, ready for OCR
    return warped

def clean_plate_text(raw_text):
    """
    Cleans the raw OCR output.
    - Converts to uppercase
    - Removes all non-alphanumeric characters
    """
    if not raw_text:
        return ""
    return ''.join(char for char in raw_text if char.isalnum()).upper()