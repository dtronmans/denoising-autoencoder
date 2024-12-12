import os

import cv2
import numpy as np


def invert_remove(image_path, show_contours=False):
    # Step 1: Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found or unable to open")
        return None

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur

    # Step 3: Thresholding - use Otsu's method
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 4: Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found.")
        return None

    # Sort contours by area and remove the largest if it's the size of the image
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if cv2.contourArea(contours[0]) > 0.9 * image.size / image.shape[
        2]:  # Check if contour is nearly as big as the image
        contours.pop(0)  # Remove the largest contour

    if not contours:
        print("No valid contours found after removing the largest.")
        return None

    # Step 5: Draw the largest valid contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contours[0]], -1, (255), thickness=cv2.FILLED)

    # Optional: Draw all contours on the image for visualization
    if show_contours:
        contour_image = image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
        cv2.imshow('All Contours', contour_image)
        cv2.waitKey(0)

    # Step 6: Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)

    return result


if __name__ == "__main__":
    # result_image = invert_remove('LUMC_RDG_clean_inferred/benign/LUM00001_1.png', show_contours=True)
    for image in os.listdir("all_datasets/LUMC_RDG_clean_inferred/benign"):
        image_path = os.path.join("all_datasets/LUMC_RDG_clean_inferred", "benign", image)
        result_image = invert_remove(image_path, show_contours=True)
        if result_image is not None:
            cv2.imshow('Largest Object', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
