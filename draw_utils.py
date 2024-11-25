import os

import cv2
import numpy as np
from enum import Enum
import random


# draw the bounding boxes, the arrows and the noise you see in non-clean images

# Next steps:
# 1) Draw arrows from one end of the lesion to the other, like with annotations
# 2) Add text

# to draw:

# 1) RdGG_00007_0002: white cross-hairs on each side of the line, white dots form the line
# 2) RdGG_00008_0000: yellow cross-hairs, yellow dotted lines
# 3) RdGG_00022_0000: light blue dots and cross-hairs
# 4) RdGG_00045_0006: green bounding circle/box with artefacts in the middle

class Color(Enum):
    WHITE = (255, 255, 255)
    YELLOW = (0, 255, 255)
    LIGHT_BLUE = (255, 204, 51)


class DrawUtils:

    def __init__(self):
        pass

    @staticmethod
    def color_to_palette(color):
        if not isinstance(color, Color):
            raise ValueError("Provided color must be an instance of the Color enum.")
        return color.value

    @staticmethod
    def find_ovaries(image, display_contour=True):
        """
        Identify the central non-white region (assumed to be the ovaries) in the image.
        :param image: Input image
        :return: Coordinates of two random points within the region
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Threshold the image to isolate the non-white regions
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

        # Find contours of the non-white regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assume the largest contour near the center is the ovary
        if len(contours) == 0:
            raise ValueError("No non-white regions found in the image.")

        # Get the center point of the image
        height, width = gray.shape
        center = (width // 2, height // 2)

        # Sort contours by proximity to the center
        contours = sorted(contours, key=lambda cnt: cv2.pointPolygonTest(cnt, center, True))
        ovary_contour = contours[0]

        if display_contour:
            contour_image = image.copy()
            cv2.drawContours(contour_image, [ovary_contour], -1, (0, 255, 0), 2)
            cv2.imshow("Ovary Contour", contour_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Generate two random points within the ovary contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [ovary_contour], -1, 255, -1)
        points = np.column_stack(np.where(mask == 255))

        if len(points) < 2:
            raise ValueError("Insufficient points found in the ovary region.")

        point1 = tuple(points[random.randint(0, len(points) - 1)])
        point2 = tuple(points[random.randint(0, len(points) - 1)])
        return point1, point2

    @staticmethod
    def draw_arrows(image, color=Color.WHITE, num_dots=30):
        image_copy = image.copy()
        rgb_color = DrawUtils.color_to_palette(color)

        try:
            # Find two random points within the ovary
            end1, end2 = DrawUtils.find_ovaries(image)

            # Draw cross-hairs at both ends
            crosshair_size = 5
            cv2.line(image_copy, (end1[0] - crosshair_size, end1[1]), (end1[0] + crosshair_size, end1[1]), rgb_color, 3)
            cv2.line(image_copy, (end1[0], end1[1] - crosshair_size), (end1[0], end1[1] + crosshair_size), rgb_color, 3)

            cv2.line(image_copy, (end2[0] - crosshair_size, end2[1]), (end2[0] + crosshair_size, end2[1]), rgb_color, 3)
            cv2.line(image_copy, (end2[0], end2[1] - crosshair_size), (end2[0], end2[1] + crosshair_size), rgb_color, 3)

            # Draw dotted line between the two ends
            for i in range(num_dots + 1):
                t = i / num_dots
                x = int((1 - t) * end1[0] + t * end2[0])
                y = int((1 - t) * end1[1] + t * end2[1])
                cv2.circle(image_copy, (x, y), 1, rgb_color, -1)

        except ValueError as e:
            print(f"Error: {e}")

        return image_copy

    @staticmethod
    def draw_bounding_box(image):
        return


if __name__ == "__main__":
    path_clean = os.path.join("rdg_set", "clean")
    path_annotated = os.path.join("rdg_set", "annotated")

    for file in os.listdir(path_clean):
        image = cv2.imread(os.path.join(path_clean, file))
        drawn_image = DrawUtils.draw_arrows(image, Color.WHITE, num_dots=100)
        cv2.imwrite(os.path.join(path_annotated, file), drawn_image)

    # test_image = cv2.imread("car.jpg")
    # DrawUtils.draw_arrows(test_image, Color.LIGHT_BLUE)
    # cv2.imshow("Drawn arrows", test_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
