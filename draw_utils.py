import os

import cv2
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
        """
        Convert a Color enum to its RGB tuple for OpenCV.
        :param color: Color enum value
        :return: Tuple representing the RGB value
        """
        if not isinstance(color, Color):
            raise ValueError("Provided color must be an instance of the Color enum.")
        return color.value

    @staticmethod
    def draw_arrows(image, color=Color.WHITE, num_dots=5, max_distance=100):
        rgb_color = DrawUtils.color_to_palette(color)

        # Generate random positions for the two ends of the arrow
        height, width, _ = image.shape
        end1 = (random.randint(50, width - 50), random.randint(50, height - 50))
        end2 = (end1[0] + random.randint(-max_distance, max_distance), end1[1] + random.randint(-max_distance, max_distance))

        distance_ratio = max_distance / 100
        num_dots = int(num_dots * distance_ratio)

        # Draw cross-hairs at both ends
        crosshair_size = 5
        cv2.line(image, (end1[0] - crosshair_size, end1[1]), (end1[0] + crosshair_size, end1[1]), rgb_color, 1)
        cv2.line(image, (end1[0], end1[1] - crosshair_size), (end1[0], end1[1] + crosshair_size), rgb_color, 1)

        cv2.line(image, (end2[0] - crosshair_size, end2[1]), (end2[0] + crosshair_size, end2[1]), rgb_color, 1)
        cv2.line(image, (end2[0], end2[1] - crosshair_size), (end2[0], end2[1] + crosshair_size), rgb_color, 1)

        # Draw dotted line between the two ends
        for i in range(num_dots + 1):
            t = i / num_dots
            x = int((1 - t) * end1[0] + t * end2[0])
            y = int((1 - t) * end1[1] + t * end2[1])
            cv2.circle(image, (x, y), 1, rgb_color, -1)

    @staticmethod
    def draw_bounding_box(image):
        return


if __name__ == "__main__":
    path_clean = os.path.join("dataset", "clean")
    path_annotated = os.path.join("dataset", "annotated")

    for file in os.listdir(path_clean):
        image = cv2.imread(os.path.join(path_clean, file))
        DrawUtils.draw_arrows(image, Color.YELLOW, max_distance=1000)
        cv2.imwrite(os.path.join(path_annotated, file), image)

    # test_image = cv2.imread("car.jpg")
    # DrawUtils.draw_arrows(test_image, Color.LIGHT_BLUE)
    # cv2.imshow("Drawn arrows", test_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
