import os

import cv2
import numpy as np
from enum import Enum
import random
from tqdm import tqdm


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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            raise ValueError("No non-white regions found in the image.")

        height, width = gray.shape
        center = (width // 2, height // 2)

        contours = sorted(contours, key=lambda cnt: cv2.pointPolygonTest(cnt, center, True))
        ovary_contour = contours[0]

        if display_contour:
            contour_image = image.copy()
            cv2.drawContours(contour_image, [ovary_contour], -1, (0, 255, 0), 2)
            cv2.imshow("Ovary Contour", contour_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [ovary_contour], -1, 255, -1)
        points = np.column_stack(np.where(mask == 255))

        if len(points) < 2:
            raise ValueError("Insufficient points found in the ovary region.")

        point1 = tuple(points[random.randint(0, len(points) - 1)])
        point2 = tuple(points[random.randint(0, len(points) - 1)])
        return point1, point2

    @staticmethod
    def get_points_interactively(image_copy):
        selected_points = []

        def click_event(event, x, y, flags, param):
            nonlocal selected_points
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_points.append((y, x))
                cv2.drawMarker(image_copy, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
                cv2.imshow("Select two points", image_copy)

                if len(selected_points) == 2:
                    cv2.destroyAllWindows()

        cv2.imshow("Select two points", image_copy)
        cv2.setMouseCallback("Select two points", click_event)
        cv2.waitKey(0)

        if len(selected_points) != 2:
            print("Error: Two points were not selected.")
            return selected_points

        return selected_points

    @staticmethod
    def draw_arrows(image, ovarian_points, color=Color.WHITE, num_dots=30, interactive_mode=False):
        image_copy = image.copy()
        rgb_color = DrawUtils.color_to_palette(color)

        if interactive_mode:
            selected_points = DrawUtils.get_points_interactively(image_copy)
            end1, end2 = selected_points[0], selected_points[1]
        else:
            end1, end2 = ovarian_points

        try:
            crosshair_size = 5
            cv2.line(image_copy, (end1[1] - crosshair_size, end1[0]), (end1[1] + crosshair_size, end1[0]), rgb_color, 3)
            cv2.line(image_copy, (end1[1], end1[0] - crosshair_size), (end1[1], end1[0] + crosshair_size), rgb_color, 3)

            cv2.line(image_copy, (end2[1] - crosshair_size, end2[0]), (end2[1] + crosshair_size, end2[0]), rgb_color, 3)
            cv2.line(image_copy, (end2[1], end2[0] - crosshair_size), (end2[1], end2[0] + crosshair_size), rgb_color, 3)

            for i in range(num_dots + 1):
                t = i / num_dots
                x = int((1 - t) * end1[1] + t * end2[1])
                y = int((1 - t) * end1[0] + t * end2[0])
                cv2.circle(image_copy, (x, y), 1, rgb_color, -1)

        except ValueError as e:
            print(f"Error: {e}")

        return image_copy

    @staticmethod
    def remove_template_match(image, template_match_path, threshold=0.8, color=(255, 255, 255)):
        template = cv2.imread(template_match_path, cv2.IMREAD_COLOR)
        if template is None:
            raise FileNotFoundError(f"Template file not found: {template_match_path}")

        template_height, template_width = template.shape[:2]

        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

        loc = np.where(result >= threshold)

        processed_image = image.copy()

        for pt in zip(*loc[::-1]):
            cv2.rectangle(
                processed_image,
                pt,
                (pt[0] + template_width, pt[1] + template_height),
                color,
                -1
            )

        return processed_image

    @staticmethod
    def random_draw_text(image, ovarian_points, chance=1.0):
        if random.random() > chance:
            return image

        try:
            point1, _ = ovarian_points

            text = random.choice(["Rt. Ovary", "LL Ovary", "Li Ov"])
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2

            text_color = (0, 255, 255)
            cv2.putText(image, text, (point1[1], point1[0] - 10), font, font_scale, text_color, font_thickness)

        except ValueError as e:
            print(f"Error finding ovaries: {e}")

        return image

    @staticmethod
    def draw_bounding_box(image):
        return


if __name__ == "__main__":
    path_clean = os.path.join("train_set", "clean")
    path_annotated = os.path.join("train_set", "annotated")
    template_path = "train_set/template_match.png"
    template_path_2 = "train_set/template_match_2.png"

    for file in tqdm(os.listdir(path_clean)):
        image = cv2.imread(os.path.join(path_clean, file))
        processed_image = DrawUtils.remove_template_match(image, template_path)
        processed_image = DrawUtils.remove_template_match(processed_image, template_path_2, color=(0, 0, 0))
        ovarian_mask = DrawUtils.find_ovaries(processed_image, display_contour=False)
        ovarian_mask_2 = DrawUtils.find_ovaries(processed_image, display_contour=False)
        processed_image = DrawUtils.random_draw_text(processed_image, ovarian_mask, chance=0.6)
        drawn_image = DrawUtils.draw_arrows(processed_image, ovarian_mask_2, Color.WHITE, num_dots=60,
                                            interactive_mode=True)
        cv2.imwrite(os.path.join(path_annotated, file), drawn_image)
