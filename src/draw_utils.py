import os

import cv2
import numpy as np
from enum import Enum
import random


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
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            raise ValueError("No non-white regions found in the image.")

        height, width = gray.shape
        center = (width // 2, height // 2)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
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
    def random_draw_heatmap(image):
        heatmap_png_path = "../media/heatmap.png"

        # if random.random() < 0.5:
        #     return image

        heatmap = cv2.imread(heatmap_png_path, cv2.IMREAD_UNCHANGED)
        heatmap_height, heatmap_width = heatmap.shape[:2]
        heatmap = cv2.resize(heatmap, (int(heatmap_width * 0.6), int(heatmap_height * 0.6)))
        heatmap_height, heatmap_width = heatmap.shape[:2]
        image_height, image_width = image.shape[:2]

        x_offset = int((image_width - heatmap_width) // 1.05)
        y_offset = (image_height - heatmap_height) // 6

        alpha_channel = heatmap[:, :, 3] / 255.0
        alpha_mask = cv2.merge([alpha_channel, alpha_channel, alpha_channel])
        region = image[y_offset:y_offset + heatmap_height, x_offset:x_offset + heatmap_width]
        heatmap_rgb = heatmap[:, :, :3]
        blended_region = cv2.convertScaleAbs(
            alpha_mask * heatmap_rgb + (1 - alpha_mask) * region
        )

        image[y_offset:y_offset + heatmap_height, x_offset:x_offset + heatmap_width] = blended_region

        return image

    @staticmethod
    def draw_bounding_box(image):
        return


if __name__ == "__main__":
    dataset_path = "draw"
    path_clean = os.path.join(dataset_path, "clean")
    path_annotated = os.path.join(dataset_path, "annotated")

    for filename in os.listdir(path_clean):
        image = cv2.imread(os.path.join(path_clean, filename))
        ovarian_mask = DrawUtils.find_ovaries(image, display_contour=False)
        ovarian_mask_2 = DrawUtils.find_ovaries(image, display_contour=False)
        ovarian_mask_3 = DrawUtils.find_ovaries(image, display_contour=False)
        processed_image = DrawUtils.random_draw_text(image, ovarian_mask, chance=0.6)
        # processed_image = DrawUtils.random_draw_heatmap(processed_image)
        drawn_image = DrawUtils.draw_arrows(processed_image, ovarian_mask_2, Color.WHITE, num_dots=60,
                                            interactive_mode=False)
        drawn_image = DrawUtils.draw_arrows(drawn_image, ovarian_mask_3, Color.WHITE, num_dots=60,
                                            interactive_mode=False)
        cv2.imwrite(os.path.join(path_annotated, filename), drawn_image)
