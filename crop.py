import cv2
from pdf2image import convert_from_path
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from tqdm import tqdm

import os


class Line:

    def __init__(self, line):
        self.x1, self.y1, self.x2, self.y2 = line
        if self.x1 > self.x2:
            self.x1, self.y1, self.x2, self.y2 = self.x2, self.y2, self.x1, self.y1
        self.angle = np.arctan2(self.y2 - self.y1, self.x2 - self.x1) * 180 / np.pi
        self.length = np.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)

    def p1(self):
        return int(self.x1), int(self.y1)

    def p2(self):
        return int(self.x2), int(self.y2)


CORE_COUNT = len(os.sched_getaffinity(0))


def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rotate pdf pages based on horizontal lines"
    )
    parser.add_argument("pdf", type=str, help="input pdf file path")
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=10,
        help="angle threshold to consider a line as horizontal",
    )
    # add debug flag to show images
    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="show each page with detected lines and rotation angle",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="out.pdf",
        required=False,
        help="output pdf file path",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    print("loading pdf file", args.pdf)
    images = convert_from_path(args.pdf, thread_count=CORE_COUNT)
    print("loaded", len(images), "pages")

    pil_images = []

    # cli progress bar

    print("start processing pages")
    for i, image in tqdm(enumerate(images), desc="Processing pages", total=len(images)):
        img_orig = np.array(image)
        img = cv2.resize(img_orig, None, fx=0.5, fy=0.5)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.createLineSegmentDetector()
        lines, width, prec, nfa = detector.detect(img_grey)

        horizontal_lines: list[Line] = []
        for line in lines:
            line = Line(line[0])
            # if nearly horizontal
            if -args.angle_threshold < line.angle < args.angle_threshold:
                # print("angle:", line.angle, "length:", line.length)
                horizontal_lines.append(line)

        # calculate avarage angle
        length_threshold = np.quantile([h.length for h in horizontal_lines], 0.6)
        rotation_angle_lines = [
            h for h in horizontal_lines if h.length > length_threshold
        ]
        angles = [h.angle for h in rotation_angle_lines]
        weights = [h.length for h in rotation_angle_lines]

        avg_angle = np.average(angles, weights=weights)

        most_left = int(min([line.x1 for line in rotation_angle_lines]))
        most_right = int(max([line.x2 for line in rotation_angle_lines]))
        most_top = int(min([line.y1 for line in rotation_angle_lines]))
        most_bottom = int(max([line.y2 for line in rotation_angle_lines]))

        if args.debug:
            # debug img
            debug_img = np.copy(img)
            for line in rotation_angle_lines:
                cv2.line(debug_img, line.p1(), line.p2(), (255, 0, 0), 1)

            # draw line with avg angle
            cv2.line(
                debug_img,
                (0, int(img.shape[0] / 2)),
                (
                    img.shape[1],
                    int(
                        img.shape[0] / 2
                        + img.shape[1] * np.tan(avg_angle * np.pi / 180)
                    ),
                ),
                (0, 255, 0),
                1,
            )

            # draw bounding box
            cv2.rectangle(
                debug_img,
                (int(most_left), int(most_top)),
                (most_right, most_bottom),
                (0, 0, 255),
                1,
            )

            cv2.imshow("debug", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # print("median angle:", avg_angle)
        # rotate image
        rows, cols = img_orig.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), avg_angle, 1)

        dest_img = cv2.warpAffine(
            img_orig,
            M,
            (cols, rows),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        # write rotated image to pil_images
        pil_images.append(Image.fromarray(dest_img))

    pil_images[0].save(args.output, "PDF", save_all=True, append_images=pil_images[1:])
