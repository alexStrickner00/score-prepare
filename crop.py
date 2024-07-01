import cv2
from pdf2image import convert_from_path
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


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


# create cli interface to get input pdf, output pdf, angle threshold using argparse
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
parser.add_argument(
    "--output", type=str, default="out.pdf", required=False, help="output pdf file path"
)

args = parser.parse_args()

print("loading pdf file", args.pdf)
images = convert_from_path(args.pdf)
print("loaded", len(images), "pages")


pil_images = []

# cli progress bar
from tqdm import tqdm

print("start processing pages")
for i, image in tqdm(enumerate(images), desc="Processing pages", total=len(images)):
    img_orig = np.array(image)
    img = cv2.resize(img_orig, None, fx=0.5, fy=0.5)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print("detecting lines in image", i)
    detector = cv2.createLineSegmentDetector()
    lines, width, prec, nfa = detector.detect(img_grey)
    # print("number of lines detected:", len(lines))

    horizontal_lines: list[Line] = []
    # detector.drawSegments(img, lines)
    for line in lines:
        line = Line(line[0])
        # if nearly horizontal
        if -args.angle_threshold < line.angle < args.angle_threshold:
            # print("angle:", line.angle, "length:", line.length)
            horizontal_lines.append(line)
            cv2.line(img, line.p1(), line.p2(), (255, 0, 0), 1)
        # cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
    # calculate avarage angle
    median_length = np.median([h.length for h in horizontal_lines])
    rotation_angle_lines = [h for h in horizontal_lines if h.length > median_length]
    angles = [h.angle for h in rotation_angle_lines]
    weights = [h.length for h in rotation_angle_lines]

    avg_angle = np.average(angles, weights=weights)
    # print("median angle:", avg_angle)
    # rotate image
    rows, cols = img_orig.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), avg_angle, 1)
    img_rot = cv2.warpAffine(img_orig, M, (cols, rows))

    # write rotated image to pil_images
    pil_images.append(Image.fromarray(img_rot))

pil_images[0].save(args.output, "PDF", save_all=True, append_images=pil_images[1:])
