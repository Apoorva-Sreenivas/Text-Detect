import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

class LicensePlateDetector:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = None
        self.gray = None
        self.edged = None
        self.location = None
        self.warped_image = None

    def preprocess_image(self):
        # Read the image
        self.img = cv2.imread(self.img_path)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Display the grayscale image
        # plt.imshow(cv2.cvtColor(self.gray, cv2.COLOR_BGR2RGB))
        # plt.title("Grayscale Image")
        # plt.show()

        # Noise reduction and edge detection
        bfilter = cv2.bilateralFilter(self.gray, 11, 17, 17)
        self.edged = cv2.Canny(bfilter, 30, 200)

        # Display the edges
        # plt.imshow(cv2.cvtColor(self.edged, cv2.COLOR_BGR2RGB))
        # plt.title("Edge Detection")
        # plt.show()

    def find_license_plate(self):
        # Find contours
        keypoints = cv2.findContours(self.edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # Find the rectangular contour
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                self.location = approx
                break

        # Print the found location
        # print("Location of license plate contour:", self.location)

    def perspective_transform(self):
        if self.location is not None:
            rect = cv2.boundingRect(self.location)
            x, y, w, h = rect
            cropped = self.img[y:y+h, x:x+w].copy()

            pts = self.location - self.location.min(axis=0)
            mask = np.zeros(cropped.shape[:2], np.uint8)
            cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

            dst = cv2.bitwise_and(cropped, cropped, mask=mask)

            bg = np.ones_like(cropped, np.uint8) * 255
            cv2.bitwise_not(bg, bg, mask=mask)
            dst2 = bg + dst

            # Perspective correction
            rect = cv2.minAreaRect(self.location)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            width = int(rect[1][0])
            height = int(rect[1][1])

            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height-1],
                                [0, 0],
                                [width-1, 0],
                                [width-1, height-1]], dtype="float32")

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            self.warped_image = cv2.warpPerspective(self.img, M, (width, height), flags=cv2.INTER_LINEAR)

            if height > width:
                self.warped_image = cv2.rotate(self.warped_image, cv2.ROTATE_90_CLOCKWISE)

            # Denoising and sharpening
            self.warped_image = cv2.cvtColor(self.warped_image, cv2.COLOR_BGR2GRAY)
            self.warped_image = cv2.fastNlMeansDenoising(self.warped_image, None, 30, 7, 21)
            self.warped_image = cv2.GaussianBlur(self.warped_image, (3, 3), 0)
            self.warped_image = cv2.adaptiveThreshold(self.warped_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def annotate_image(self):
        if self.warped_image is not None:
            
            # Read text from the cropped image using EasyOCR
            reader = easyocr.Reader(['en'])
            result = reader.readtext(self.warped_image)

            # Print the result
            # print("OCR Result:", result)

            if result:
                text = result[0][-2]
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Draw text and rectangle on the original image
                res = cv2.putText(self.img, text=text, org=(self.location[0][0][0], self.location[1][0][1] + 60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                res = cv2.rectangle(self.img, tuple(self.location[0][0]), tuple(self.location[2][0]), (0, 255, 0), 3)

                # Display the final image
                plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
                plt.title("Detected License Plate")
                plt.show()
                return text
            else:
                # print("No text detected")
                return("No text detetcted")

    def detect_license_plate(self):
        self.preprocess_image()
        self.find_license_plate()
        self.perspective_transform()

        recognised_text = ""

        # Display the warped image
        if self.warped_image is not None:
            plt.imshow(cv2.cvtColor(self.warped_image, cv2.COLOR_BGR2RGB))
            plt.title("Warped Image")
            plt.show()

            recognised_text=self.annotate_image()
        else:
            # print("No license plate found or unable to correct perspective.")
            recognised_text="No license plate found or unable to correct perspective."
        return recognised_text

# Main execution
def main(img_path):
    # print("main of license plateenetred")
    detector = LicensePlateDetector(img_path)
    plates=detector.detect_license_plate()
    return plates


