import cv2
import pytesseract
import easyocr

pytesseract.pytesseract.tesseract_cmd = (
    r"C:/Users/Apoorva/AppData/Local/Programs/Tesseract-OCR/tesseract"
)

class OCRApp:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.ocr_tool = self.decide_ocr_tool()

    def has_complex_fonts(self):
        # Convert the image to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply binary thresholding to get binary image
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply median blur to reduce noise
        denoised = cv2.medianBlur(binary, 3)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate area and perimeter for each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                continue
            
            # Calculate circularity
            circularity = 4 * 3.1415 * area / (perimeter * perimeter)
            
            # Check if contour is likely text (using circularity as a proxy)
            if circularity < 0.6:
                return True
        return False

    def has_multilingual_text(self):
        # Example function to check for multilingual text
        custom_config = r'--oem 3 --psm 6'  # Example custom configuration for pytesseract
        text = pytesseract.image_to_string(self.image, config=custom_config)
        
        # Check if the text contains non-Latin characters
        for char in text:
            if ord(char) > 128:  # Check if character code is beyond ASCII range
                return True
        return False

    def decide_ocr_tool(self):
        # Example criteria to decide OCR tool based on image analysis
        if self.has_complex_fonts():
            return "EasyOCR"
        elif self.has_multilingual_text():
            return "pytesseract"
        else:
            return "pytesseract"  # Default to pytesseract if criteria are not met

    def perform_ocr(self):
        recognized_text=[]
        print(f"Recommended OCR tool: {self.ocr_tool}")

        if self.ocr_tool == "EasyOCR":
            reader = easyocr.Reader(['en'])  # Initialize EasyOCR with English language
            result = reader.readtext(self.image_path)
            print("EasyOCR Result:")
            for (bbox,text, prob) in result:
                if prob>0.3:
                    print(text)
                    recognized_text.append(text)
            # print(recognized_text)
            full_text = "\n".join(recognized_text)
            # print(full_text)
        elif self.ocr_tool == "pytesseract":
            custom_config = r'--oem 3 --psm 6'  # Example custom configuration for pytesseract
            image = cv2.imread(self.image_path)
            img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform OCR using pytesseract
            result = pytesseract.image_to_data(img_RGB, config=custom_config, output_type=pytesseract.Output.DICT)

            # Print OCR results with confidence scores
            print("OCR Results:")
            for i in range(len(result['text'])):
                text = result['text'][i]
                confidence = int(result['conf'][i]) if 'conf' in result else None
                if confidence>50:
                    print(f"Text: {text}, Confidence: {confidence}")
                    recognized_text.append(text)
            full_text = "\n".join(recognized_text)
            # print("Full text",full_text)
        else:
            # print("No suitable OCR tool selected.")
            self.ocr_tool="No suitable OCR tool selected"

        return full_text,self.ocr_tool
    
def main(image_path):
    print("main enetred")
    ocr_app = OCRApp(image_path) 
    recognised_text,ocr_tool = ocr_app.perform_ocr()
    return recognised_text,ocr_tool


