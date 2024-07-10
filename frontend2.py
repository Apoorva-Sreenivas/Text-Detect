import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import text_detection
import license_plates
import handwriting
from tkinter import scrolledtext
from tkinter import font

class ImageTextRecognizerApp:
    def __init__(self, root,x):
        self.root = root
        self.root.title("Image Selector and Text Recognizer")
        self.root.geometry("600x650")
        self.x = x

        img = Image.open("background-image.jpg")
        self.bg_image = ImageTk.PhotoImage(img)
        # Create a Label widget to hold the background image
        self.bg_label = tk.Label(root, image=self.bg_image)
        self.bg_label.place(relwidth=1, relheight=1) 

        self.norm_font = font.Font(family="Lucida Handwriting", size=12)
        
        self.img_path = None

        # Create and pack the select image button
        self.select_button = tk.Button(root, text="Select Image", command=self.select_image, font=self.norm_font)
        self.select_button.pack(pady=10)
        
        # Create a label to display the image
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)
        
        # Create and pack the recognize text button
        self.recognize_button = tk.Button(root, text="Recognize Text", command=self.recognize_text, font=self.norm_font)
        self.recognize_button.pack(pady=5)

        self.ocr_tool_label = tk.Label(root,text="OCR Tool : ", font=self.norm_font)
        self.ocr_tool_label.pack(padx=10)

        # Create a label to display the recognized text
        self.text_label = tk.Label(root, text="Recognized Text: ", font=self.norm_font)
        self.text_label.pack(padx=10)

        self.scrollbar = tk.Scrollbar(root)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create a Text widget
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=10, yscrollcommand=self.scrollbar.set)
        self.text_area.pack(expand=True,padx=10)
        self.scrollbar.config(command=self.text_area.yview)

        self.probability_label= tk.Label(root,text="Confidence Level :", font=self.norm_font)
        self.probability_label.pack(padx=10)

       
    
    def select_image(self):
        # Open file dialog to select an image
        self.img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        
        if self.img_path:
            # Load the image
            img = Image.open(self.img_path)
            img.thumbnail((250,250))  # Resize the image to fit the display area
            img_tk = ImageTk.PhotoImage(img)
            
            # Display the image in the label
            self.img_label.config(image=img_tk)
            self.img_label.image = img_tk
    
    def recognize_text(self):
        recognized_text=""
        ocr_tool=""
        if self.img_path:
            if self.x==1:
                recognized_text,ocr_tool=text_detection.main(self.img_path)
                if ocr_tool=="EasyOCR":
                    self.probability_label.config(text="Confidence Level : 0.3")
                else:
                    self.probability_label.config(text="Confidence Level : 0.5")
            elif self.x==2:
                recognized_text=handwriting.main(self.img_path)
                ocr_tool = "neual network"
                self.probability_label.config(text="Confidence Level : 0.65")
            elif self.x==3:
                recognized_text=license_plates.main(self.img_path)
                ocr_tool="easy ocr"
                self.probability_label.config(text="Confidence Level : 0.5")
            # Display recognized text
            self.text_area.delete('1.0', tk.END)
            self.text_area.insert(tk.END,recognized_text)
            self.ocr_tool_label.config(text="OCR Tool : "+ocr_tool)

            



