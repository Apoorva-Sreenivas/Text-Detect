import tkinter as tk
import frontend2
from PIL import Image, ImageTk
from tkinter import font

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Main Application")
        self.root.geometry("400x400")
        img = Image.open("background-image.jpg")
        self.bg_image = ImageTk.PhotoImage(img)
        # Create a Label widget to hold the background image
        self.bg_label = tk.Label(root, image=self.bg_image)
        self.bg_label.place(relwidth=1, relheight=1) 

        self.bold_font = font.Font(family="Lucida Handwriting", size=24, weight="bold")
        self.norm_font = font.Font(family="Lucida Handwriting", size=16)

        self.heading = tk.Label(root, text="Text Detect", font=self.bold_font)
        self.heading.pack(pady=20)

        # Create a frame for the buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)

        # Create buttons
        self.button1 = tk.Button(self.button_frame, text="Recognise text",command=lambda:self.open_image_text_recognizer(1), font=self.norm_font)
        self.button2 = tk.Button(self.button_frame, text="Recognise Handwriting",command=lambda:self.open_image_text_recognizer(2), font=self.norm_font)
        self.button3 = tk.Button(self.button_frame, text="Detect License Plates",command=lambda:self.open_image_text_recognizer(3), font=self.norm_font)

        # Pack buttons in the frame
        self.button1.pack(side="top", padx=10,pady=10)
        self.button2.pack(side="top", padx=10,pady=10)
        self.button3.pack(side="bottom", padx=10,pady=10)
        
    
    def open_image_text_recognizer(self,x):
        # Create a new window
        new_window = tk.Toplevel(self.root)
        frontend2.ImageTextRecognizerApp(new_window,x)
        new_window.grab_set()
        new_window.wait_window()

# Create the main application window
root = tk.Tk()
main_app = MainApp(root)

# Run the application
root.mainloop()

