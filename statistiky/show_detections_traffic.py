import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import os

# Funkce pro načtení YOLO detekcí z txt souboru
def load_yolo_detections_gt(file_path):
    detections = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = list(map(float, line.strip().split()))
            if values[0] != 2:
                continue
            detections.append(values)  # [classID, centerX, centerY, width, height]
    return detections

def load_yolo_detections(file_path):
    detections = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = list(map(float, line.strip().split()))
            detections.append(values)  # [classID, centerX, centerY, width, height]
    return detections

# Funkce pro vykreslení detekcí na obrázek
def draw_detections(image, detections, color):
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    
    for detection in detections:
        class_id, cx, cy, w, h = detection
        x1 = int((cx - w / 2) * img_width)
        y1 = int((cy - h / 2) * img_height)
        x2 = int((cx + w / 2) * img_width)
        y2 = int((cy + h / 2) * img_height)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

# Funkce pro načtení obrázku a souborů s detekcemi
def load_image_and_detections():
    img_path = filedialog.askopenfilename(title="Vyber obrázek", initialdir="./data/datasety/traffic/test/images/")  # Zrušíme omezení na typ souboru
    if img_path:
        img = Image.open(img_path)

        #det_file1 = filedialog.askopenfilename(title="Vyber první soubor s detekcemi", filetypes=[("Text files", "*.txt")])
        #det_file2 = filedialog.askopenfilename(title="Vyber druhý soubor s detekcemi", filetypes=[("Text files", "*.txt")])
        det_file1 = "./data/datasety/traffic/test/labels/" + os.path.basename(img_path)[:-3] + "txt"
        det_file2 = "./data/output/traffic/" + os.path.basename(img_path) + ".txt"
        if det_file1 and det_file2:
            detections1 = load_yolo_detections(det_file1)
            detections2 = load_yolo_detections(det_file2)

            img_copy = img.copy()
            
            # Vykreslení detekcí
            draw_detections(img_copy, detections1, color="red")
            draw_detections(img_copy, detections2, color="blue")

            # Zobrazení obrázku s detekcemi
            img_tk = ImageTk.PhotoImage(img_copy)
            canvas.create_image(0, 0, anchor="nw", image=img_tk)
            canvas.image = img_tk

# Vytvoření hlavního okna
root = tk.Tk()
root.title("YOLO Detekce v obrázku")

# Plátno pro zobrazení obrázku
canvas = tk.Canvas(root, width=800, height=600)
canvas.pack()

# Tlačítko pro načtení obrázku a detekcí
btn_load = tk.Button(root, text="Načíst obrázek a detekce", command=load_image_and_detections)
btn_load.pack()

# Spuštění aplikace
root.mainloop()