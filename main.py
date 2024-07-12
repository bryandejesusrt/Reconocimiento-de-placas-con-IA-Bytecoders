import cv2
import numpy as np
import csv
import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv

class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Recognition")
        
        self.frame = ttk.Frame(self.root)
        self.frame.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(self.frame, width=640, height=480)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        self.treeview = ttk.Treeview(self.frame, columns=("car_id", "license_plate", "time"), show="headings")
        self.treeview.heading("car_id", text="Car ID")
        self.treeview.heading("license_plate", text="License Plate")
        self.treeview.heading("time", text="Time")
        self.treeview.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.start()
        
        self.update_treeview()

    def process_video(self):
        results = {}
        mot_tracker = Sort()
        coco_model = YOLO('yolov8n.pt')
        license_plate_detector = YOLO('./models/license_plate_detector.pt')
        cap = cv2.VideoCapture('./placa1.mp4')

        vehicles = [2, 3, 5, 7]
        frame_nmr = -1
        ret = True

        while ret:
            frame_nmr += 1
            ret, frame = cap.read()
            if ret:
                results[frame_nmr] = {}
                detections = coco_model(frame)[0]
                detections_ = []
                for detection in detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    if int(class_id) in vehicles:
                        detections_.append([x1, y1, x2, y2, score])

                track_ids = mot_tracker.update(np.asarray(detections_))
                license_plates = license_plate_detector(frame)[0]
                
                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                    if car_id != -1:
                        license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                        
                        if license_plate_text is not None:
                            capture_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                          'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                            'text': license_plate_text,
                                                                            'bbox_score': score,
                                                                            'text_score': license_plate_text_score,
                                                                            'time': capture_time}}
                            self.update_results(car_id, license_plate_text, capture_time)

                
                self.display_frame(frame)

        write_csv(results, './test.csv')

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.img_tk = img_tk
        self.root.update_idletasks()

    def update_results(self, car_id, license_plate, capture_time):
        self.treeview.insert("", "end", values=(car_id, license_plate, capture_time))

    def update_treeview(self):
        self.root.after(1000, self.update_treeview)

if __name__ == "__main__":
    import threading
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()
