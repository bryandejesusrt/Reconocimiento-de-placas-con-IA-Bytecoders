from ultralytics import YOLO
import cv2
from util import get_car
import util
from sort.sort import *

# load models
coco_model = YOLO("yolov8n.pt")
licence_plate_detector = YOLO('./models/license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./video_modelo.mp4')

mot_tracker = Sort()

vehicles = [2, 3, 5, 7]
results = {}

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
        


# track vehicles
track_ids = mot_tracker.update(np.asarray(detections_))

# detect license plates
license_plates = licence_plate_detector(frame)[0]
for license_plates in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id, = license_plates

# assign license plate to car
xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plates, track_ids)

# crop license plate
license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

# process license plate
license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
_, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

# read license plate number


# write results
write_csv(results, './test.csv')