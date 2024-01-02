import cv2
import yolov5



# load model
model = yolov5.load('keremberke/yolov5n-license-plate')
#model= YOLOv5('yolov5/best.pt')
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

cap = cv2.VideoCapture('../video/195724.png')

results=[]
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to BGR format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect cars using YOLOv5
    result = model(frame, size=640)
    results.append(result)

for result in results:
    # parse results
    predictions = result.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    
    # show detection bounding boxes on image
    result.show()

    # save results into "results/" folder
    result.save(save_dir='results/')
    break
