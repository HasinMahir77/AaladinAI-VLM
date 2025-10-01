from ultralytics import YOLO

# Load a model
model = YOLO("yolo12n.pt").to("cpu")  # pretrained YOLO12n model

# Run batched inference on a list of images
results = model("https://www.youtube.com/watch?v=JqhdBCCUVyQ", stream=True)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen