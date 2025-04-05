"""
@author: Dr Yen Fred WOGUEM 

@description: This script trains a GAN model to generate image

"""


from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

start_time = datetime.now()  # Start timer


def train_model():
    """
    Train a YOLOv8 model on COCO128 dataset
    Returns path to best trained weights
    """
    print("Starting YOLOv8 training...")
    
    # Load pre-trained model (nano version)
    model = YOLO("yolov8n.pt")  # Try yolov8s.pt/m.pt/l.pt/x.pt for larger models
    
    # Train the model with augmentation on CPU
    results = model.train(
        data="coco128.yaml",       # Dataset config file
        epochs=10,                # Number of training epochs
        imgsz=640,                # Input image size
        batch=8,                  # Reduced batch size for CPU
        patience=10,              # Early stopping patience
        lr0=0.01,                # Initial learning rate
        device="cpu",             # Force CPU training
        workers=4,                # Reduced workers for CPU
        name='my_coco128_run',    # Experiment name
        augment=True,             # Enable augmentation
        hsv_h=0.015,             # Hue augmentation
        hsv_s=0.7,               # Saturation augmentation
        hsv_v=0.4,               # Value augmentation
        degrees=45,              # Rotation augmentation
        flipud=0.5,              # Vertical flip probability
    )
    
    print("Training completed!")
    return f"runs/detect/my_coco128_run/weights/best.pt"  # Return path to best weights

def run_inference(model_path, image_path):
    """
    Run object detection on an image
    Args:
        model_path: Path to trained weights
        image_path: Path to input image
    """
    print(f"Running inference on {image_path}")
    
    # Load trained model (automatically uses CPU if no GPU available)
    model = YOLO(model_path)
    
    # Perform detection
    results = model(image_path, 
                   save=True,     # Save results
                   conf=0.5,      # Confidence threshold
                   iou=0.45,      # IoU threshold
                   show_labels=True,
                   show_conf=True)
    
    # Display results
    for r in results:
        # Get annotated image (BGR format)
        im_array = r.plot()  
        
        # Convert to RGB for matplotlib
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.imshow(im_rgb)
        plt.axis('off')
        plt.title("YOLOv8 Detection Results")
        #plt.show()
        
        # Save detection image
        cv2.imwrite("Detection_output.jpg", im_array)
        print("Saved results to detection_output.jpg")

def real_time_detection(model_path, video_path=None):
    """
    Run real-time object detection using webcam
    Args:
        model_path: Path to trained weights
    """
    print("Starting real-time detection (Press 'q' to quit)")
    
    # Load model (will use CPU if no GPU available)
    model = YOLO(model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0 if video_path is None else video_path)  # Webcam ou vidéo
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection (stream=True for faster processing)
        results = model(frame, stream=True)  
        
        for r in results:
            # Draw bounding boxes
            annotated_frame = r.plot()
            
            # Display
            cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# --- Main Execution Flow ---

# First train the model
print("Starting training phase")
trained_model_path = train_model()

# Then perform inference with the trained model
print("\nStarting inference phase")
sample_image = "https://ultralytics.com/images/bus.jpg"  # Sample image
run_inference(trained_model_path, sample_image)

# print("\nStarting real-time detection")
# real_time_detection(trained_model_path)  # Webcam
# real_time_detection(trained_model_path, "./test.mp4")  # Video



end_time = datetime.now()  # End of timer
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")











