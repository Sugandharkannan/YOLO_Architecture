import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import shutil
import base64
import json
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from backend.model_manager import YOLOModelManager
from backend.cam_generator import CAMGenerator
from backend.simulator import YOLOSimulator

app = FastAPI(title="YOLO Vision Explorer API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Model Manager
model_manager = YOLOModelManager()

# Global state to keep track of the last processed image and its feature maps
last_processed_img_path = None
last_feature_maps = {}
last_metadata = {}

class Hyperparameters(BaseModel):
    lr: float = 0.01
    batch_size: int = 16
    epochs: int = 50
    momentum: float = 0.937
    weight_decay: float = 0.0005
    box_loss_w: float = 7.5
    cls_loss_w: float = 0.5
    dfl_loss_w: float = 1.5
    mosaic: float = 1.0
    mixup: float = 0.0
    hsv_h: float = 0.015
    rotation: float = 0.0
    scaling: float = 0.5
    flipping: float = 0.5
    conf_thres: float = 0.25
    iou_thres: float = 0.7
    nms_thres: float = 0.45

def img_to_base64(path: str) -> str:
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ndarray_to_base64_jpeg(arr: np.ndarray, colormap: Optional[int] = None) -> str:
    # Make sure array is 8-bit
    if arr.dtype != np.uint8:
        # Scale to 0-255
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
            
    if colormap is not None:
        arr = cv2.applyColorMap(arr, colormap)
        
    _, buffer = cv2.imencode('.jpg', arr)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    model_version: str = Form("YOLOv8")
):
    global last_processed_img_path, last_feature_maps, last_metadata
    
    # Save the file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    last_processed_img_path = file_path
    
    # Update model version if changed
    if model_manager.current_version != model_version:
        model_manager.change_model(model_version)
        
    # Run Inference
    try:
        metadata, feature_maps = model_manager.run_inference(file_path)
        last_feature_maps = feature_maps
        last_metadata = metadata
        
        # Include base64 original image
        metadata["image_base64"] = img_to_base64(file_path)
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/api/layer-details")
def get_layer_details():
    return model_manager.get_layer_structure()

@app.get("/api/feature-maps")
def get_feature_maps(layer_name: str = Query(...)):
    global last_feature_maps, last_processed_img_path
    
    if not last_feature_maps or layer_name not in last_feature_maps:
        raise HTTPException(status_code=400, detail=f"No activations found for layer '{layer_name}'. Run upload first.")
        
    activation = last_feature_maps[layer_name]
    A = CAMGenerator._prepare_activation(activation)
    C = A.shape[0]
    
    # Generate Eigen-CAM for layer heatmap
    cv_img = cv2.imread(last_processed_img_path)
    h, w, _ = cv_img.shape
    cam_gray = CAMGenerator.generate_eigen_cam(activation, (w, h))
    cam_colored = CAMGenerator.apply_colormap_to_cam(cam_gray)
    cam_base64 = ndarray_to_base64_jpeg(cam_colored)
    
    # Extract top activated channels
    # Saliency heuristic: channels with highest mean and variance
    means = np.mean(A, axis=(1, 2))
    vars_ = np.var(A, axis=(1, 2))
    saliency = means * vars_
    
    # Get top 8 sorted channel indices
    top_channels = np.argsort(saliency)[::-1][:8].tolist()
    
    # Render top 8 channels as base64 images
    channels_data = []
    # If channels count is less than 8, just take what we have
    for idx in top_channels:
        channel_activation = A[idx]
        # Generate base64 representation of this channel map using Viridis
        base64_map = ndarray_to_base64_jpeg(channel_activation, cv2.COLORMAP_VIRIDIS)
        
        # Simulated descriptions based on layer index and channel attributes
        desc = "Detecting basic contours & boundaries"
        if "Layer 0" in layer_name or "Layer 1" in layer_name:
            if idx % 3 == 0:
                desc = "Extracting vertical edges and gradients"
            elif idx % 3 == 1:
                desc = "Extracting horizontal textures"
            else:
                desc = "Extracting high-contrast corners and shapes"
        elif "Layer 8" in layer_name or "Layer 9" in layer_name:
            if idx % 3 == 0:
                desc = "Segmenting foreground objects from background"
            elif idx % 3 == 1:
                desc = "Resolving semantic part clusters (limbs/wheels)"
            else:
                desc = "Fusing multi-scale bounding context"
        else:
            if idx % 2 == 0:
                desc = "Highlighting object color transitions"
            else:
                desc = "Extracting edge orientations and shadows"
                
        channels_data.append({
            "channel_id": idx,
            "description": desc,
            "image_base64": base64_map
        })
        
    return {
        "layer_name": layer_name,
        "channels_count": C,
        "eigen_cam_base64": cam_base64,
        "top_channels": channels_data,
        "explanation": YOLOSimulator.get_layer_explanation(layer_name)
    }

@app.get("/api/all-layer-heatmaps")
def get_all_layer_heatmaps():
    global last_feature_maps, last_processed_img_path
    
    if not last_feature_maps:
        raise HTTPException(status_code=400, detail="No activations found. Upload an image first.")
        
    results = []
    ordered_layers = model_manager.get_layer_structure()
    
    for layer in ordered_layers:
        name = layer["name"]
        if name in last_feature_maps:
            activation = last_feature_maps[name]
            # Generate a fast 120x120 thumbnail of the Eigen-CAM feature map
            cam_gray = CAMGenerator.generate_eigen_cam(activation, (120, 120))
            cam_colored = CAMGenerator.apply_colormap_to_cam(cam_gray)
            cam_base64 = ndarray_to_base64_jpeg(cam_colored)
            
            results.append({
                "layer_name": name,
                "layer_type": layer["type"],
                "image_base64": cam_base64
            })
            
    return results

@app.get("/api/cam")
def get_targeted_cam(
    layer_name: str = Query(...),
    bbox_json: str = Query(...)
):
    global last_feature_maps, last_processed_img_path
    
    if not last_feature_maps or layer_name not in last_feature_maps:
        raise HTTPException(status_code=400, detail="No activations found. Upload an image first.")
        
    try:
        bbox = json.loads(bbox_json) # [x1, y1, x2, y2]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bbox format. Must be JSON array of 4 floats.")
        
    activation = last_feature_maps[layer_name]
    cv_img = cv2.imread(last_processed_img_path)
    h, w, _ = cv_img.shape
    
    # Compute targeted saliency attention map
    attn_gray = CAMGenerator.generate_targeted_attention(activation, bbox, (w, h), layer_name)
    attn_colored = CAMGenerator.apply_colormap_to_cam(attn_gray)
    attn_base64 = ndarray_to_base64_jpeg(attn_colored)
    
    return {
        "layer_name": layer_name,
        "cam_base64": attn_base64
    }

@app.post("/api/simulate-training")
def simulate_training(params: Hyperparameters):
    params_dict = params.dict()
    metrics = YOLOSimulator.simulate_metrics(params_dict)
    curves = YOLOSimulator.generate_training_curves(params_dict)
    return {
        "metrics": metrics,
        "curves": curves
    }

@app.post("/api/gradient-flow")
def gradient_flow(params: Hyperparameters):
    return YOLOSimulator.generate_gradient_flow(params.dict())

@app.get("/api/weight-updates")
def weight_updates(layer_name: str = Query(...)):
    return YOLOSimulator.generate_weight_updates(layer_name)

@app.get("/api/explain")
def explain_prediction(
    class_name: str = Query(...),
    confidence: float = Query(...)
):
    return YOLOSimulator.get_prediction_explanation(class_name, confidence)

# Optionally serve upload assets statically
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
