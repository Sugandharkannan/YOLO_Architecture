import numpy as np
import random
from typing import Dict, List, Any, Tuple

class YOLOSimulator:
    @staticmethod
    def simulate_metrics(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate training metrics based on hyperparameters.
        """
        lr = float(params.get("lr", 0.01))
        batch_size = int(params.get("batch_size", 16))
        epochs = int(params.get("epochs", 50))
        momentum = float(params.get("momentum", 0.937))
        weight_decay = float(params.get("weight_decay", 0.0005))
        
        box_loss_w = float(params.get("box_loss_w", 7.5))
        cls_loss_w = float(params.get("cls_loss_w", 0.5))
        dfl_loss_w = float(params.get("dfl_loss_w", 1.5))
        
        mosaic = float(params.get("mosaic", 1.0))
        mixup = float(params.get("mixup", 0.0))
        rotation = float(params.get("rotation", 0.0))
        scaling = float(params.get("scaling", 0.5))
        
        conf_thres = float(params.get("conf_thres", 0.25))
        iou_thres = float(params.get("iou_thres", 0.7))

        # Core heuristics to estimate mAP
        base_map = 0.78 # Default baseline
        
        # Learning rate effect
        if lr > 0.05: # Too high -> unstable, exploding gradients
            lr_factor = -0.15 * (lr / 0.1)
        elif lr < 1e-4: # Too low -> slow convergence
            lr_factor = -0.10
        else: # Optimal zone
            lr_factor = 0.05 * (1.0 - abs(lr - 0.01) / 0.01)
            
        # Batch size effect
        batch_factor = 0.02 * (np.log2(batch_size) - np.log2(16))
        
        # Loss weight settings
        # Standard balanced: box_loss=7.5, cls_loss=0.5, dfl=1.5
        loss_balance = 1.0 - (abs(box_loss_w - 7.5) / 15.0 + abs(cls_loss_w - 0.5) / 2.0 + abs(dfl_loss_w - 1.5) / 5.0)
        loss_factor = 0.04 * max(-1.0, min(1.0, loss_balance))

        # Augmentation effect (helps generalization, increases accuracy)
        aug_factor = 0.03 * (mosaic + mixup * 1.5 + (rotation / 45.0) + (scaling * 0.5))

        # Final estimated metrics
        mAP50_95 = max(0.1, min(0.95, base_map + lr_factor + batch_factor + loss_factor + aug_factor))
        mAP50 = min(0.99, mAP50_95 * 1.35)
        
        # Precision & Recall adjusted by confidence threshold
        precision = min(0.99, mAP50 * (1.0 + (conf_thres - 0.25) * 0.3))
        recall = min(0.99, mAP50 * (1.0 - (conf_thres - 0.25) * 0.6))
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Speed estimate (ms per image)
        speed = 8.5 + (batch_size * 0.1) + (mosaic * 0.5)
        
        return {
            "mAP50_95": round(mAP50_95, 3),
            "mAP50": round(mAP50, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "inference_speed_ms": round(speed, 1)
        }

    @staticmethod
    def generate_training_curves(params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate epoch-by-epoch metrics to simulate active training logs.
        """
        epochs = int(params.get("epochs", 50))
        lr = float(params.get("lr", 0.01))
        
        metrics = YOLOSimulator.simulate_metrics(params)
        final_map50 = metrics["mAP50"]
        final_map95 = metrics["mAP50_95"]
        
        history = []
        
        # Instability simulation if LR is too high
        is_exploding = lr > 0.05
        
        for epoch in range(1, epochs + 1):
            progress = epoch / epochs
            
            # Simulated training/val losses
            if is_exploding and epoch > 15:
                # Loss goes to infinity / nan
                train_loss = float('nan') if epoch > 20 else random.uniform(5.0, 15.0)
                val_loss = float('nan') if epoch > 20 else random.uniform(6.0, 18.0)
                m50 = 0.0
                m95 = 0.0
                p = 0.0
                r = 0.0
            else:
                # Logarithmic convergence
                factor = 1.0 - np.exp(-4.0 * progress)
                train_loss = max(0.2, 3.5 - 3.0 * factor + random.uniform(-0.05, 0.05))
                val_loss = max(0.3, 3.8 - 2.8 * factor + random.uniform(-0.08, 0.08))
                
                # Metrics growth
                m50 = final_map50 * factor + random.uniform(-0.02, 0.02) * (1 - progress)
                m95 = final_map95 * factor + random.uniform(-0.02, 0.02) * (1 - progress)
                p = metrics["precision"] * factor + random.uniform(-0.02, 0.02) * (1 - progress)
                r = metrics["recall"] * factor + random.uniform(-0.02, 0.02) * (1 - progress)
            
            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4) if not np.isnan(train_loss) else "NaN",
                "val_loss": round(val_loss, 4) if not np.isnan(val_loss) else "NaN",
                "mAP50": round(max(0.0, m50), 3),
                "mAP50_95": round(max(0.0, m95), 3),
                "precision": round(max(0.0, p), 3),
                "recall": round(max(0.0, r), 3)
            })
            
        return history

    @staticmethod
    def generate_gradient_flow(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates gradient magnitudes for all layers based on learning rate.
        Checks for vanishing or exploding gradients.
        """
        lr = float(params.get("lr", 0.01))
        
        layers = [
            "Input", "Layer 0 (Conv)", "Layer 1 (Conv)", "Layer 2 (C2f)", 
            "Layer 3 (Conv)", "Layer 4 (C2f)", "Layer 5 (Conv)", "Layer 6 (C2f)", 
            "Layer 7 (Conv)", "Layer 8 (C2f)", "Layer 9 (SPPF)", "Layer 12 (C2f)", 
            "Layer 15 (C2f)"
        ]
        
        magnitudes = []
        status = "Normal"
        description = "Gradients are propagating stably through all network blocks."

        if lr > 0.05:
            status = "Exploding"
            description = "WARNING: Learning rate is too high. Gradient magnitudes are extremely large (exploding) in early layers, which will cause weights to oscillate and trigger NaNs."
            # Early layers have massive gradients, later layers explode too
            for i, layer in enumerate(layers):
                if layer == "Input":
                    magnitudes.append(0.0)
                else:
                    magnitudes.append(round(10.0 ** (i / 2.0) + random.uniform(10, 50), 3))
        elif lr < 1e-5:
            status = "Vanishing"
            description = "WARNING: Learning rate is too low. Gradient magnitudes are near zero (vanishing) in early convolutional layers, causing learning to stall."
            # Gradients die out rapidly in early layers
            for i, layer in enumerate(layers):
                if layer == "Input":
                    magnitudes.append(0.0)
                else:
                    magnitudes.append(round(1e-7 * (1.5 ** i), 8))
        else:
            # Stably decreasing/backpropagating gradient chart
            # Backpropagation propagates error backward, so earlier layers (indices 0,1,2) have smaller gradients, head has larger
            for i, layer in enumerate(layers):
                if layer == "Input":
                    magnitudes.append(0.0)
                else:
                    # Healthy gradient flow: e.g., head has 0.15, early layers have 0.01
                    val = 0.005 + (0.12 * (i / len(layers)) ** 2) + random.uniform(-0.002, 0.002)
                    magnitudes.append(round(max(1e-4, val), 5))

        return {
            "status": status,
            "description": description,
            "layers": layers,
            "magnitudes": magnitudes
        }

    @staticmethod
    def generate_weight_updates(layer_name: str) -> Dict[str, Any]:
        """
        Generates simulated pre/post training weight matrices for a selected layer.
        """
        # Define matrix size e.g. 8x8 for visual representation
        size = 8
        
        # Set seed depending on layer name to keep it consistent per layer
        np.random.seed(sum(ord(c) for c in layer_name) % 1000)
        
        # Generate initial weights (Matrix A)
        weight_a = np.random.normal(0, 0.2, (size, size))
        
        # Generate final weights (Matrix B) with slight updates
        # Early layers change less, deep layers or C2f blocks change more
        update_scale = 0.05
        if "C2f" in layer_name:
            update_scale = 0.08
        elif "SPPF" in layer_name:
            update_scale = 0.03
            
        noise = np.random.normal(0, update_scale, (size, size))
        weight_b = weight_a + noise
        
        # Percentage changes
        pct_change = np.abs(weight_b - weight_a) / (np.abs(weight_a) + 1e-5) * 100
        avg_pct_update = float(np.mean(pct_change))
        
        return {
            "layer": layer_name,
            "matrix_a": weight_a.tolist(),
            "matrix_b": weight_b.tolist(),
            "pct_change": pct_change.tolist(),
            "avg_pct_update": round(avg_pct_update, 2)
        }

    @staticmethod
    def get_prediction_explanation(class_name: str, confidence: float) -> Dict[str, Any]:
        """
        Returns explanations about why a specific object was detected.
        """
        explanations = {
            "person": {
                "features": ["Vertical aspect ratio", "Human-like silhouette", "Head and shoulder outline", "Limb joints geometry"],
                "layers": ["Layer 0 (Conv) - Detects low-level vertical contours", "Layer 8 (C2f) - Integrates torso & limb relationships", "Layer 15 (C2f) - Captures fine body boundary textures", "Layer 22 (Detect) - Evaluates person bounding boxes"],
                "description": "The YOLO architecture activated strongly on vertical geometric shapes with characteristic human aspect ratios, combined with skin/clothing color distributions."
            },
            "dog": {
                "features": ["Horizontal posture", "Four limbs structure", "Fur texture representation", "Muzzle and tail shapes"],
                "layers": ["Layer 1 (Conv) - Identifies fur edge gradients", "Layer 4 (C2f) - Captures limb shapes", "Layer 12 (C2f) - Resolves complex snout and ear geometries"],
                "description": "Detected based on quadruped layout patterns. Neck feature fusion merged early texture details (fur) with later structural abstractions (torso alignment)."
            },
            "cup": {
                "features": ["Cylindrical shape", "Opal/hollow top rim", "Handle loop curvature", "Bright specular highlights"],
                "layers": ["Layer 0 (Conv) - Detects circular curves", "Layer 2 (C2f) - Captures handle loops", "Layer 15 (C2f) - Localizes container volumes"],
                "description": "The model matched oval contours (from the top rim) and intersecting straight vertical lines forming the body of the cup, alongside handle shapes."
            },
            "default": {
                "features": ["Closed geometric contour", "Color contrast separation", "Object-like spatial aspect ratios"],
                "layers": ["Layer 2 (C2f) - Texture boundary contrast", "Layer 8 (C2f) - Intermediate shape structure", "Layer 12 (C2f) - Neck fusion localization"],
                "description": "Detected by matching general category feature shapes and aspect-ratio anchors at multi-scale detection levels."
            }
        }
        
        expl = explanations.get(class_name.lower(), explanations["default"])
        return {
            "class_name": class_name,
            "confidence": f"{round(confidence * 100, 1)}%",
            "reasons": expl["features"],
            "contributing_layers": expl["layers"],
            "summary": expl["description"]
        }

    @staticmethod
    def get_layer_explanation(layer_name: str) -> Dict[str, Any]:
        """
        Returns explanations about what a specific hidden layer does and what features it extracts.
        """
        explanations = {
            "Input": {
                "role": "Input Stage",
                "summary": "Loads the preprocessed RGB image. No features have been extracted yet. The values are standard pixel intensities normalized between 0.0 and 1.0.",
                "features_learned": ["Raw RGB channels", "Image spatial coordinates"],
                "active_regions": "Entire image surface",
                "important_channels": "0 (Red), 1 (Green), 2 (Blue)"
            },
            "Layer 0": {
                "role": "Edge & Gradient Detection (Early Backbone)",
                "summary": "Performs initial downsampling by applying 3x3 convolutions with stride 2. This layer acts as a directional gradient filter, identifying vertical lines, horizontal boundaries, and sharp corners.",
                "features_learned": ["Vertical contours", "Horizontal boundary lines", "High-frequency edge transitions"],
                "active_regions": "Object boundaries, contrast interfaces, and sharp corners",
                "important_channels": "1, 4, 11"
            },
            "Layer 1": {
                "role": "Texture & Orientation Filter (Early Backbone)",
                "summary": "Further downsamples spatial features while doubling channel capacity. It processes early edge lines to identify orientations, micro-textures, and high-contrast shadow lines.",
                "features_learned": ["Oriented textures", "Fine-grain surface gradients", "Shadow boundaries"],
                "active_regions": "Textured surfaces, hair/fur/fabric edges, and object intersections",
                "important_channels": "5, 12, 28"
            },
            "Layer 2": {
                "role": "CSP Local Geometry Aggregator (Backbone C2f)",
                "summary": "A Cross Stage Partial C2f block that enables deep gradient flow without computational bottleneck. It aggregates orientations and textures into localized geometry segments like circles, loops, and parallel intersections.",
                "features_learned": ["Circular curvatures", "Repetitive patterns", "Small intersecting joints"],
                "active_regions": "Curved object surfaces (wheels, handles, circular borders)",
                "important_channels": "3, 17, 30"
            },
            "Layer 3": {
                "role": "Spatial Shape Compressor (Mid Backbone)",
                "summary": "Downsamples features to a 80x80 grid to identify shape intersections. It acts as a spatial summarizer of early geometric elements.",
                "features_learned": ["Grid intersections", "Corner vertices", "Color/spatial boundaries"],
                "active_regions": "Structural corners, object junctions, and boundary vertices",
                "important_channels": "15, 34, 52"
            },
            "Layer 4": {
                "role": "Object Part Assembler (Backbone C2f)",
                "summary": "Processes 80x80 spatial features to assemble local contours into meaningful object parts (e.g., wheels, vehicle grills, human limbs, container lids).",
                "features_learned": ["Object-part shapes", "Medium-scale surface structures", "Symmetric components"],
                "active_regions": "Candidate object centers, repeating structures, and part boundaries",
                "important_channels": "9, 27, 45"
            },
            "Layer 5": {
                "role": "Semantic Context Compressor (Mid-Deep Backbone)",
                "summary": "Downsamples features to a 40x40 spatial footprint while increasing capacity to 128 channels. Prepares local shapes for high-level semantic cataloging.",
                "features_learned": ["Mid-scale structures", "Semantic category boundaries", "Regional contexts"],
                "active_regions": "Object hulls, localized foreground structures",
                "important_channels": "22, 64, 110"
            },
            "Layer 6": {
                "role": "Category Component Assembler (Backbone C2f)",
                "summary": "Aggregates localized parts into holistic semantic category structures (e.g. human bodies, animal poses, vehicle profiles).",
                "features_learned": ["Class pose structures", "Volumetric shapes", "Category outlines"],
                "active_regions": "Whole bodies, silhouettes, and large spatial object frames",
                "important_channels": "14, 73, 125"
            },
            "Layer 7": {
                "role": "Deep Spatial Downsampler (Deep Backbone)",
                "summary": "Compresses features to a 20x20 footprint (P5 level). This scale is responsible for representing large objects and broad semantic scenes with 256 channels.",
                "features_learned": ["Global category labels", "Coarse spatial layouts", "Deep semantic groupings"],
                "active_regions": "Central focus regions of large objects, bounding anchors",
                "important_channels": "48, 156, 210"
            },
            "Layer 8": {
                "role": "Global Semantic Abstractor (Deep Backbone C2f)",
                "summary": "Integrates deep backbone features into abstract category representations. Filters out minor background textures to focus solely on high-level target classes.",
                "features_learned": ["Abstract category shapes", "Coarse boundary structures", "Occlusion contexts"],
                "active_regions": "Object center anchors, foreground categories",
                "important_channels": "56, 128, 242"
            },
            "Layer 9": {
                "role": "Spatial Pyramid Pooling (SPPF)",
                "summary": "Applies multi-scale MaxPool (5x5, 9x9, 13x13) to capture broad global receptive fields. It allows the model to process objects of vastly different scales concurrently.",
                "features_learned": ["Global context", "Scale-invariant category shapes", "Global background suppression"],
                "active_regions": "Entire foreground objects and their local surroundings",
                "important_channels": "12, 115, 230"
            },
            "Layer 10": {
                "role": "PAN/FPN Feature Concatenator (Neck)",
                "summary": "Fuses upsampled deep semantic features from Layer 9 (20x20) with fine spatial features from Layer 6 (40x40). This connects high-level labels with low-level boundaries.",
                "features_learned": ["Fused semantic boundaries", "Multi-scale contour guides", "Fine object boundaries"],
                "active_regions": "Object edges, color boundaries, and spatial intersection boundaries",
                "important_channels": "88, 192, 310"
            },
            "Layer 12": {
                "role": "Neck Feature Fusion C2f (Neck)",
                "summary": "Refines multi-scale features in the PANet Neck, specializing in mapping object contours at medium scales (40x40 spatial footprint).",
                "features_learned": ["Refined spatial part boundaries", "Medium-scale category context"],
                "active_regions": "Medium sized object anchors, silhouettes, and foreground components",
                "important_channels": "25, 60, 92"
            },
            "Layer 14": {
                "role": "Low-Level Feature Concatenator (Neck)",
                "summary": "Concatenates upsampled mid-level Neck features with high-resolution early Backbone features from Layer 4 (80x80). Captures fine edge lines to assist in locating small objects.",
                "features_learned": ["Fine-resolution details", "High-frequency object borders", "Local coordinate guides"],
                "active_regions": "Very small object boundaries, texture junctions, and corner locations",
                "important_channels": "32, 114, 150"
            },
            "Layer 15": {
                "role": "Small Object Feature Fusion (Neck C2f)",
                "summary": "Fuses spatial details and semantic cues in the Neck at an 80x80 spatial scale. Prepares features for fine small-scale anchor grid predictions.",
                "features_learned": ["Small category outlines", "High-resolution target boundaries", "Edge details"],
                "active_regions": "Small categories (books, bottles, remote details) and close boundaries",
                "important_channels": "11, 42, 59"
            }
        }
        
        matched_key = "default"
        for key in explanations.keys():
            if key in layer_name:
                matched_key = key
                break
                
        if matched_key == "default":
            return {
                "role": "Intermediate Block",
                "summary": f"This intermediate layer processes activation maps for layer {layer_name}. It refines and abstracts feature gradients to assist in boundary detection and classification.",
                "features_learned": ["Intermediate contours", "Local textures", "Activation gradients"],
                "active_regions": "Foreground object areas and category interfaces",
                "important_channels": "4, 16, 24"
            }
            
        return explanations[matched_key]
