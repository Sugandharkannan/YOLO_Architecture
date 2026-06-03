import torch
import numpy as np
import cv2
from typing import Dict, Any, Tuple

class CAMGenerator:
    @staticmethod
    def _prepare_activation(activation_tensor: Any) -> np.ndarray:
        if isinstance(activation_tensor, np.ndarray):
            act_np = activation_tensor
        else:
            act_np = activation_tensor.detach().cpu().numpy()
        
        # Remove batch dimension if it is 1
        if act_np.ndim > 1 and act_np.shape[0] == 1:
            act_np = act_np[0]
            
        ndim = act_np.ndim
        if ndim == 1:
            return act_np.reshape(-1, 1, 1)
        elif ndim == 2:
            C, L = act_np.shape
            side = int(np.sqrt(L))
            if side * side == L:
                H, W = side, side
                return act_np.reshape(C, H, W)
            else:
                H = 1
                W = L
                return act_np.reshape(C, H, W)
        elif ndim >= 3:
            shape = act_np.shape
            C = shape[0]
            H = shape[1]
            W = int(np.prod(shape[2:]))
            return act_np.reshape(C, H, W)
        else:
            return act_np.reshape(1, 1, 1)

    @staticmethod
    def generate_eigen_cam(activation_tensor: torch.Tensor, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Computes Eigen-CAM for a given activation tensor using a highly optimized covariance-based formulation.
        activation_tensor: torch.Tensor of shape [1, C, H, W]
        target_size: (width, height) to resize the final heatmap to
        """
        A = CAMGenerator._prepare_activation(activation_tensor)
        C, H, W = A.shape
        
        # 1. Downsample spatial dimensions if they are large (e.g. Layer 0 at 320x320)
        # This speeds up calculation on initial layers while preserving the primary principal components
        max_dim = 80
        if H > max_dim or W > max_dim:
            step_h = int(np.ceil(H / max_dim))
            step_w = int(np.ceil(W / max_dim))
            A_down = A[:, ::step_h, ::step_w]
        else:
            A_down = A
            
        C_down, H_down, W_down = A_down.shape
        A_flat = A_down.reshape(C_down, H_down * W_down)
        
        # 2. Zero center
        A_mean = A_flat - np.mean(A_flat, axis=1, keepdims=True)
        
        # 3. Optimized SVD via Covariance Matrix Eigen-Decomposition
        # Instead of SVD on C x (H*W) matrix directly, compute C x C covariance matrix: Cov = A_mean @ A_mean.T.
        # Since C is small, finding eigenvectors of Cov is extremely fast.
        # The principal right singular vector Vt[0] is proportional to A_mean.T @ u1.
        try:
            cov = np.dot(A_mean, A_mean.T)
            # np.linalg.eigh returns eigenvalues/vectors in ascending order
            eigvals, eigvecs = np.linalg.eigh(cov)
            u1 = eigvecs[:, -1]  # eigenvector corresponding to the largest eigenvalue
            
            # Project back to spatial dimensions
            cam = np.dot(A_mean.T, u1).reshape(H_down, W_down)
            
            # Correct the sign ambiguity: ensure the heatmap highlights positive activations
            if np.sum(cam * np.mean(A_down, axis=0)) < 0:
                cam = -cam
        except Exception:
            # Fallback if decomposition fails: average of activations across channels
            cam = np.mean(A_down, axis=0)

        # Normalize cam to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to original target image size
        cam_resized = cv2.resize(cam, target_size)
        return cam_resized

    @staticmethod
    def generate_targeted_attention(
        activation_tensor: torch.Tensor, 
        bbox: list, 
        original_size: Tuple[int, int],
        layer_name: str
    ) -> np.ndarray:
        """
        Generates an attention map targeted at a specific bounding box.
        This represents the spatial areas that contributed to detecting that object.
        """
        A = CAMGenerator._prepare_activation(activation_tensor)
        C, H, W = A.shape
        orig_w, orig_h = original_size
        
        # Find coordinates of the bbox mapped to the feature map scale
        x1, y1, x2, y2 = bbox
        
        # Map bbox to feature map coordinates
        feat_x1 = max(0, int((x1 / orig_w) * W))
        feat_y1 = max(0, int((y1 / orig_h) * H))
        feat_x2 = min(W, int((x2 / orig_w) * W))
        feat_y2 = min(H, int((y2 / orig_h) * H))
        
        # Create a mask in the feature map space
        mask = np.zeros((H, W), dtype=np.float32)
        if feat_x2 > feat_x1 and feat_y2 > feat_y1:
            mask[feat_y1:feat_y2, feat_x1:feat_x2] = 1.0
        else:
            mask[:] = 1.0 # Fallback to all region if bbox is invalid
            
        # Compute channel activations that are strongest in the bounding box
        channel_weights = np.zeros(C)
        for c in range(C):
            channel_weights[c] = np.sum(A[c] * mask) / (np.sum(A[c]) + 1e-6)
            
        # We compute weighted sum of activations
        cam = np.zeros((H, W), dtype=np.float32)
        for c in range(C):
            cam += channel_weights[c] * A[c]
            
        # Apply Gaussian blur to make it smooth and gorgeous
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            
        cam_resized = cv2.resize(cam, original_size)
        
        # Let's post-process the heat map so it peaks nicely around the object
        # Apply a Gaussian blur to create a smooth, modern heat flow
        cam_blurred = cv2.GaussianBlur(cam_resized, (21, 21), 0)
        if cam_blurred.max() > 0:
            cam_blurred = cam_blurred / cam_blurred.max()
            
        return cam_blurred

    @staticmethod
    def apply_colormap_to_cam(cam: np.ndarray) -> np.ndarray:
        """
        Converts a [0, 1] single-channel attention heatmap to an RGB JET colormap.
        """
        cam_8u = (cam * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_8u, cv2.COLORMAP_JET)
        # Convert BGR (from OpenCV) to RGB
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return heatmap_rgb
