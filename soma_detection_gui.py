#!/usr/bin/env python3
"""
Soma Detection GUI
==================
A Streamlit web application for detecting and counting somas in images
using the trained U-Net model.
"""

import streamlit as st
import torch
from pathlib import Path
from PIL import Image
import io
import time
import zipfile
import tempfile
import os
from segmentation_models_pytorch import Unet
import sys
sys.setrecursionlimit(10000)

import numpy as np
import cv2

# Page configuration
st.set_page_config(
    page_title="DendriteIQ",
    page_icon="🧠",
    layout="wide"
)

# Constants
SOMA_MODEL_PATH = "soma_unet.pt"
JOINT_MODEL_PATH = "joint_unet.pt"
IMPROVED_JOINT_MODEL_PATH = "joint_unet_improved.pt"
AGGRESSIVE_JOINT_MODEL_PATH = "joint_unet_aggressive.pt"
BALANCED_JOINT_MODEL_PATH = "joint_unet_balanced.pt"
IMG_DIR = Path("images")
MASK_DIR = Path("masks")

@st.cache_resource
def load_model(model_path):
    """Load the trained U-Net model"""
    try:
        # Determine number of classes based on model file
        if any(x in model_path for x in ["joint", "balanced", "aggressive", "improved"]):
            classes = 3  # Background, soma, dendrite
        else:
            classes = 1  # Soma only
        
        # Load model architecture
        model = Unet("resnet18", in_channels=3, classes=classes, activation=None)
        
        # Load trained weights
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            return model, classes
        else:
            st.error(f"Model file {model_path} not found!")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def load_model_direct(model_path):
    """Load model directly without caching for reliable switching"""
    try:
        # Determine number of classes based on model file
        if any(x in model_path for x in ["joint", "balanced", "aggressive", "improved"]):
            classes = 3  # Background, soma, dendrite
        else:
            classes = 1  # Soma only
        
        # Load model architecture
        model = Unet("resnet18", in_channels=3, classes=classes, activation=None)
        
        # Load trained weights
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            return model, classes
        else:
            st.error(f"Model file {model_path} not found!")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image):
    """Preprocess image for model inference"""
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to BGR if RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Convert to RGB for model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    image_norm = image_rgb.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0)
    
    return image_tensor, image_rgb

def predict_somas_and_dendrites(model, image_tensor, num_classes=1):
    """Predict soma and dendrite locations using the model"""
    with torch.no_grad():
        # Get model prediction
        logits = model(image_tensor)
        
        if num_classes == 1:
            # Single class: apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            soma_mask = probs.squeeze().cpu().numpy()
            dendrite_mask = None
        else:
            # Multi-class: apply softmax and get both soma and dendrite classes
            probs = torch.softmax(logits, dim=1)
            soma_mask = probs[:, 1].squeeze().cpu().numpy()  # Class 1 = soma
            dendrite_mask = probs[:, 2].squeeze().cpu().numpy()  # Class 2 = dendrite
        
        return soma_mask, dendrite_mask

def count_somas(mask, threshold=0.5):
    """Count somas in the predicted mask and return confidence scores"""
    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Filter out background (label 0) and small components
    soma_count = 0
    soma_centers = []
    soma_confidences = []
    
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Stricter minimum area threshold
        if area < 100:  # Increased from 50 to 100
            continue
            
        # Get the region for this component
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Additional filtering for false positives
        # Check aspect ratio (somas are typically roundish)
        aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
        if aspect_ratio < 0.4:  # Stricter aspect ratio (was 0.3)
            continue
            
        # Check if area is reasonable for a soma (not too large)
        if area > 3000:  # Reduced from 5000 to 3000
            continue
            
        # Edge filtering - avoid detections too close to image edges
        img_height, img_width = mask.shape
        center_x = int(centroids[i][0])
        center_y = int(centroids[i][1])
        
        # Skip if too close to edges (likely artifacts)
        edge_margin = 50
        if (center_x < edge_margin or center_x > img_width - edge_margin or 
            center_y < edge_margin or center_y > img_height - edge_margin):
            continue
            
        # Calculate average confidence for this region
        region_mask = labels[y:y+h, x:x+w] == i
        region_probs = mask[y:y+h, x:x+w]
        avg_confidence = np.mean(region_probs[region_mask])
        
        # Additional confidence threshold - require higher confidence for small areas
        if area < 200 and avg_confidence < 0.7:  # Small areas need high confidence
            continue
            
        soma_count += 1
        soma_centers.append((center_x, center_y))
        soma_confidences.append(avg_confidence)
    
    return soma_count, soma_centers, soma_confidences, binary_mask

def filter_dendrite_artifacts(dendrite_mask, min_area=100, max_area=3000):
    """Filter out autofluorescence and artifacts from dendrite mask"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(dendrite_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    filtered_mask = np.zeros_like(cleaned_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
        # Always keep very large or long components (likely dendrites)
        if area > 5000 or width > 100 or height > 100:
            filtered_mask[labels == i] = 255
            continue
        if area < min_area or area > max_area:
            continue
        if aspect_ratio > 0.8:
            continue
        filtered_mask[labels == i] = 255
    return filtered_mask

def aggressive_gut_granule_filter(dendrite_mask, min_area=200, max_area=3000, max_circularity=0.7, min_aspect_ratio=0.2):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dendrite_mask, connectivity=8)
    filtered_mask = np.zeros_like(dendrite_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        # Always keep very large or long components (likely dendrites)
        if area > 5000 or width > 100 or height > 100:
            filtered_mask[labels == i] = 255
            continue
        aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
        if area < min_area or area > max_area or aspect_ratio > max_circularity or aspect_ratio < min_aspect_ratio:
            continue
        filtered_mask[labels == i] = 255
    return filtered_mask

def connect_faded_dendrites(dendrite_mask, max_gap=50, min_overlap=0.3):
    """Connect dendrite segments that fade in/out gradually"""
    # Apply more aggressive morphological dilation to expand dendrite regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # Larger kernel
    dilated_mask = cv2.dilate(dendrite_mask, kernel, iterations=4)  # More iterations
    
    # Find contours of the dilated mask
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 2:
        return dendrite_mask
    
    # Create a copy of the original mask
    connected_mask = dendrite_mask.copy()
    
    # For each pair of contours, check if they should be connected
    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            # Get bounding boxes
            x1, y1, w1, h1 = cv2.boundingRect(contours[i])
            x2, y2, w2, h2 = cv2.boundingRect(contours[j])
            
            # Check if bounding boxes are close
            center1 = (x1 + w1//2, y1 + h1//2)
            center2 = (x2 + w2//2, y2 + h2//2)
            center_dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            if center_dist <= max_gap:
                # More permissive angle check for dendrites
                angle_diff = abs(np.arctan2(center2[1] - center1[1], center2[0] - center1[0]))
                if angle_diff < np.pi/2:  # Within 90 degrees (very permissive)
                    # Create a thicker, more gradual connection
                    points = np.array([center1, center2], dtype=np.int32)
                    cv2.polylines(connected_mask, [points], False, 255, 6)  # Even thicker line
    
    # Apply final morphological closing to smooth connections
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    connected_mask = cv2.morphologyEx(connected_mask, cv2.MORPH_CLOSE, kernel)
    
    return connected_mask

def calculate_dendrite_length(dendrite_mask):
    """Calculate the total length of dendrites in pixels by skeletonizing and counting pixels"""
    if dendrite_mask is None or np.sum(dendrite_mask) == 0:
        return 0
    
    # Ensure mask is binary
    binary_mask = (dendrite_mask > 0).astype(np.uint8)
    
    # Skeletonize the dendrite mask to get single-pixel width representation
    # Use morphological operations to create skeleton
    kernel = np.ones((3,3), np.uint8)
    skeleton = np.zeros_like(binary_mask)
    img = binary_mask.copy()
    
    while True:
        # Morphological opening
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        
        # Extract the result and write it on skel
        temp = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        
        if cv2.countNonZero(img) == 0:
            break
    
    # Count the number of white pixels in the skeleton (this is the total length)
    total_length = np.sum(skeleton > 0)
    
    return total_length

def create_overlay(image, soma_mask, soma_centers, soma_confidences, threshold=0.5, dendrite_mask=None, dendrite_threshold=0.3):
    """Create overlay visualization with confidence colors"""
    # Create colored mask
    colored_mask = np.zeros_like(image)
    
    # Add soma regions (red) ONLY for filtered/accepted somas, using the full region
    binary_mask = (soma_mask > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    for i in range(1, num_labels):
        center_x = int(centroids[i][0])
        center_y = int(centroids[i][1])
        if (center_x, center_y) in soma_centers:
            colored_mask[labels == i] = [255, 0, 0]  # Red for somas
    
    # Add dendrite regions (blue) if available
    if dendrite_mask is not None:
        colored_mask[dendrite_mask > 0] = [0, 0, 255]  # Blue for dendrites
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 0.5, colored_mask, 0.5, 0)
    
    # Draw soma centers with confidence-based colors
    for i, (center_x, center_y) in enumerate(soma_centers):
        confidence = soma_confidences[i]
        # Color based on confidence: Green (high) to Yellow (medium) to Red (low)
        if confidence > 0.8:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.6:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        cv2.circle(overlay, (center_x, center_y), 5, color, -1)
    
    return overlay

def process_single_image(model, image, soma_threshold=0.5, num_classes=1, dendrite_threshold=0.3, min_area=100, max_area=3000, connect_dendrites=False, max_gap=20):
    """Process a single image and return results"""
    # Initialize timing variables
    preprocessing_time = 0
    prediction_time = 0
    counting_time = 0
    
    # Preprocess with timing
    start_time = time.time()
    image_tensor, image_rgb = preprocess_image(image)
    preprocessing_time = time.time() - start_time
    
    # Predict with timing
    start_time = time.time()
    soma_mask, dendrite_mask = predict_somas_and_dendrites(model, image_tensor, num_classes)
    prediction_time = time.time() - start_time
    
    # Count somas with timing
    start_time = time.time()
    soma_count, soma_centers, soma_confidences, soma_binary_mask = count_somas(soma_mask, soma_threshold)
    counting_time = time.time() - start_time
    
    # Filter dendrite artifacts if dendrite mask exists
    filtered_dendrite_mask = None
    raw_dendrite_mask = None
    if dendrite_mask is not None:
        raw_dendrite_mask = dendrite_mask.copy()  # Save raw probability map for debug
        # Apply artifact filtering
        dendrite_binary = (dendrite_mask > dendrite_threshold).astype(np.uint8) * 255
        filtered_dendrite_mask = filter_dendrite_artifacts(dendrite_binary, min_area, max_area)
        
        # Apply aggressive gut granule filtering
        filtered_dendrite_mask = aggressive_gut_granule_filter(filtered_dendrite_mask, min_area, max_area, max_circularity=0.7, min_aspect_ratio=0.2)
        
        # Connect faded dendrites if enabled
        if connect_dendrites:
            filtered_dendrite_mask = connect_faded_dendrites(filtered_dendrite_mask, max_gap)
    
    # Calculate total processing time
    total_time = preprocessing_time + prediction_time + counting_time
    
    # Calculate dendrite length if dendrite mask exists
    dendrite_length = 0
    if filtered_dendrite_mask is not None:
        dendrite_length = calculate_dendrite_length(filtered_dendrite_mask)
    
    # Create overlay
    overlay = create_overlay(image_rgb, soma_mask, soma_centers, soma_confidences, soma_threshold, filtered_dendrite_mask, dendrite_threshold)
    
    # Convert overlay back to PIL for saving
    overlay_pil = Image.fromarray(overlay)
    
    return {
        'soma_count': soma_count,
        'soma_centers': soma_centers,
        'soma_confidences': soma_confidences,
        'dendrite_length': dendrite_length,
        'overlay': overlay_pil,
        'original': image,
        'soma_mask': soma_mask,
        'dendrite_mask': filtered_dendrite_mask,
        'raw_dendrite_mask': raw_dendrite_mask,  # For debug
        'timing': {
            'preprocessing': preprocessing_time,
            'prediction': prediction_time,
            'counting': counting_time,
            'total': total_time
        }
    }

def create_download_zip(results, filenames):
    """Create a ZIP file with all processed images"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, (result, filename) in enumerate(zip(results, filenames)):
            # Save overlay image
            overlay_bytes = io.BytesIO()
            result['overlay'].save(overlay_bytes, format='PNG', quality=95)
            overlay_bytes.seek(0)
            zip_file.writestr(f"processed_{filename}", overlay_bytes.getvalue())
            
            # Save original image
            original_bytes = io.BytesIO()
            result['original'].save(original_bytes, format='PNG', quality=95)
            original_bytes.seek(0)
            zip_file.writestr(f"original_{filename}", original_bytes.getvalue())
            
            # Save results summary
            summary = f"Results for {filename}:\n"
            summary += f"Somas detected: {result['soma_count']}\n"
            summary += f"Average confidence: {np.mean(result['soma_confidences']):.1%}\n"
            summary += f"Processing time: {result['timing']['total']:.3f}s\n"
            summary += f"Individual confidences: {[f'{c:.1%}' for c in result['soma_confidences']]}\n"
            zip_file.writestr(f"summary_{filename}.txt", summary)
    
    zip_buffer.seek(0)
    return zip_buffer

def auto_adjust_brightness_contrast(image):
    """
    Automatically adjust brightness and contrast for optimal soma and dendrite detection.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        numpy array: Adjusted image
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # Convert to grayscale for analysis
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    
    # Calculate current statistics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    min_val = np.min(gray)
    max_val = np.max(gray)
    current_range = max_val - min_val
    
    # Target values for optimal detection
    target_mean = 128  # Middle brightness
    target_std = 50    # Good contrast
    target_range = 200 # Good dynamic range
    
    # Calculate adjustment factors
    if mean_brightness > 0:
        brightness_factor = target_mean / mean_brightness
    else:
        brightness_factor = 1.0
    
    if std_brightness > 0:
        contrast_factor = target_std / std_brightness
    else:
        contrast_factor = 1.0
    
    # Limit factors to reasonable ranges
    brightness_factor = np.clip(brightness_factor, 0.5, 2.0)
    contrast_factor = np.clip(contrast_factor, 0.5, 2.0)
    
    # Apply adjustments
    adjusted_gray = gray.astype(np.float32)
    
    # Brightness adjustment
    adjusted_gray = adjusted_gray * brightness_factor
    
    # Contrast adjustment using histogram equalization for better distribution
    if current_range > 0:
        # Normalize to 0-1
        normalized = (adjusted_gray - np.min(adjusted_gray)) / (np.max(adjusted_gray) - np.min(adjusted_gray))
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply((normalized * 255).astype(np.uint8))
        adjusted_gray = enhanced.astype(np.float32)
    
    # Ensure values are in valid range
    adjusted_gray = np.clip(adjusted_gray, 0, 255)
    
    # Convert back to original format
    if len(img_array.shape) == 3:
        # For color images, apply to all channels
        adjusted_img = img_array.astype(np.float32)
        for channel in range(img_array.shape[2]):
            adjusted_img[:, :, channel] = adjusted_gray
        adjusted_img = np.clip(adjusted_img, 0, 255).astype(np.uint8)
    else:
        adjusted_img = adjusted_gray.astype(np.uint8)
    
    return adjusted_img

def analyze_image_quality(image):
    """
    Analyze image quality and determine if adjustment is needed.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        dict: Quality metrics and recommendations
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert to grayscale for analysis
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Calculate quality metrics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    min_val = np.min(gray)
    max_val = np.max(gray)
    dynamic_range = max_val - min_val
    
    # Determine if adjustment is needed
    needs_adjustment = False
    adjustment_reason = []
    
    if mean_brightness < 80 or mean_brightness > 180:
        needs_adjustment = True
        adjustment_reason.append("Brightness outside optimal range")
    
    if std_brightness < 30:
        needs_adjustment = True
        adjustment_reason.append("Low contrast")
    
    if dynamic_range < 100:
        needs_adjustment = True
        adjustment_reason.append("Limited dynamic range")
    
    # Calculate quality score (0-100)
    brightness_score = max(0, 100 - abs(mean_brightness - 128) / 1.28)
    contrast_score = min(100, std_brightness * 2)
    range_score = min(100, dynamic_range / 2)
    
    overall_score = (brightness_score + contrast_score + range_score) / 3
    
    return {
        'needs_adjustment': needs_adjustment,
        'adjustment_reason': adjustment_reason,
        'quality_score': overall_score,
        'metrics': {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'dynamic_range': dynamic_range,
            'min_val': min_val,
            'max_val': max_val
        }
    }


def main():
    # Sidebar: Pixels per micrometer ratio input (must be at the top)
    # Initialize session state for model tracking
    if 'current_model_path' not in st.session_state:
        st.session_state.current_model_path = None
        st.session_state.current_model = None
        st.session_state.current_num_classes = None
    
    # Model selection in sidebar
    st.sidebar.header("Model Selection")
    # Check which models exist
    improved_model_exists = Path(IMPROVED_JOINT_MODEL_PATH).exists()
    aggressive_model_exists = Path(AGGRESSIVE_JOINT_MODEL_PATH).exists()
    balanced_model_exists = Path(BALANCED_JOINT_MODEL_PATH).exists()
    
    if balanced_model_exists:
        model_options = ["Soma Detection (soma_unet.pt)", "Joint Detection (joint_unet.pt)", "Improved Joint (joint_unet_improved.pt)", "Aggressive Joint (joint_unet_aggressive.pt)", "Balanced Joint (joint_unet_balanced.pt)"]
    elif aggressive_model_exists:
        model_options = ["Soma Detection (soma_unet.pt)", "Joint Detection (joint_unet.pt)", "Improved Joint (joint_unet_improved.pt)", "Aggressive Joint (joint_unet_aggressive.pt)"]
    elif improved_model_exists:
        model_options = ["Soma Detection (soma_unet.pt)", "Joint Detection (joint_unet.pt)", "Improved Joint (joint_unet_improved.pt)"]
    else:
        model_options = ["Soma Detection (soma_unet.pt)", "Joint Detection (joint_unet.pt)"]
    
    model_choice = st.sidebar.selectbox(
        "Choose Model",
        model_options,
        help="Soma: Original soma-only model. Joint: Combined soma+dendrite model. Improved: Better trained joint model."
    )
    
    # Place pixels_per_micron input here, right after model selection
    pixels_per_micron = st.sidebar.number_input(
        "Pixels per Micrometer",
        min_value=0.0001,
        max_value=100.0,
        value=10.0,
        step=0.01,
        format="%.4f",
        help="How many pixels are there in one micrometer? Default is 10 px/μm."
    )
    st.sidebar.info("⚠️ Dendrite length accuracy depends on correct calibration and model segmentation. Results may differ from manual tracing.")
    
    # Determine model path
    if "Soma Detection" in model_choice:
        model_path = SOMA_MODEL_PATH
        model_name = "Soma Detection Model"
    elif "Balanced Joint" in model_choice:
        model_path = BALANCED_JOINT_MODEL_PATH
        model_name = "Balanced Joint Model (Anti-Autofluorescence + Dendrites)"
    elif "Aggressive Joint" in model_choice:
        model_path = AGGRESSIVE_JOINT_MODEL_PATH
        model_name = "Aggressive Joint Model (Anti-Autofluorescence)"
    elif "Improved Joint" in model_choice:
        model_path = IMPROVED_JOINT_MODEL_PATH
        model_name = "Improved Joint Model"
    else:
        model_path = JOINT_MODEL_PATH
        model_name = "Joint Detection Model"
    
    # Load model with session state tracking for reliable switching
    if (st.session_state.current_model_path != model_path or 
        st.session_state.current_model is None):
        # Model changed or not loaded, reload it
        st.session_state.current_model_path = model_path
        model, num_classes = load_model_direct(model_path)
        if model is None or num_classes is None:
            st.stop()
        st.session_state.current_model = model
        st.session_state.current_num_classes = num_classes
        st.success(f"✅ Loaded {model_name}")
    else:
        # Use cached model from session state
        model = st.session_state.current_model
        num_classes = st.session_state.current_num_classes
    
    # At this point, num_classes is guaranteed to be 1 or 3
    num_classes = int(num_classes)  # type: ignore
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    
    # Auto-adjustment section
    st.sidebar.subheader("🖼️ Image Enhancement")
    # 1. Uncheck Auto Adjust Brightness/Contrast by default
    # Find the auto_adjust checkbox and set value=False
    auto_adjust = st.sidebar.checkbox(
        "Auto Adjust Brightness/Contrast", value=False, help="Automatically enhance image brightness and contrast before analysis."
    )

    soma_threshold = st.sidebar.slider("Soma Detection Threshold", 0.1, 0.9, 0.7, 0.05, 
                                      help="Higher threshold = fewer false positives")

    # Add option to show raw dendrite probability map for debugging
    # Remove the checkbox and always show the raw dendrite output for joint models
    show_raw_dendrite = False  # Remove the checkbox entirely
    if num_classes == 3:
        dendrite_threshold = st.sidebar.slider("Dendrite Detection Threshold", 0.1, 0.9, 0.3, 0.05,
                                             help="Lower threshold to capture more dendrite details")
        # Simplified artifact filtering controls
        st.sidebar.subheader("Dendrite Post-Processing Filters")
        enable_artifact_filter = st.sidebar.checkbox("Enable Artifact Filtering", value=True)
        enable_gut_filter = st.sidebar.checkbox("Enable Gut Granule Filtering", value=True)
        enable_connection = st.sidebar.checkbox("Enable Dendrite Connection", value=True)

        # Show parameter sliders only if the filter is enabled
        if enable_artifact_filter:
            st.sidebar.subheader("Artifact Filter Parameters")
            min_area = st.sidebar.slider("Min Dendrite Area", 50, 500, 100, 10,
                                       help="Remove very small noise (pixels)")
            max_area = st.sidebar.slider("Max Dendrite Area", 500, 10000, 3000, 100, help="Remove dendrite/glob components larger than this area (px)")
        else:
            min_area = 100
            max_area = 3000
        if enable_gut_filter:
            st.sidebar.subheader("Gut Granule Filter Parameters")
            gut_min_area = st.sidebar.slider("Min Area (px)", 10, 1000, 200, 10, help="Remove objects smaller than this area")
            gut_max_circularity = st.sidebar.slider("Max Circularity", 0.3, 1.0, 0.7, 0.01, help="Remove objects more round than this (lower = less aggressive)")
            gut_min_aspect = st.sidebar.slider("Min Aspect Ratio", 0.05, 1.0, 0.2, 0.01, help="Remove objects fatter than this (lower = less aggressive)")
        else:
            gut_min_area = 200
            gut_max_circularity = 0.7
            gut_min_aspect = 0.2
        if enable_connection:
            st.sidebar.subheader("Dendrite Connection Parameters")
            max_gap = st.sidebar.slider("Max Gap Distance", 50, 150, 80, 5,
                                      help="Maximum distance to connect faded dendrite segments (pixels)")
        else:
            dendrite_threshold = 0.3  # Default value for soma-only models
            gut_min_area = 200
            gut_max_circularity = 0.7
            gut_min_aspect = 0.2
            max_gap = 20
            enable_artifact_filter = True
            enable_gut_filter = True
            enable_connection = True
            connect_dendrites = True
            min_area = 100
            max_area = 3000

    # Add option to show dendrite processing steps for debugging
    show_dendrite_steps = False
    if num_classes == 3:
        # Always show debug checkbox for joint models
        show_dendrite_steps = st.sidebar.checkbox("Show Dendrite Processing Steps (Debug)", value=False,
                                                 help="Display each step of dendrite post-processing for debugging.")
    
    # Add option to show soma detection debug
    show_soma_debug = st.sidebar.checkbox("Show Soma Detection Debug", value=False,
                                         help="Show soma detection probability maps and filtering details")

    # 1. Remove the 'keep N largest' slider from the sidebar (delete num_keep_largest slider)
    # 2. Remove num_keep_largest parameter from filter_dendrite_artifacts and aggressive_gut_granule_filter
    # 3. Restore filtering logic to only use area, circularity, aspect ratio, and always keep very large/long components

    # Processing mode selection - moved to top
    st.header("🧠 DendriteIQ")
    st.markdown("Upload PVD neuron images to detect and count somas and dendrites using trained U-Net models.")
    
    processing_mode = st.radio(
        "Processing Mode",
        ["Single Image", "Batch Processing"],
        horizontal=True,
        help="Single: Process one image with side-by-side view. Batch: Process multiple images."
    )
    
    if processing_mode == "Single Image":
        # Single image processing - original layout
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp']
        )
        
        if uploaded_file is not None:
            # Load and display original image
            image = Image.open(uploaded_file)
            
            # Auto-adjustment if enabled
            if auto_adjust:
                st.subheader("��️ Image Enhancement")
                
                # Analyze image quality
                quality_analysis = analyze_image_quality(image)
                
                col_enhance1, col_enhance2 = st.columns(2)
                
                with col_enhance1:
                    st.write("**Original Image Quality:**")
                    st.write(f"• Quality Score: {quality_analysis['quality_score']:.1f}/100")
                    st.write(f"• Brightness: {quality_analysis['metrics']['mean_brightness']:.1f}")
                    st.write(f"• Contrast: {quality_analysis['metrics']['std_brightness']:.1f}")
                    st.write(f"• Dynamic Range: {quality_analysis['metrics']['dynamic_range']:.1f}")
                    
                    if quality_analysis['needs_adjustment']:
                        st.write("**Adjustment Needed:**")
                        for reason in quality_analysis['adjustment_reason']:
                            st.write(f"• {reason}")
                    else:
                        st.success("✅ Image quality is good!")
                
                with col_enhance2:
                    st.write("**Enhanced Image:**")
                    # Apply auto-adjustment
                    enhanced_image = auto_adjust_brightness_contrast(image)
                    enhanced_pil = Image.fromarray(enhanced_image)
                    st.image(enhanced_pil, use_container_width=True)
                    
                    # Analyze enhanced image
                    enhanced_quality = analyze_image_quality(enhanced_image)
                    st.write(f"• New Quality Score: {enhanced_quality['quality_score']:.1f}/100")
                    st.write(f"• Improvement: +{enhanced_quality['quality_score'] - quality_analysis['quality_score']:.1f}")
                
                # Use enhanced image for processing
                processing_image = enhanced_pil
                st.write("---")
            else:
                processing_image = image
        
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            # Process image
            try:
                # --- Step-by-step processing for dendrites ---
                dendrite_debug = {}
                # Timing: Preprocessing
                start_time = time.time()
                image_tensor, image_rgb = preprocess_image(processing_image)
                preprocessing_time = time.time() - start_time
                # Timing: Prediction
                start_time = time.time()
                soma_mask, dendrite_mask = predict_somas_and_dendrites(model, image_tensor, num_classes)
                prediction_time = time.time() - start_time
                dendrite_debug['raw'] = dendrite_mask.copy() if dendrite_mask is not None else None
                if dendrite_mask is not None:
                    # Step 1: Threshold
                    dendrite_thresh = (dendrite_mask > dendrite_threshold).astype(np.uint8) * 255
                    dendrite_debug['threshold'] = dendrite_thresh.copy()
                    # Step 2: Artifact filtering (optional)
                    if enable_artifact_filter:
                        dendrite_artfilt = filter_dendrite_artifacts(dendrite_thresh, min_area, max_area)
                    else:
                        dendrite_artfilt = dendrite_thresh.copy()
                    dendrite_debug['artifact'] = dendrite_artfilt.copy()
                    # Step 3: Gut granule filtering (optional)
                    if enable_gut_filter:
                        dendrite_gut = aggressive_gut_granule_filter(dendrite_artfilt, min_area, max_area, max_circularity=gut_max_circularity, min_aspect_ratio=gut_min_aspect)
                    else:
                        dendrite_gut = dendrite_artfilt.copy()
                    dendrite_debug['gut'] = dendrite_gut.copy()
                    # Step 4: Dendrite connection (optional)
                    if enable_connection:
                        dendrite_conn = connect_faded_dendrites(dendrite_gut, max_gap)
                    else:
                        dendrite_conn = dendrite_gut.copy()
                    dendrite_debug['connect'] = dendrite_conn.copy()
                else:
                    dendrite_debug = {k: None for k in ['raw','threshold','artifact','gut','connect']}
                # --- End step-by-step ---

                # Use the final mask for overlay
                final_dendrite_mask = dendrite_debug['connect'] if num_classes == 3 else None
                # Timing: Counting
                start_time = time.time()
                soma_count, soma_centers, soma_confidences, soma_binary_mask = count_somas(soma_mask, soma_threshold)
                counting_time = time.time() - start_time
                overlay = create_overlay(image_rgb, soma_mask, soma_centers, soma_confidences, soma_threshold, final_dendrite_mask, dendrite_threshold)
                overlay_pil = Image.fromarray(overlay)
                # Timing: Total
                total_time = preprocessing_time + prediction_time + counting_time
                # Calculate dendrite length if dendrite mask exists
                dendrite_length = 0
                if final_dendrite_mask is not None:
                    dendrite_length = calculate_dendrite_length(final_dendrite_mask)
                
                # --- Reconstruct result dict for downstream code ---
                result = {
                    'overlay': overlay_pil,
                    'soma_count': soma_count,
                    'soma_centers': soma_centers,
                    'soma_confidences': soma_confidences,
                    'dendrite_length': dendrite_length,
                    'soma_mask': soma_mask,
                    'soma_binary_mask': soma_binary_mask,
                    'dendrite_mask': final_dendrite_mask,
                    'raw_dendrite_mask': dendrite_debug['raw'],
                    'timing': {
                        'preprocessing': preprocessing_time,
                        'prediction': prediction_time,
                        'counting': counting_time,
                        'total': total_time
                    },
                    'original': image
                }
                # Store the latest result in session state for canvas access
                st.session_state['last_result'] = result
                # --- Display results ---
                with col2:
                    st.subheader("Detection Results")
                    st.image(overlay_pil, use_container_width=True)
                    
                    # Show dendrite processing steps if enabled
                    if show_dendrite_steps and num_classes == 3:
                        st.subheader("Dendrite Processing Steps (Debug)")
                        step_names = [
                            ("Raw Dendrite Probability Map", 'raw'),
                            ("After Thresholding", 'threshold'),
                            ("After Artifact Filtering", 'artifact'),
                            ("After Gut Granule Filtering", 'gut'),
                            ("After Dendrite Connection", 'connect'),
                        ]
                        for label, key in step_names:
                            img = dendrite_debug.get(key)
                            if img is not None:
                                if key == 'raw':
                                    img_disp = (img * 255).astype(np.uint8)
                                    st.image(img_disp, caption=label, use_container_width=True, channels="GRAY")
                                else:
                                    st.image(img, caption=label, use_container_width=True, channels="GRAY")

                    # Show raw dendrite output only if steps debug is enabled
                    # (no longer always show it)
                    # if num_classes == 3 and result.get('raw_dendrite_mask') is not None:
                    if show_dendrite_steps and num_classes == 3 and result.get('raw_dendrite_mask') is not None:
                        st.subheader("Raw Dendrite Probability Map (Debug)")
                        raw_map = result['raw_dendrite_mask']
                        raw_map_disp = (raw_map * 255).astype(np.uint8)
                        st.image(raw_map_disp, caption="Raw Dendrite Output", use_container_width=True, channels="GRAY")

                    # Show soma detection debug if enabled
                    if show_soma_debug:
                        st.subheader("🔍 Soma Detection Debug")
                        soma_prob_display = (soma_mask * 255).astype(np.uint8)
                        st.image(soma_prob_display, caption="Raw Soma Probability Map", use_container_width=True, channels="GRAY")
                        st.image(soma_binary_mask, caption="Binary Mask After Thresholding", use_container_width=True, channels="GRAY")
                        if result['soma_count'] > 0:
                            st.write(f"**Detected {result['soma_count']} soma(s):**")
                            for i, (center, confidence) in enumerate(zip(result['soma_centers'], result['soma_confidences'])):
                                st.write(f"Soma {i+1}: Center=({center[0]}, {center[1]}), Confidence={confidence:.1%}")
                        else:
                            st.write("No somas detected with current filtering criteria")
                    
                    # Display results with timing and confidence
                    st.metric("Somas Detected", result['soma_count'])
                    
                    # Display dendrite length if available
                    if num_classes == 3 and result['dendrite_length'] > 0:
                        st.metric("Total Dendrite Length", f"{result['dendrite_length'] / pixels_per_micron:.1f} μm")
                    elif num_classes == 3:
                        st.metric("Total Dendrite Length", "0 μm (no dendrites detected)")
                    
                    # Processing time metrics
                    st.subheader("⏱️ Processing Time")
                    col_time1, col_time2, col_time3 = st.columns(3)
                    
                    with col_time1:
                        st.metric("Preprocessing", f"{result['timing']['preprocessing']:.3f}s")
                    with col_time2:
                        st.metric("Prediction", f"{result['timing']['prediction']:.3f}s")
                    with col_time3:
                        st.metric("Counting", f"{result['timing']['counting']:.3f}s")
                    
                    st.metric("Total Time", f"{result['timing']['total']:.3f}s")
                    
                    if result['soma_count'] > 0:
                        st.subheader("🎯 Confidence Analysis")
                        
                        # Average confidence
                        avg_confidence = np.mean(result['soma_confidences'])
                        st.metric("Average Confidence", f"{avg_confidence:.1%}")
                        
                        # Confidence distribution
                        high_conf = sum(1 for c in result['soma_confidences'] if c > 0.8)
                        med_conf = sum(1 for c in result['soma_confidences'] if 0.6 <= c <= 0.8)
                        low_conf = sum(1 for c in result['soma_confidences'] if c < 0.6)
                        
                        col_conf1, col_conf2, col_conf3 = st.columns(3)
                        with col_conf1:
                            st.metric("High (>80%)", high_conf, delta=None)
                        with col_conf2:
                            st.metric("Medium (60-80%)", med_conf, delta=None)
                        with col_conf3:
                            st.metric("Low (<60%)", low_conf, delta=None)
                        
                        # Individual soma confidences
                        st.write("**Individual Soma Confidences:**")
                        for i, confidence in enumerate(result['soma_confidences']):
                            st.write(f"Soma {i+1}: {confidence:.1%}")
                        
                        st.write(f"Detection threshold: {soma_threshold:.2f}")
                        st.write(f"Found {len(result['soma_centers'])} soma centers")
                    else:
                        st.warning("No somas detected with the current threshold. Try lowering the threshold.")
                    
                    # Download button for single image (on the right side)
                    st.subheader("📥 Download Results")
                    overlay_bytes = io.BytesIO()
                    result['overlay'].save(overlay_bytes, format='PNG', quality=95)
                    overlay_bytes.seek(0)
                    st.download_button(
                        label="Download Processed Image (Full Res)",
                        data=overlay_bytes.getvalue(),
                        file_name=f"processed_{uploaded_file.name}",
                        mime="image/png"
                    )

                    # --- Download Masks Section ---
                    st.subheader("📥 Download Masks")
                    # Soma mask (only filtered somas)
                    soma_mask_img = np.zeros(result['soma_binary_mask'].shape, dtype='uint8')
                    binary_mask = (result['soma_mask'] > soma_threshold).astype(np.uint8)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
                    for i in range(1, num_labels):
                        center_x = int(centroids[i][0])
                        center_y = int(centroids[i][1])
                        if (center_x, center_y) in result['soma_centers']:
                            soma_mask_img[labels == i] = 255
                    soma_mask_pil = Image.fromarray(soma_mask_img)
                    soma_bytes = io.BytesIO()
                    soma_mask_pil.save(soma_bytes, format='PNG')
                    soma_bytes.seek(0)
                    st.download_button(
                        label="Download Soma Mask (PNG)",
                        data=soma_bytes.getvalue(),
                        file_name="soma_mask.png",
                        mime="image/png"
                    )
                    # Dendrite mask (only for joint models)
                    if num_classes == 3 and result['dendrite_mask'] is not None:
                        dendrite_mask_img = (result['dendrite_mask'] > 0).astype('uint8') * 255
                        dendrite_mask_pil = Image.fromarray(dendrite_mask_img)
                        dendrite_bytes = io.BytesIO()
                        dendrite_mask_pil.save(dendrite_bytes, format='PNG')
                        dendrite_bytes.seek(0)
                        st.download_button(
                            label="Download Dendrite Mask (PNG)",
                            data=dendrite_bytes.getvalue(),
                            file_name="dendrite_mask.png",
                            mime="image/png"
                        )
                        # Combined RGB mask: soma=red, dendrite=blue, with thinner outline for export
                        combined_mask = np.zeros((*dendrite_mask_img.shape, 3), dtype='uint8')
                        # Draw soma and dendrite masks
                        combined_mask[..., 0] = soma_mask_img  # Red channel
                        combined_mask[..., 2] = dendrite_mask_img  # Blue channel
                        # Draw thin blue outline for dendrite
                        contours, _ = cv2.findContours(dendrite_mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        thin_thickness = 1
                        cv2.drawContours(combined_mask, contours, -1, (0, 0, 255), thin_thickness)
                        combined_mask_pil = Image.fromarray(combined_mask)
                        combined_bytes = io.BytesIO()
                        combined_mask_pil.save(combined_bytes, format='PNG')
                        combined_bytes.seek(0)
                        st.download_button(
                            label="Download Combined Mask (RGB PNG)",
                            data=combined_bytes.getvalue(),
                            file_name="combined_mask.png",
                            mime="image/png"
                        )
                        # --- Download skeletonized masks for Fiji ---
                        from skimage.morphology import skeletonize
                        # Dendrite skeleton
                        dend_skel = skeletonize((dendrite_mask_img > 0).astype('uint8')) * 255
                        dend_skel_pil = Image.fromarray(dend_skel.astype('uint8'))
                        dend_skel_bytes = io.BytesIO()
                        dend_skel_pil.save(dend_skel_bytes, format='PNG')
                        dend_skel_bytes.seek(0)
                        st.download_button(
                            label="Download Dendrite Skeleton Mask (PNG)",
                            data=dend_skel_bytes.getvalue(),
                            file_name="dendrite_skeleton.png",
                            mime="image/png"
                        )
                        # Soma skeleton
                        soma_skel = skeletonize((soma_mask_img > 0).astype('uint8')) * 255
                        soma_skel_pil = Image.fromarray(soma_skel.astype('uint8'))
                        soma_skel_bytes = io.BytesIO()
                        soma_skel_pil.save(soma_skel_bytes, format='PNG')
                        soma_skel_bytes.seek(0)
                        st.download_button(
                            label="Download Soma Skeleton Mask (PNG)",
                            data=soma_skel_bytes.getvalue(),
                            file_name="soma_skeleton.png",
                            mime="image/png"
                        )
                        # --- Download dendrite skeleton overlay ---
                        orig_img = np.array(result['original'])
                        if orig_img.ndim == 2:
                            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
                        overlay_img = orig_img.copy()
                        # Draw skeleton as blue line
                        skel_points = np.column_stack(np.where(dend_skel > 0))
                        for y, x in skel_points:
                            cv2.circle(overlay_img, (x, y), 0, (255, 0, 0), 1)  # Blue
                        overlay_pil = Image.fromarray(overlay_img)
                        overlay_bytes = io.BytesIO()
                        overlay_pil.save(overlay_bytes, format='PNG')
                        overlay_bytes.seek(0)
                        st.download_button(
                            label="Download Dendrite Skeleton Overlay (PNG)",
                            data=overlay_bytes.getvalue(),
                            file_name="dendrite_skeleton_overlay.png",
                            mime="image/png"
                        )

            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    else:
        # Batch processing mode
        st.subheader("📁 Batch Processing")
        st.markdown("Upload multiple images to process them all at once.")
        
        uploaded_files = st.file_uploader(
            "Choose multiple image files", 
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Always set defaults for batch processing
            min_area = 100
            max_area = 3000
            connect_dendrites = True
            dendrite_threshold = 0.3
            max_gap = 20
            enable_artifact_filter = True
            enable_gut_filter = True
            enable_connection = True
            gut_min_area = 200
            gut_max_circularity = 0.7
            gut_min_aspect = 0.2

            st.write(f"📊 Processing {len(uploaded_files)} images...")
            
            # Process all images with progress tracking
            results = []
            filenames = []
            
            # Create progress container
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                progress_text = st.empty()
            
            with status_container:
                status_text = st.empty()
            
            # Process images one by one
            for i, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                progress_text.text(f"Progress: {i+1}/{len(uploaded_files)} ({progress:.1%})")
                status_text.text(f"🔄 Processing: {uploaded_file.name}")
                
                try:
                    image = Image.open(uploaded_file)
                    
                    # Apply auto-adjustment if enabled
                    if auto_adjust:
                        enhanced_image = auto_adjust_brightness_contrast(image)
                        processing_image = Image.fromarray(enhanced_image)
                    else:
                        processing_image = image
                    
                    result = process_single_image(model, processing_image, soma_threshold, num_classes, dendrite_threshold, min_area, max_area, connect_dendrites, max_gap)
                    results.append(result)
                    filenames.append(uploaded_file.name)
                
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                    # Add empty result to maintain indexing
                    results.append(None)
                    filenames.append(uploaded_file.name)
            
            # Clear progress indicators
            progress_bar.empty()
            progress_text.empty()
            status_text.text("✅ Processing complete!")
            
            # Filter out failed results
            successful_results = [(r, f) for r, f in zip(results, filenames) if r is not None]
            if not successful_results:
                st.error("No images were processed successfully.")
                return
            
            results, filenames = zip(*successful_results)
            
            # Display batch results in original layout style
            st.subheader("📈 Batch Results")
            
            # Show each result in the original layout (original left, circled right)
            for i, (result, filename) in enumerate(zip(results, filenames)):
                st.write(f"---")
                st.subheader(f"Image {i+1}: {filename}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Image**")
                    st.image(result['original'], use_container_width=True)
                
                with col2:
                    st.write("**Detection Results**")
                    st.image(result['overlay'], use_container_width=True)
                    
                    # Quick metrics
                    st.metric("Somas Detected", result['soma_count'])
                    if result['soma_count'] > 0:
                        avg_conf = np.mean(result['soma_confidences'])
                        st.metric("Avg Confidence", f"{avg_conf:.1%}")
                    if num_classes == 3:
                        st.metric("Dendrite Length", f"{result['dendrite_length'] / pixels_per_micron:.1f} μm")
                    st.metric("Processing Time", f"{result['timing']['total']:.3f}s")
            
            # Batch summary
            st.subheader("📊 Batch Summary")
            
            total_somas = sum(r['soma_count'] for r in results)
            avg_time = np.mean([r['timing']['total'] for r in results])
            total_time = sum([r['timing']['total'] for r in results])
            
            # Calculate total dendrite length for joint models
            total_dendrite_length = 0
            if num_classes == 3:
                total_dendrite_length = sum(r['dendrite_length'] for r in results)
            
            if num_classes == 3:
                col_sum1, col_sum2, col_sum3, col_sum4, col_sum5 = st.columns(5)
                with col_sum1:
                    st.metric("Total Images", len(results))
                with col_sum2:
                    st.metric("Total Somas", total_somas)
                with col_sum3:
                    # Abbreviate large numbers for display
                    if total_dendrite_length >= 1000000:
                        dendrite_display = f"{total_dendrite_length/1000000:.1f}M μm"
                    elif total_dendrite_length >= 1000:
                        dendrite_display = f"{total_dendrite_length/1000:.1f}K μm"
                    else:
                        dendrite_display = f"{total_dendrite_length / pixels_per_micron:.1f} μm"
                    st.metric("Total Dendrite Length", dendrite_display)
                with col_sum4:
                    st.metric("Avg Time/Image", f"{avg_time:.3f}s")
                with col_sum5:
                    st.metric("Total Time", f"{total_time:.3f}s")
            else:
                col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                with col_sum1:
                    st.metric("Total Images", len(results))
                with col_sum2:
                    st.metric("Total Somas", total_somas)
                with col_sum3:
                    st.metric("Avg Time/Image", f"{avg_time:.3f}s")
                with col_sum4:
                    st.metric("Total Time", f"{total_time:.3f}s")
            
            # Detailed results table
            st.subheader("📋 Detailed Results Table")
            results_data = []
            for i, (result, filename) in enumerate(zip(results, filenames)):
                avg_conf = np.mean(result['soma_confidences']) if result['soma_confidences'] else 0
                row_data = {
                    "Image": filename,
                    "Somas": result['soma_count'],
                    "Avg Confidence": f"{avg_conf:.1%}",
                    "Processing Time": f"{result['timing']['total']:.3f}s"
                }
                if num_classes == 3:
                    row_data["Dendrite Length"] = f"{result['dendrite_length'] / pixels_per_micron:.1f} μm"
                results_data.append(row_data)
            
            st.dataframe(results_data, use_container_width=True)
            
            # Export measurements table
            st.subheader("📊 Export Measurements")
            
            # Create comprehensive measurements data
            comprehensive_data = []
            for i, (result, filename) in enumerate(zip(results, filenames)):
                avg_conf = np.mean(result['soma_confidences']) if result['soma_confidences'] else 0
                max_conf = max(result['soma_confidences']) if result['soma_confidences'] else 0
                min_conf = min(result['soma_confidences']) if result['soma_confidences'] else 0
                
                row_data = {
                    "Image": filename,
                    "Somas": int(result['soma_count']),  # Convert to regular int
                    "Avg Confidence": f"{avg_conf:.1%}",
                    "Max Confidence": f"{max_conf:.1%}",
                    "Min Confidence": f"{min_conf:.1%}",
                    "Processing Time (s)": f"{result['timing']['total']:.3f}",
                    "Preprocessing Time (s)": f"{result['timing']['preprocessing']:.3f}",
                    "Prediction Time (s)": f"{result['timing']['prediction']:.3f}",
                    "Counting Time (s)": f"{result['timing']['counting']:.3f}"
                }
                if num_classes == 3:
                    row_data["Dendrite Length (pixels)"] = int(result['dendrite_length'])  # Convert to regular int
                    row_data["Dendrite Length (μm)"] = f"{result['dendrite_length'] / pixels_per_micron:.1f}"  # Assuming 0.1 μm per pixel
                comprehensive_data.append(row_data)
            
            # Create export options
            col_export1, col_export2, col_export3 = st.columns(3)
            
            with col_export1:
                # Export as CSV
                import pandas as pd
                df = pd.DataFrame(comprehensive_data)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📄 Export as CSV",
                    data=csv_data,
                    file_name="soma_dendrite_measurements.csv",
                    mime="text/csv",
                    type="primary"
                )
            
            with col_export2:
                # Export as Excel
                try:
                    import openpyxl
                    import tempfile
                    import os
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                        with pd.ExcelWriter(tmp_file.name, engine='openpyxl') as writer:
                            df.to_excel(writer, sheet_name='Measurements', index=False)
                            
                            # Add summary statistics sheet
                            summary_data = {
                                'Metric': ['Total Images', 'Total Somas', 'Total Dendrite Length', 'Avg Somas per Image', 'Avg Dendrite Length per Image'],
                                'Value': [
                                    len(results),
                                    total_somas,
                                    total_dendrite_length if num_classes == 3 else 'N/A',
                                    f"{total_somas/len(results):.2f}" if len(results) > 0 else 'N/A',
                                    f"{total_dendrite_length/len(results):.0f} μm" if num_classes == 3 and len(results) > 0 else 'N/A'
                                ]
                            }
                            summary_df = pd.DataFrame(summary_data)
                            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                        
                        # Read the file and provide download
                        with open(tmp_file.name, 'rb') as f:
                            excel_data = f.read()
                        
                        # Clean up temporary file
                        os.unlink(tmp_file.name)
                    
                    st.download_button(
                        label="📊 Export as Excel",
                        data=excel_data,
                        file_name="soma_dendrite_measurements.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
                except (ImportError, ModuleNotFoundError) as e:
                    st.error(f"Excel export requires openpyxl. Install with: pip install openpyxl\nError: {e}")
                except Exception as e:
                    st.error(f"Error creating Excel file: {e}")
            
            with col_export3:
                # Export as JSON
                import json
                json_data = {
                    'summary': {
                        'total_images': int(len(results)),
                        'total_somas': int(total_somas),
                        'total_dendrite_length': int(total_dendrite_length) if num_classes == 3 else None,
                        'avg_somas_per_image': float(total_somas/len(results)) if len(results) > 0 else 0.0,
                        'avg_dendrite_length_per_image': float(total_dendrite_length/len(results)) if num_classes == 3 and len(results) > 0 else None,
                        'total_processing_time': float(sum(float(r['timing']['total']) for r in results)),
                        'avg_processing_time': float(sum(float(r['timing']['total']) for r in results) / len(results)) if len(results) > 0 else 0.0
                    },
                    'measurements': comprehensive_data
                }
                st.download_button(
                    label="📋 Export as JSON",
                    data=json.dumps(json_data, indent=2),
                    file_name="soma_dendrite_measurements.json",
                    mime="application/json",
                    type="primary"
                )
            
            # Bulk download section
            st.subheader("📥 Bulk Download")
            
            # Create download package with progress
            if st.button("🔄 Create Download Package", type="primary"):
                with st.spinner("Creating download package..."):
                    # Create progress for download package creation
                    download_progress = st.progress(0)
                    download_status = st.empty()
                    
                    zip_buffer = io.BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for i, (result, filename) in enumerate(zip(results, filenames)):
                            download_status.text(f"Adding {filename} to package...")
                            
                            # Save processed image
                            overlay_bytes = io.BytesIO()
                            result['overlay'].save(overlay_bytes, format='PNG', quality=95)
                            overlay_bytes.seek(0)
                            zip_file.writestr(f"processed_{filename}", overlay_bytes.getvalue())
                            
                            # Save results summary
                            summary = f"Results for {filename}:\n"
                            summary += f"Somas detected: {result['soma_count']}\n"
                            if num_classes == 3:
                                summary += f"Total dendrite length: {result['dendrite_length'] / pixels_per_micron:.1f} μm\n"
                            summary += f"Average confidence: {np.mean(result['soma_confidences']):.1%}\n"
                            summary += f"Processing time: {result['timing']['total']:.3f}s\n"
                            summary += f"Individual confidences: {[f'{c:.1%}' for c in result['soma_confidences']]}\n"
                            zip_file.writestr(f"summary_{filename}.txt", summary)
                            
                            # Update progress
                            download_progress.progress((i + 1) / len(results))
                    
                    download_progress.empty()
                    download_status.text("✅ Download package ready!")
                    
                    # Provide download button
                    zip_buffer.seek(0)
                    st.download_button(
                        label="📦 Download All Results (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name="soma_detection_results.zip",
                        mime="application/zip",
                        type="primary"
                    )
                    
                    st.success(f"Package contains {len(results)} processed images and summary files!")
    
    # Training performance section
    st.sidebar.header("Training Performance")
    # Read training metrics (static, always version_4)
    metrics_file = Path("lightning_logs/version_4/metrics.csv")
    if metrics_file.exists():
        import pandas as pd
        try:
            df = pd.read_csv(metrics_file)
            st.sidebar.subheader("Training Loss")
            chart_data = df[['step', 'train_loss']].set_index('step')
            st.sidebar.line_chart(chart_data)
            final_loss = df['train_loss'].iloc[-1]
            st.sidebar.metric("Final Loss", f"{final_loss:.4f}")
        except Exception as e:
            st.sidebar.error(f"Error loading metrics: {e}")
    else:
        st.sidebar.info("No training metrics found for this model.")
    
    # Model info
    st.sidebar.header("Model Info")
    st.sidebar.write(f"Model: U-Net (ResNet-18)")
    st.sidebar.write(f"Parameters: 14.3M")
    st.sidebar.write(f"File: {model_path}")
    st.sidebar.write(f"Type: {model_name}")


# --- Manual Dendrite Mask Editing Canvas ---
# Remove the entire block that starts with:
# if st.session_state.get('edit_mask_mode', False):
# Remove the open_canvas_button function definition

if __name__ == "__main__":
    main() 
