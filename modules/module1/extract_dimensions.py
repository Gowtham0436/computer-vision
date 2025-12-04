"""
Script to extract measured dimensions from annotated ROI images
Uses OCR to read the text annotations on the images
"""
import cv2
import re
import os

def extract_dimensions_from_image(image_path):
    """
    Extract width and height from annotated image
    Images have text like "W: XX.XX mm" and "H: XX.XX mm"
    """
    try:
        # Try using pytesseract if available, otherwise return None
        try:
            import pytesseract
        except ImportError:
            print("pytesseract not available. Please install it or manually enter dimensions.")
            return None, None
        
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        
        # Extract text from image
        text = pytesseract.image_to_string(img)
        
        # Look for width pattern: "W: XX.XX mm"
        width_match = re.search(r'W:\s*(\d+\.?\d*)\s*mm', text)
        height_match = re.search(r'H:\s*(\d+\.?\d*)\s*mm', text)
        
        width = float(width_match.group(1)) if width_match else None
        height = float(height_match.group(1)) if height_match else None
        
        return width, height
    except Exception as e:
        print(f"Error extracting from {image_path}: {e}")
        return None, None

if __name__ == "__main__":
    # List of images
    images = [
        "blackbox_roi.JPG",
        "cc_roi.JPG",
        "earbox_roi.JPG",
        "eraser_roi.JPG",
        "gum_roi.JPG",
        "longnotepad_roi.JPG",
        "notepad_roi.JPG",
        "phone_roi.JPG",
        "remote_roi.JPG",
        "rubix_roi.JPG",
        "sanitiser_roi.JPG",
        "wallet_roi.JPG"
    ]
    
    base_path = "static/outputs/"
    
    print("Extracting dimensions from images...")
    for img_name in images:
        img_path = base_path + img_name
        if os.path.exists(img_path):
            width, height = extract_dimensions_from_image(img_path)
            if width and height:
                print(f"{img_name}: W={width:.2f}mm, H={height:.2f}mm")
            else:
                print(f"{img_name}: Could not extract dimensions")
        else:
            print(f"{img_name}: File not found")

