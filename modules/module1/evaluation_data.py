"""
Module 1: Evaluation Data for ROI Calculation
Stores physical dimensions and measured dimensions for 10 different objects
"""

# Evaluation data structure
# Each entry contains:
# - object_name: Name/description of the object
# - physical_width_mm: Actual physical width in mm (to be filled by user)
# - physical_height_mm: Actual physical height in mm (to be filled by user)
# - measured_width_mm: Measured width from ROI calculation in mm (extracted from image or entered manually)
# - measured_height_mm: Measured height from ROI calculation in mm (extracted from image or entered manually)
# - image_path: Path to the annotated image (relative to static/)
# - notes: Optional notes about the object/scene

EVALUATION_DATA = [
    {
        "object_name": "Blackbox",
        "physical_width_mm": 0.0,  # Enter actual physical width
        "physical_height_mm": 0.0,  # Enter actual physical height
        "measured_width_mm": 0.0,  # Enter measured width from ROI
        "measured_height_mm": 0.0,  # Enter measured height from ROI
        "image_path": "outputs/blackbox_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Credit Card",
        "physical_width_mm": 0.0,
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "outputs/cc_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Earbox",
        "physical_width_mm": 0.0,
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "outputs/earbox_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Eraser",
        "physical_width_mm": 0.0,
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "outputs/eraser_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Gum",
        "physical_width_mm": 0.0,
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "outputs/gum_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Long Notepad",
        "physical_width_mm": 0.0,
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "outputs/longnotepad_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Notepad",
        "physical_width_mm": 0.0,
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "outputs/notepad_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Phone",
        "physical_width_mm": 0.0,
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "outputs/phone_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Remote",
        "physical_width_mm": 0.0,
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "outputs/remote_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Rubix Cube",
        "physical_width_mm": 0.0,
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "outputs/rubix_roi.JPG",
        "notes": ""
    }
]
