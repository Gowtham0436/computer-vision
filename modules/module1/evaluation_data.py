"""
Module 1: Evaluation Data for ROI Calculation
Stores physical dimensions and measured dimensions for 10 different objects
Static data - no longer auto-updated
"""

# Evaluation data structure
# Each entry contains:
# - object_name: Name/description of the object
# - physical_width_mm: Actual physical width in mm
# - physical_height_mm: Actual physical height in mm
# - measured_width_mm: Measured width from ROI calculation in mm
# - measured_height_mm: Measured height from ROI calculation in mm
# - image_path: Path to the annotated image (relative to static/)
# - notes: Optional notes about the object/scene

EVALUATION_DATA = [
    {
        "object_name": "Blackbox",
        "physical_width_mm": 97,
        "physical_height_mm": 124,
        "measured_width_mm": 99.51,
        "measured_height_mm": 125.15,
        "image_path": "outputs/blackbox_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Credit Card",
        "physical_width_mm": 53,
        "physical_height_mm": 95,
        "measured_width_mm": 55.31,
        "measured_height_mm": 90.34,
        "image_path": "outputs/cc_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Earbox",
        "physical_width_mm": 70,
        "physical_height_mm": 100,
        "measured_width_mm": 74.46,
        "measured_height_mm": 108.76,
        "image_path": "outputs/earbox_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Eraser",
        "physical_width_mm": 50,
        "physical_height_mm": 125,
        "measured_width_mm": 50.69,
        "measured_height_mm": 128.53,
        "image_path": "outputs/eraser_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Gum",
        "physical_width_mm": 52,
        "physical_height_mm": 87,
        "measured_width_mm": 52.7,
        "measured_height_mm": 91.61,
        "image_path": "outputs/gum_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Long Notepad",
        "physical_width_mm": 70,
        "physical_height_mm": 140,
        "measured_width_mm": 71.18,
        "measured_height_mm": 145.69,
        "image_path": "outputs/longnotepad_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Notepad",
        "physical_width_mm": 76,
        "physical_height_mm": 77,
        "measured_width_mm": 77.74,
        "measured_height_mm": 82.43,
        "image_path": "outputs/notepad_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Phone",
        "physical_width_mm": 76,
        "physical_height_mm": 154,
        "measured_width_mm": 78.41,
        "measured_height_mm": 162.18,
        "image_path": "outputs/phone_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Remote",
        "physical_width_mm": 32,
        "physical_height_mm": 160,
        "measured_width_mm": 32.28,
        "measured_height_mm": 164.19,
        "image_path": "outputs/remote_roi.JPG",
        "notes": ""
    },
    {
        "object_name": "Rubix Cube",
        "physical_width_mm": 38,
        "physical_height_mm": 38,
        "measured_width_mm": 39.51,
        "measured_height_mm": 42.15,
        "image_path": "outputs/rubix_roi.JPG",
        "notes": ""
    }
]
