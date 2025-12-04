"""
Module 1: Evaluation Data for ROI Calculation
Stores physical dimensions and measured dimensions for 10 different objects
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
        "object_name": "Object 1",
        "physical_width_mm": 100.0,
        "physical_height_mm": 50.0,
        "measured_width_mm": 98.5,
        "measured_height_mm": 49.2,
        "image_path": "outputs/roi_20251203_112835.png",
        "notes": "Sample object - update with your data"
    },
    {
        "object_name": "Object 2",
        "physical_width_mm": 120.0,
        "physical_height_mm": 80.0,
        "measured_width_mm": 118.3,
        "measured_height_mm": 79.1,
        "image_path": "outputs/roi_20251203_124516.png",
        "notes": "Sample object - update with your data"
    },
    {
        "object_name": "Object 3",
        "physical_width_mm": 0.0,  # Placeholder - update with actual data
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "",
        "notes": "Update with your data"
    },
    {
        "object_name": "Object 4",
        "physical_width_mm": 0.0,
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "",
        "notes": "Update with your data"
    },
    {
        "object_name": "Object 5",
        "physical_width_mm": 0.0,
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "",
        "notes": "Update with your data"
    },
    {
        "object_name": "Object 6",
        "physical_width_mm": 0.0,
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "",
        "notes": "Update with your data"
    },
    {
        "object_name": "Object 7",
        "physical_width_mm": 0.0,
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "",
        "notes": "Update with your data"
    },
    {
        "object_name": "Object 8",
        "physical_width_mm": 0.0,
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "",
        "notes": "Update with your data"
    },
    {
        "object_name": "Object 9",
        "physical_width_mm": 0.0,
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "",
        "notes": "Update with your data"
    },
    {
        "object_name": "Object 10",
        "physical_width_mm": 0.0,
        "physical_height_mm": 0.0,
        "measured_width_mm": 0.0,
        "measured_height_mm": 0.0,
        "image_path": "",
        "notes": "Update with your data"
    }
]

