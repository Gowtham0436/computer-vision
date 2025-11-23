# Template Database for Module 2 - Problem 3

## Setup Instructions

1. Place **10 template images** in this directory (`modules/module2/assets/templates/`)
2. Template images should be:
   - **PNG, JPG, or JPEG** format
   - **Smaller than your scene images** (templates are what you're searching for)
   - **From different scenes** (not cropped from the same image you'll test)
   - **Clear and distinct** objects for better detection

## Example Template Files

You can name them anything, for example:
- `object1.jpg`
- `object2.png`
- `template_car.jpg`
- `template_phone.png`
- etc.

## How It Works

1. Upload a scene image in Problem 3
2. System matches each template against the scene using **correlation method**
3. Objects with correlation score > 0.6 are detected
4. Detected regions are automatically blurred
5. Results show count and details of detected objects

## Tips

- Use templates that are clearly visible and distinct
- Templates should be cropped tightly around the object
- Avoid templates with too much background
- Test with different lighting conditions in your scene images

