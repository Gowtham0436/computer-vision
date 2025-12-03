# How to Use SAM2 Tracking in Module 5-6

## The Problem

The NPZ file from `IMG_1964` (cube on table) doesn't match your current video scene (person holding cube). **SAM2 tracking needs an NPZ file created from the SAME scene you want to track.**

## Solution: Create NPZ from Current Video Frame

### Method 1: Use "Capture Frame & Create NPZ" Button (Easiest)

1. **Start Camera** in Module 5-6 Problem 2
2. **Select "SAM2 Segmentation"** tracking mode
3. **Position yourself** with the cube in the frame (like in your screenshot)
4. **Click "Capture Frame & Create NPZ"** button
5. The system will:
   - Capture the current video frame
   - Detect the object (cube + person)
   - Create an NPZ file automatically
   - Load it into the tracker
   - Download the NPZ file for future use
6. **Tracking will start automatically!**

### Method 2: Manual Process

1. **Take a screenshot** of your video frame (person holding cube)
2. **Go to Module 3 → Problem 3** (Object Boundary Detection)
3. **Upload the screenshot**
4. **Download the boundary mask** (or use the overlay)
5. **Run**: `python generate_sam2_npz.py <mask_image.png>`
6. **Load the NPZ file** in Module 5-6

## Why This Happens

- SAM2 tracking uses a **pre-computed mask** from a reference image
- The mask must match the **same object/scene** you're tracking
- Different scenes = different masks needed

## Quick Fix for Your Current Situation

Since you're already seeing the person with cube in the video:

1. **Click "Capture Frame & Create NPZ"** button
2. Wait for it to process (detects object, creates NPZ)
3. The NPZ will be automatically loaded
4. Tracking should work immediately!

The new NPZ file will be saved as `sam2_captured_frame.npz` for future use.

