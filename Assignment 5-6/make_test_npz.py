import numpy as np

# Choose a resolution (match your webcam if you can, e.g. 480x640)
H, W = 480, 640

# Create 2 dummy binary masks: one square at top-left, one at bottom-right
masks = np.zeros((2, H, W), dtype=np.uint8)

masks[0, 100:200, 100:200] = 1      # first object
masks[1, 300:400, 300:400] = 1      # second object

# Centroids for each mask (x, y)
centroids = np.array([
    [150, 150],   # center of first square
    [350, 350],   # center of second square
], dtype=np.float32)

# Save as NPZ
np.savez("test_sam2_segmentation.npz", masks=masks, centroids=centroids)

print("Wrote test_sam2_segmentation.npz")
