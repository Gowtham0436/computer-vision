import numpy as np
import cv2
import os

def create_stereo_pairs(output_dir="stereo_pairs", num_pairs=3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Camera Intrinsics (simulating a typical camera)
    img_w, img_h = 1280, 720
    focal_length = 1000
    cx, cy = img_w / 2, img_h / 2
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros(5) # No distortion for simplicity

    # Stereo configuration (Right camera relative to Left)
    baseline = 100.0 # mm
    R_stereo = np.eye(3) # No rotation between cameras
    T_stereo = np.array([[-baseline], [0], [0]], dtype=np.float32) # Right camera is shifted along -X relative to Left? 
    # Convention: T is translation of world origin to camera frame, or camera 2 wrt camera 1?
    # OpenCV stereoCalibrate finds T such that P2 = R*P1 + T. 
    # Usually we define the second camera's position relative to the first.
    # If Left is at origin, Right is at (baseline, 0, 0).
    # The coordinate transform from Left to Right is X_r = R * X_l + T.
    # So if X_l = (x,y,z), X_r = (x-b, y, z). So T = [-b, 0, 0].
    
    # Chessboard settings (user specified)
    pattern_size = (9, 6) # Inner corners
    square_size = 20.0 # mm
    
    # 3D object points of the chessboard (Z=0 plane)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    for i in range(num_pairs):
        # Position the chessboard to FILL 100% OF THE SCREEN
        # Board dimensions: (9+1)*20 = 200mm width, (6+1)*20 = 140mm height
        board_width_mm = (pattern_size[0] + 1) * square_size
        board_height_mm = (pattern_size[1] + 1) * square_size
        
        # Calculate distance to make board fill exactly 100% of screen
        # Overfill slightly (1.01) so edges are at/beyond screen boundaries
        z_width = (board_width_mm * focal_length) / (img_w * 1.01)  # Overfill width
        z_height = (board_height_mm * focal_length) / (img_h * 1.01)  # Overfill height
        
        # Use the larger distance to ensure board fills entire frame
        base_z = max(z_width, z_height)
        z_dist = base_z * np.random.uniform(0.98, 1.0)  # Slight closer for full coverage
        
        # Keep board perfectly centered with minimal offset
        t_vec = np.array([
            [np.random.uniform(-3, 3)], 
            [np.random.uniform(-3, 3)], 
            [z_dist]
        ], dtype=np.float32)
        
        # Very minimal rotation to keep coverage
        r_vec = np.random.uniform(-0.02, 0.02, (3, 1)).astype(np.float32)
        R_board, _ = cv2.Rodrigues(r_vec)

        # --------------------------
        # Project to Left Camera
        # --------------------------
        # The board is at R_board, t_vec relative to Left Camera
        img_points_left, _ = cv2.projectPoints(objp, r_vec, t_vec, K, dist_coeffs)
        
        # Draw Left Image
        img_left = np.full((img_h, img_w), 255, dtype=np.uint8) # White background
        # We can't easily "draw" a chessboard with perspective warping just from points efficiently without texture mapping.
        # Instead, let's just draw the corners and lines or filling polygons.
        # Better: use fillConvexPoly for squares.
        
        def draw_chessboard(img, img_points, pattern_size):
            points = img_points.reshape(-1, 2)
            # Draw black squares
            # Grid is pattern_size[0] x pattern_size[1] internal corners.
            # This means there are (rows+1) * (cols+1) squares roughly? No.
            # Let's just draw the connected lines to make it look like a grid.
            # Or better, just draw circles at corners which is enough for detection sometimes, 
            # but findChessboardCorners needs quad topology.
            
            # Actually, rendering a proper chessboard is non-trivial with just points.
            # Let's approximate by drawing black quadrilaterals between appropriate points.
            # The objp definition: (0,0), (1,0)... (8,0), (0,1)...
            # Let's rely on OpenCV drawing logic or simple lines. 
            # But findChessboardCorners needs strong contrast squares.
            
            # To properly render, we should project the 4 corners of each black square.
            # The objp we have are the *internal* corners.
            # We need to expand the board slightly to render the squares.
            pass

        # Create high-resolution chessboard optimized for full-screen display
        # Calculate square pixel size to fill the screen
        sq_px_width = img_w / (pattern_size[0] + 1)  # Divide screen by number of squares
        sq_px_height = img_h / (pattern_size[1] + 1)
        sq_px = int(max(sq_px_width, sq_px_height))  # Use larger to ensure coverage
        
        # Create larger source board for warping without white borders
        margin_squares = 2  # Extra squares for perspective warping
        board_w_px = (pattern_size[0] + 1 + margin_squares * 2) * sq_px
        board_h_px = (pattern_size[1] + 1 + margin_squares * 2) * sq_px
        src_board = np.full((board_h_px, board_w_px), 255, dtype=np.uint8)
        
        # Draw extended chessboard pattern
        for r in range(pattern_size[1] + 1 + margin_squares * 2):
            for c in range(pattern_size[0] + 1 + margin_squares * 2):
                if (r + c) % 2 == 1:
                    top_left = (c * sq_px, r * sq_px)
                    bottom_right = ((c + 1) * sq_px, (r + 1) * sq_px)
                    cv2.rectangle(src_board, top_left, bottom_right, 0, -1)
        
        # Source points for homography (corresponding to objp, but in pixel coordinates of src_board)
        # objp (0,0) corresponds to the first internal corner.
        # Account for margin squares added for full coverage
        src_pts = []
        for r in range(pattern_size[1]):
            for c in range(pattern_size[0]):
                # Add margin_squares and 1 for the first square edge
                src_pts.append([(c + 1 + margin_squares) * sq_px, (r + 1 + margin_squares) * sq_px])
        src_pts = np.array(src_pts, dtype=np.float32)
        
        # Compute Homography for Left
        H_left, _ = cv2.findHomography(src_pts, img_points_left)
        warped_left = cv2.warpPerspective(src_board, H_left, (img_w, img_h), 
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=255)
        
        # --------------------------
        # Project to Right Camera
        # --------------------------
        # Right camera pose:
        # P_right = R_stereo * P_left + T_stereo (This converts Left coords to Right coords)
        # Board position in Right Frame:
        # P_board_right = R_stereo * (R_board * P_model + t_vec) + T_stereo
        #               = (R_stereo * R_board) * P_model + (R_stereo * t_vec + T_stereo)
        
        R_total = R_stereo @ R_board
        t_total = R_stereo @ t_vec + T_stereo
        
        # Convert back to Rodrigues for projectPoints
        r_vec_right, _ = cv2.Rodrigues(R_total)
        
        img_points_right, _ = cv2.projectPoints(objp, r_vec_right, t_total, K, dist_coeffs)
        
        # Compute Homography for Right
        H_right, _ = cv2.findHomography(src_pts, img_points_right)
        warped_right = cv2.warpPerspective(src_board, H_right, (img_w, img_h),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=255)

        # Save
        cv2.imwrite(f"{output_dir}/left_{i:02d}.jpg", warped_left)
        cv2.imwrite(f"{output_dir}/right_{i:02d}.jpg", warped_right)
        
    print(f"Generated {num_pairs} stereo pairs in '{output_dir}'")

if __name__ == "__main__":
    create_stereo_pairs()
