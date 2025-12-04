# Uncalibrated Stereo Vision for Object Size Estimation

## Overview
In uncalibrated stereo, we estimate object sizes without knowing camera intrinsics (focal length, principal point, distortion) or extrinsics (rotation and translation between cameras). We rely on **epipolar geometry** and the **fundamental matrix**.

---

## 1. Fundamental Matrix and Epipolar Geometry

### 1.1 Epipolar Constraint
For a point **p** in 3D space, its projections in two cameras are **x₁** (left) and **x₂** (right). These satisfy:

```
x₂ᵀ F x₁ = 0
```

Where:
- **F** is the 3×3 fundamental matrix
- **x₁ = [u₁, v₁, 1]ᵀ** is the homogeneous coordinate in left image
- **x₂ = [u₂, v₂, 1]ᵀ** is the homogeneous coordinate in right image

### 1.2 Estimating the Fundamental Matrix

Given **n ≥ 8** point correspondences (x₁ⁱ, x₂ⁱ), we solve:

```
x₂ⁱᵀ F x₁ⁱ = 0    for i = 1, 2, ..., n
```

**Method: 8-Point Algorithm**

1. For each correspondence, form the equation:
```
[u₂ u₁, u₂ v₁, u₂, v₂ u₁, v₂ v₁, v₂, u₁, v₁, 1] · f = 0
```

where **f** is a 9-vector containing the entries of F.

2. Stack all equations into matrix **A** (n × 9):
```
A · f = 0
```

3. Solve using SVD: **A = U Σ Vᵀ**
   - F is the last column of V (smallest singular value)
   - Reshape into 3×3 matrix

4. Enforce rank-2 constraint (F must be rank 2):
   - Compute SVD of F: **F = U Σ Vᵀ**
   - Set smallest singular value to 0: **Σ' = diag(σ₁, σ₂, 0)**
   - **F = U Σ' Vᵀ**

### 1.3 Normalization (Hartley's Normalized 8-Point)

For numerical stability, normalize points before computing F:

1. **Translate** points so centroid is at origin
2. **Scale** so average distance from origin is √2

```
T = [s  0  -s·cx]     where s = √2 / avg_distance
    [0  s  -s·cy]           (cx, cy) = centroid
    [0  0    1  ]
```

3. Compute F̂ from normalized points
4. Denormalize: **F = T₂ᵀ F̂ T₁**

---

## 2. Projective Reconstruction

### 2.1 Camera Projection Matrices

Without calibration, we can assume canonical camera matrices:

**Left camera (world origin):**
```
P₁ = [I | 0] = [1 0 0 0]
              [0 1 0 0]
              [0 0 1 0]
```

**Right camera:**
```
P₂ = [[e₂]ₓ F | e₂]
```

Where:
- **e₂** is the right epipole (satisfies **Fᵀ e₂ = 0**)
- **[e₂]ₓ** is the skew-symmetric matrix of e₂

```
[e₂]ₓ = [ 0    -e₂z   e₂y]
        [ e₂z    0   -e₂x]
        [-e₂y   e₂x    0 ]
```

### 2.2 Triangulation (DLT - Direct Linear Transform)

For corresponding points **x₁** and **x₂**, find 3D point **X** that satisfies:

```
x₁ ≅ P₁ X
x₂ ≅ P₂ X
```

**Cross-product formulation:**
```
x₁ × (P₁ X) = 0
x₂ × (P₂ X) = 0
```

Expanding (using only 2 independent equations per image):
```
u₁(p₁³ᵀX) - (p₁¹ᵀX) = 0
v₁(p₁³ᵀX) - (p₁²ᵀX) = 0
u₂(p₂³ᵀX) - (p₂¹ᵀX) = 0
v₂(p₂³ᵀX) - (p₂²ᵀX) = 0
```

Form matrix equation **A X = 0** (4 × 4):
```
[u₁p₁³ᵀ - p₁¹ᵀ]     [X]
[v₁p₁³ᵀ - p₁²ᵀ]  ·  [Y]  = 0
[u₂p₂³ᵀ - p₂¹ᵀ]     [Z]
[v₂p₂³ᵀ - p₂²ᵀ]     [W]
```

Solve using SVD (last column of V).

Convert from homogeneous to 3D: **X_3D = [X/W, Y/W, Z/W]ᵀ**

---

## 3. Metric Reconstruction (Resolving Scale Ambiguity)

### 3.1 The Scale Ambiguity Problem

Projective reconstruction is only up to an unknown projective transformation. We need at least one **known measurement** to resolve scale.

**Given:** A reference object with known size **L_known**

### 3.2 Computing Scale Factor

1. **Triangulate two points** on the reference object:
   - Get 3D coordinates: **P₁ = (X₁, Y₁, Z₁)** and **P₂ = (X₂, Y₂, Z₂)**

2. **Compute Euclidean distance** in reconstructed space:
```
L_reconstructed = √[(X₂-X₁)² + (Y₂-Y₁)² + (Z₂-Z₁)²]
```

3. **Calculate scale factor:**
```
λ = L_known / L_reconstructed
```

4. **Scale all 3D points:**
```
X_metric = λ · X_reconstructed
```

---

## 4. Object Size Estimation Procedure

### Complete Algorithm

#### **Input:**
- Stereo image pair (left and right)
- At least 8 point correspondences for fundamental matrix
- One known reference measurement

#### **Step 1: Find Point Correspondences**
- Manual selection or feature matching (SIFT, SURF, ORB)
- At least 8 pairs for fundamental matrix
- More pairs for robustness (use RANSAC)

#### **Step 2: Estimate Fundamental Matrix**
```python
# Using RANSAC to handle outliers
F, mask = cv2.findFundamentalMat(points_left, points_right, 
                                  cv2.FM_RANSAC, 1.0, 0.99)
```

Mathematical solution (normalized 8-point):
```
1. Normalize points: x₁' = T₁ x₁, x₂' = T₂ x₂
2. Form matrix A from correspondences
3. Solve A·f = 0 using SVD
4. Reshape f to 3×3 matrix F'
5. Enforce rank 2: F' = U diag(σ₁, σ₂, 0) Vᵀ
6. Denormalize: F = T₂ᵀ F' T₁
```

#### **Step 3: Compute Camera Matrices**
```
# Left camera (canonical)
P₁ = [I₃ | 0]

# Right camera (from F and epipole)
e₂ = nullspace(Fᵀ)  # Solve Fᵀe₂ = 0
P₂ = [[e₂]ₓ F | e₂]
```

#### **Step 4: Triangulate Reference Points**

For known reference distance between points **A** and **B**:

```
# Triangulate point A
X_A = triangulate(x₁_A, x₂_A, P₁, P₂)

# Triangulate point B  
X_B = triangulate(x₁_B, x₂_B, P₁, P₂)

# Compute reconstructed distance
d_reconstructed = ||X_B - X_A||
```

#### **Step 5: Compute Scale Factor**
```
λ = d_known / d_reconstructed
```

#### **Step 6: Triangulate Object Points**

For object of interest with corner points **C₁, C₂, ..., Cₙ**:

```
For each corner i:
    X_i = λ · triangulate(x₁_i, x₂_i, P₁, P₂)
```

#### **Step 7: Compute Object Dimensions**

**For rectangular object:**
```
Width = ||X₂ - X₁||
Height = ||X₃ - X₁||  
Area = Width × Height
```

**For circular object:**
```
Diameter = ||X₂ - X₁||
Radius = Diameter / 2
Area = π × Radius²
```

**For arbitrary polygon:**
```
Perimeter = Σ ||X_{i+1} - X_i||
Area = (Shoelace formula in 3D plane)
```

---

## 5. Mathematical Summary

### 5.1 Key Equations

**Epipolar Constraint:**
```
x₂ᵀ F x₁ = 0
```

**Triangulation (for point X):**
```
X = argmin ||x₁ - P₁X||² + ||x₂ - P₂X||²
```

**Scale Recovery:**
```
λ = Σᵢ (Lᵢ_known / Lᵢ_reconstructed) / n
```
where n = number of known measurements

**Final Metric Coordinates:**
```
X_metric = λ · X_projective
```

### 5.2 Distance Computation

For two 3D points **A = (X_A, Y_A, Z_A)** and **B = (X_B, Y_B, Z_B)**:

```
d = √[(X_B - X_A)² + (Y_B - Y_A)² + (Z_B - Z_A)²]
```

In matrix form:
```
d = ||B - A|| = √[(B - A)ᵀ(B - A)]
```

---

## 6. Error Analysis and Considerations

### 6.1 Sources of Error

1. **Correspondence Error:** Inaccurate point matching
   - Use sub-pixel refinement
   - Employ RANSAC for outlier rejection

2. **Fundamental Matrix Accuracy:**
   - Depends on point distribution
   - More points = better conditioning
   - Hartley normalization reduces numerical errors

3. **Triangulation Error:**
   - Increases with distance from cameras
   - Better with larger baseline
   - Minimize reprojection error

4. **Scale Ambiguity:**
   - Accumulates from reference measurement
   - Use multiple reference measurements
   - Average for robustness

### 6.2 Improving Accuracy

**Bundle Adjustment:**
```
min Σᵢⱼ ||xᵢⱼ - π(Pⱼ, Xᵢ)||²
```
Optimize over all 3D points X and cameras P.

**Multiple Reference Measurements:**
```
λ_optimal = (1/n) Σᵢ (dᵢ_known / dᵢ_reconstructed)
```

**Weighted Averaging:**
```
λ = Σᵢ wᵢ λᵢ / Σᵢ wᵢ
```
where weights wᵢ based on measurement confidence.

---

## 7. Practical Implementation

### Algorithm Pseudocode

```python
def estimate_size_uncalibrated(img_left, img_right, object_points, 
                                reference_points, reference_size):
    """
    Estimate object size using uncalibrated stereo.
    
    Args:
        img_left, img_right: Stereo image pair
        object_points: List of [(x1, y1), (x2, y2)] for object
        reference_points: List of [(x1, y1), (x2, y2)] for reference
        reference_size: Known size of reference object
    
    Returns:
        object_size: Estimated size in same units as reference
    """
    
    # Step 1: Extract and match features
    keypoints_left, keypoints_right = match_features(img_left, img_right)
    
    # Step 2: Estimate fundamental matrix (RANSAC)
    F, mask = estimate_fundamental_matrix_ransac(
        keypoints_left, keypoints_right, 
        threshold=1.0, confidence=0.999
    )
    
    # Step 3: Compute projection matrices
    P1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    
    # Find right epipole
    U, S, Vt = svd(F.T)
    e2 = Vt[-1]
    e2 = e2 / e2[2]  # Normalize
    
    # Skew-symmetric matrix
    e2_cross = np.array([[0, -e2[2], e2[1]],
                         [e2[2], 0, -e2[0]],
                         [-e2[1], e2[0], 0]])
    
    P2 = np.hstack([e2_cross @ F, e2.reshape(-1, 1)])
    
    # Step 4: Triangulate reference points
    ref_3d = []
    for (x1, y1), (x2, y2) in reference_points:
        X = triangulate_point(x1, y1, x2, y2, P1, P2)
        ref_3d.append(X)
    
    # Step 5: Compute scale
    reconstructed_dist = euclidean_distance(ref_3d[0], ref_3d[1])
    scale = reference_size / reconstructed_dist
    
    # Step 6: Triangulate object points
    obj_3d = []
    for (x1, y1), (x2, y2) in object_points:
        X = triangulate_point(x1, y1, x2, y2, P1, P2)
        X_scaled = scale * X
        obj_3d.append(X_scaled)
    
    # Step 7: Compute object size
    object_size = euclidean_distance(obj_3d[0], obj_3d[1])
    
    return object_size, obj_3d


def triangulate_point(u1, v1, u2, v2, P1, P2):
    """DLT triangulation."""
    A = np.array([
        u1 * P1[2] - P1[0],
        v1 * P1[2] - P1[1],
        u2 * P2[2] - P2[0],
        v2 * P2[2] - P2[1]
    ])
    
    U, S, Vt = svd(A)
    X_homogeneous = Vt[-1]
    X = X_homogeneous[:3] / X_homogeneous[3]
    
    return X


def euclidean_distance(p1, p2):
    """Compute 3D Euclidean distance."""
    return np.sqrt(np.sum((np.array(p2) - np.array(p1))**2))
```

---

## 8. Comparison: Calibrated vs Uncalibrated

| Aspect | Calibrated | Uncalibrated |
|--------|-----------|--------------|
| **Camera parameters** | Known (K, dist, R, T) | Unknown |
| **Calibration needed** | Yes (chessboard, etc.) | No |
| **3D reconstruction** | Metric (true scale) | Projective (up to scale) |
| **Scale recovery** | Automatic | Requires reference |
| **Accuracy** | Higher | Lower (more error sources) |
| **Flexibility** | Less (fixed calibration) | More (any camera) |
| **Use case** | Precision measurement | Quick estimates |

---

## 9. Conclusion

**Uncalibrated stereo estimation procedure:**

1. ✓ Find correspondences between images
2. ✓ Estimate fundamental matrix F
3. ✓ Compute camera matrices P₁ and P₂
4. ✓ Triangulate reference points
5. ✓ Calculate scale factor λ
6. ✓ Triangulate object points
7. ✓ Compute scaled object dimensions

**Key advantages:**
- No prior calibration required
- Works with any stereo pair
- Simple to implement

**Key limitations:**
- Requires known reference measurement
- Less accurate than calibrated stereo
- Sensitive to point correspondence errors

**Best practices:**
- Use RANSAC for robust F estimation
- Multiple reference measurements for better scale
- Sub-pixel corner/feature detection
- Bundle adjustment for refinement

