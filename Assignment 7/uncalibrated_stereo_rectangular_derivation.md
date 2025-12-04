# Uncalibrated Stereo Vision for Rectangular Object Size Estimation

## Problem Statement
Given a stereo image pair (left and right) from uncalibrated cameras, estimate the **width and height** of a rectangular object without knowing camera intrinsics or extrinsics.

---

## Mathematical Framework

### 1. Fundamental Matrix from Point Correspondences

#### 1.1 Epipolar Constraint

For any point **P** in 3D space with projections **x₁** in the left image and **x₂** in the right image:

```
x₂ᵀ F x₁ = 0
```

Where:
- **F** is the 3×3 fundamental matrix (rank 2)
- **x₁ = [u₁, v₁, 1]ᵀ** (homogeneous coordinates in left image)
- **x₂ = [u₂, v₂, 1]ᵀ** (homogeneous coordinates in right image)

#### 1.2 Estimating F using 8-Point Algorithm

**Given:** At least 8 point correspondences {(x₁ⁱ, x₂ⁱ) | i = 1, ..., n}

**Step 1: Normalize coordinates (Hartley normalization)**

Transform points so that:
- Centroid is at origin
- Average distance from origin is √2

```
x̃₁ = T₁ x₁
x̃₂ = T₂ x₂
```

Where normalization matrix:
```
T = [s   0   -s·cₓ]     where s = √2 / avg_distance
    [0   s   -s·cᵧ]           (cₓ, cᵧ) = centroid
    [0   0      1 ]
```

**Step 2: Form constraint matrix**

For each correspondence i, construct row:
```
aᵢ = [u₂u₁, u₂v₁, u₂, v₂u₁, v₂v₁, v₂, u₁, v₁, 1]
```

Stack into matrix **A** (n × 9):
```
⎡ a₁ ⎤
⎢ a₂ ⎥
⎢ ⋮  ⎥  f = 0
⎣ aₙ ⎦
```

**Step 3: Solve using SVD**

```
A = U Σ Vᵀ
```

The solution **f** is the last column of **V** (corresponding to smallest singular value).

Reshape **f** into 3×3 matrix **F̃**.

**Step 4: Enforce rank-2 constraint**

Compute SVD of F̃:
```
F̃ = U_F Σ_F V_Fᵀ
```

Set smallest singular value to zero:
```
Σ' = diag(σ₁, σ₂, 0)

F̃ = U_F Σ' V_Fᵀ
```

**Step 5: Denormalize**

```
F = T₂ᵀ F̃ T₁
```

**Result:** Fundamental matrix **F** (3×3, rank 2)

---

### 2. Camera Projection Matrices

Without calibration, we use a canonical configuration:

#### 2.1 Left Camera (World Origin)

```
P₁ = [I₃ | 0] = ⎡1  0  0  0⎤
                ⎢0  1  0  0⎥
                ⎣0  0  1  0⎦
```

#### 2.2 Right Camera from Fundamental Matrix

First, find the right epipole **e₂** (nullspace of Fᵀ):

```
Fᵀ e₂ = 0
```

Solve using SVD of Fᵀ:
```
Fᵀ = U Σ Vᵀ
e₂ = last column of V
```

Normalize: **e₂ = e₂ / e₂[2]**

Form skew-symmetric matrix **[e₂]ₓ**:

```
[e₂]ₓ = ⎡  0     -e₂z    e₂y ⎤
        ⎢ e₂z      0    -e₂x ⎥
        ⎣-e₂y    e₂x      0  ⎦
```

Right camera projection matrix:

```
P₂ = [[e₂]ₓ F | e₂]    (3 × 4 matrix)
```

---

### 3. Triangulation via Direct Linear Transform (DLT)

For a point with correspondences **x₁ = (u₁, v₁, 1)** and **x₂ = (u₂, v₂, 1)**:

#### 3.1 Cross-Product Formulation

From projection equation **x ≅ PX**:

```
x × (PX) = 0
```

This gives 3 equations, but only 2 are independent.

#### 3.2 Linear System

For left image:
```
u₁(p₁³ᵀX) - (p₁¹ᵀX) = 0
v₁(p₁³ᵀX) - (p₁²ᵀX) = 0
```

For right image:
```
u₂(p₂³ᵀX) - (p₂¹ᵀX) = 0
v₂(p₂³ᵀX) - (p₂²ᵀX) = 0
```

Where **pⁱʲᵀ** is the j-th row of camera matrix Pᵢ.

#### 3.3 Matrix Form

Stack into homogeneous system **AX = 0**:

```
⎡u₁p₁³ᵀ - p₁¹ᵀ⎤     ⎡X⎤
⎢v₁p₁³ᵀ - p₁²ᵀ⎥     ⎢Y⎥
⎢u₂p₂³ᵀ - p₂¹ᵀ⎥  ·  ⎢Z⎥  = 0
⎣v₂p₂³ᵀ - p₂²ᵀ⎦     ⎣W⎦
```

**A** is 4×4 matrix.

#### 3.4 Solution

Solve using SVD:
```
A = U Σ Vᵀ
X_homogeneous = last column of V
```

Convert to 3D:
```
X_3D = [X/W, Y/W, Z/W]ᵀ
```

**Result:** 3D point in projective reconstruction (scale ambiguous!)

---

## 4. Scale Recovery for Rectangular Object

### 4.1 The Scale Ambiguity Problem

Projective reconstruction is only defined up to an unknown scale factor **λ**. 

To obtain metric measurements, we need a **reference object with known dimension**.

### 4.2 Reference Measurement

**Given:** A reference object (e.g., ruler, checkerboard square) with known dimension **L_ref**.

**Procedure:**

**Step 1:** Identify two points **A** and **B** on the reference object in both images.

Left image: **A₁ = (u_A1, v_A1)**, **B₁ = (u_B1, v_B1)**

Right image: **A₂ = (u_A2, v_A2)**, **B₂ = (u_B2, v_B2)**

**Step 2:** Triangulate both points:

```
X_A = Triangulate(A₁, A₂, P₁, P₂)
X_B = Triangulate(B₁, B₂, P₁, P₂)
```

**Step 3:** Compute reconstructed distance:

```
L_reconstructed = ||X_B - X_A||

L_reconstructed = √[(X_B - X_A)² + (Y_B - Y_A)² + (Z_B - Z_A)²]
```

**Step 4:** Calculate scale factor:

```
┌────────────────────────────┐
│  λ = L_ref / L_reconstructed │
└────────────────────────────┘
```

This is the **critical equation** for metric reconstruction!

---

## 5. Rectangular Object Dimension Estimation

### 5.1 Rectangle Definition

A rectangle is defined by **4 corner points**: **C₁, C₂, C₃, C₄**

Typical labeling (counter-clockwise):
```
C₄ ────────── C₃
│              │
│              │  Height (H)
│              │
C₁ ────────── C₂
    Width (W)
```

### 5.2 Corner Point Detection

**For each corner Cᵢ:**

Identify in left image: **Cᵢ,₁ = (uᵢ₁, vᵢ₁)**

Identify in right image: **Cᵢ,₂ = (uᵢ₂, vᵢ₂)**

### 5.3 3D Reconstruction of Corners

**For each corner i = 1, 2, 3, 4:**

```
X̂ᵢ = Triangulate(Cᵢ,₁, Cᵢ,₂, P₁, P₂)
```

This gives projective coordinates.

**Apply scale factor:**

```
Xᵢ = λ · X̂ᵢ
```

Where **λ** was computed from reference measurement.

**Result:** Metric 3D coordinates **X₁, X₂, X₃, X₄**

### 5.4 Width Calculation

Width is the distance between bottom corners (or top corners):

```
W = ||X₂ - X₁||

W = √[(X₂ - X₁)² + (Y₂ - Y₁)² + (Z₂ - Z₁)²]
```

Or equivalently (using top edge):
```
W' = ||X₃ - X₄||
```

**For robustness, average both:**
```
W_final = (W + W') / 2
```

### 5.5 Height Calculation

Height is the distance between left edge corners (or right edge):

```
H = ||X₄ - X₁||

H = √[(X₄ - X₁)² + (Y₄ - Y₁)² + (Z₄ - Z₁)²]
```

Or equivalently (using right edge):
```
H' = ||X₃ - X₂||
```

**For robustness, average both:**
```
H_final = (H + H') / 2
```

### 5.6 Additional Measurements

**Diagonal (for verification):**
```
D₁ = ||X₃ - X₁||
D₂ = ||X₄ - X₂||
```

For a perfect rectangle: D₁ ≈ D₂ ≈ √(W² + H²)

**Area:**
```
A = W × H
```

**Perimeter:**
```
P = 2(W + H)
```

---

## 6. Complete Algorithm

### Input
1. Stereo image pair (left and right)
2. At least 8 feature correspondences for F estimation
3. 4 corner points of rectangular object (in both images)
4. Reference measurement: 2 points with known distance L_ref

### Algorithm Steps

```
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: Find Point Correspondences                          │
│   - Extract features (manual or SIFT/SURF/ORB)              │
│   - Match between left and right images                     │
│   - Obtain n ≥ 8 correspondences: {(x₁ⁱ, x₂ⁱ)}             │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: Estimate Fundamental Matrix                         │
│   - Normalize coordinates: T₁, T₂                           │
│   - Form matrix A (n × 9)                                   │
│   - Solve Af = 0 using SVD                                  │
│   - Reshape to 3×3, enforce rank 2                          │
│   - Denormalize: F = T₂ᵀ F̃ T₁                               │
│   - Use RANSAC for outlier rejection                        │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: Compute Projection Matrices                         │
│   - P₁ = [I | 0]                                            │
│   - Solve Fᵀe₂ = 0 for right epipole                       │
│   - P₂ = [[e₂]ₓ F | e₂]                                     │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: Triangulate Reference Points                        │
│   - X_A = Triangulate(A₁, A₂, P₁, P₂)                      │
│   - X_B = Triangulate(B₁, B₂, P₁, P₂)                      │
│   - L_recon = ||X_B - X_A||                                 │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 5: Compute Scale Factor                                │
│   - λ = L_ref / L_recon                                     │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 6: Triangulate Rectangle Corners                       │
│   - For i = 1, 2, 3, 4:                                     │
│       X̂ᵢ = Triangulate(Cᵢ,₁, Cᵢ,₂, P₁, P₂)                 │
│       Xᵢ = λ · X̂ᵢ    (apply scale)                         │
└──────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 7: Compute Rectangle Dimensions                        │
│   - Width:  W = (||X₂-X₁|| + ||X₃-X₄||) / 2               │
│   - Height: H = (||X₄-X₁|| + ||X₃-X₂||) / 2               │
│   - Area:   A = W × H                                       │
└──────────────────────────────────────────────────────────────┘
                           ↓
                    ┌─────────────┐
                    │   OUTPUT    │
                    │  Width (W)  │
                    │  Height (H) │
                    │   Area (A)  │
                    └─────────────┘
```

### Output
- Width: W (in same units as reference)
- Height: H (in same units as reference)
- Area: A = W × H

---

## 7. Numerical Example

### Given Data

**Reference measurement:**
- Known distance between two marks on a ruler: L_ref = 100 mm
- Reconstructed distance after triangulation: L_recon = 0.0873 (projective units)

**Rectangle corner projections:**

Left image coordinates:
- C₁,₁ = (245, 380)
- C₂,₁ = (420, 378)
- C₃,₁ = (418, 290)
- C₄,₁ = (247, 292)

Right image coordinates:
- C₁,₂ = (198, 380)
- C₂,₂ = (373, 378)
- C₃,₂ = (371, 290)
- C₄,₂ = (200, 292)

### Calculation

**Step 1: Scale factor**
```
λ = L_ref / L_recon = 100 / 0.0873 = 1145.48
```

**Step 2: Triangulate corners** (after normalization, F estimation, etc.)

Projective coordinates (example values):
```
X̂₁ = (0.0120, 0.0156, 0.0980)
X̂₂ = (0.0670, 0.0154, 0.0978)
X̂₃ = (0.0668, 0.0068, 0.0976)
X̂₄ = (0.0122, 0.0070, 0.0978)
```

**Step 3: Apply scale**
```
X₁ = λ · X̂₁ = (13.75, 17.87, 112.26)
X₂ = λ · X̂₂ = (76.75, 17.64, 112.03)
X₃ = λ · X̂₃ = (76.52, 7.79, 111.80)
X₄ = λ · X̂₄ = (13.97, 8.02, 112.03)
```

**Step 4: Width**
```
W_bottom = ||X₂ - X₁|| = √[(76.75-13.75)² + (17.64-17.87)² + (112.03-112.26)²]
         = √[63.00² + (-0.23)² + (-0.23)²]
         = √[3969.00 + 0.05 + 0.05]
         = 63.00 mm

W_top = ||X₃ - X₄|| = √[(76.52-13.97)² + (7.79-8.02)² + (111.80-112.03)²]
      = √[62.55² + (-0.23)² + (-0.23)²]
      = 62.56 mm

W = (63.00 + 62.56) / 2 = 62.78 mm
```

**Step 5: Height**
```
H_left = ||X₄ - X₁|| = √[(13.97-13.75)² + (8.02-17.87)² + (112.03-112.26)²]
       = √[0.22² + (-9.85)² + (-0.23)²]
       = √[0.05 + 97.02 + 0.05]
       = 9.85 mm

H_right = ||X₃ - X₂|| = √[(76.52-76.75)² + (7.79-17.64)² + (111.80-112.03)²]
        = √[(-0.23)² + (-9.85)² + (-0.23)²]
        = 9.86 mm

H = (9.85 + 9.86) / 2 = 9.86 mm
```

**Step 6: Area**
```
A = W × H = 62.78 × 9.86 = 619.01 mm²
```

### Final Result
```
┌─────────────────────────────┐
│  Rectangle Dimensions:      │
│  Width:  62.78 mm          │
│  Height:  9.86 mm          │
│  Area:   619.01 mm²        │
└─────────────────────────────┘
```

---

## 8. Summary of Key Equations

### Epipolar Geometry
```
x₂ᵀ F x₁ = 0                           (Fundamental constraint)
```

### Fundamental Matrix (8-point)
```
Af = 0  where A is n×9                 (Linear system)
F = T₂ᵀ F̃ T₁                           (Denormalization)
```

### Camera Matrices
```
P₁ = [I | 0]                           (Left camera)
P₂ = [[e₂]ₓ F | e₂]                    (Right camera)
Fᵀe₂ = 0                               (Right epipole)
```

### Triangulation (DLT)
```
⎡u₁p₁³ᵀ - p₁¹ᵀ⎤
⎢v₁p₁³ᵀ - p₁²ᵀ⎥     
⎢u₂p₂³ᵀ - p₂¹ᵀ⎥  X = 0                (4×4 system)
⎣v₂p₂³ᵀ - p₂²ᵀ⎦

X₃D = [X/W, Y/W, Z/W]ᵀ                 (Homogeneous to 3D)
```

### Scale Recovery
```
λ = L_known / L_reconstructed           (Scale factor)
X_metric = λ · X_projective             (Metric coordinates)
```

### Rectangle Measurements
```
W = (||X₂ - X₁|| + ||X₃ - X₄||) / 2    (Width)
H = (||X₄ - X₁|| + ||X₃ - X₂||) / 2    (Height)
A = W × H                               (Area)
```

### 3D Euclidean Distance
```
d = √[(X₂-X₁)² + (Y₂-Y₁)² + (Z₂-Z₁)²]
```

---

## 9. Advantages and Limitations

### ✓ Advantages
- No camera calibration required
- Works with any stereo camera setup
- Only requires feature correspondences
- Can measure any rectangular object

### ✗ Limitations
- **Requires known reference measurement** (critical!)
- Less accurate than calibrated stereo
- Sensitive to correspondence errors
- Assumes planar rectangular object
- Scale ambiguity must be resolved externally

### Error Sources
1. **Correspondence errors** → Use sub-pixel refinement, RANSAC
2. **F estimation errors** → Use more points, normalization
3. **Triangulation errors** → Larger baseline, minimize reprojection
4. **Reference measurement** → Accurate measurement is crucial
5. **Non-planar rectangle** → Assumption violation

---

## Conclusion

To estimate a rectangular object's size using uncalibrated stereo:

1. Estimate fundamental matrix **F** from correspondences
2. Compute camera matrices **P₁** and **P₂**
3. Triangulate reference points → compute scale **λ**
4. Triangulate rectangle corners → scale by **λ**
5. Compute width and height from corner distances

**The critical step is scale recovery** using a known reference measurement, without which only relative dimensions can be obtained.

