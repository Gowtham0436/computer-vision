// Assignment 3 Core Logic - Client-side OpenCV.js implementation
// Based on Assignment 3/webapp/app.js

// ---------- ArUco API Detection ----------
function getArucoAPI(cv) {
  const enumNames = ['DICT_6X6_250', 'DICT_ARUCO_6X6_250', 'aruco_DICT_6X6_250', 'DICT_4X4_50'];
  let dictId = null;
  for (const name of enumNames) {
    if (cv[name] !== undefined) {
      dictId = cv[name];
      break;
    }
  }

  if (dictId == null) return null;

  const dictionaryFactories = [
    () => (typeof cv.getPredefinedDictionary === 'function' ? cv.getPredefinedDictionary(dictId) : null),
    () => {
      if (typeof cv.aruco_Dictionary === 'function' && typeof cv.aruco_Dictionary.getPredefinedDictionary === 'function') {
        return cv.aruco_Dictionary.getPredefinedDictionary(dictId);
      }
      return null;
    },
    () => (typeof cv.Dictionary === 'function' ? new cv.Dictionary(dictId) : null),
    () => {
      if (typeof cv.aruco_Dictionary === 'function') {
        const d = new cv.aruco_Dictionary();
        if (typeof d.fromPredefined === 'function') {
          d.fromPredefined(dictId);
          return d;
        }
      }
      return null;
    }
  ];

  let dictionary = null;
  for (const create of dictionaryFactories) {
    if (!create) continue;
    try {
      const candidate = create();
      if (candidate) {
        dictionary = candidate;
        break;
      }
    } catch (err) {}
  }
  if (!dictionary) return null;

  let draw = null;
  const drawCandidates = [
    typeof cv.drawDetectedMarkers === 'function' ? (img, corners, ids) => cv.drawDetectedMarkers(img, corners, ids) : null,
    typeof cv.aruco_DrawDetectedMarkers === 'function' ? (img, corners, ids) => cv.aruco_DrawDetectedMarkers(img, corners, ids) : null
  ];
  for (const fn of drawCandidates) {
    if (fn) {
      draw = fn;
      break;
    }
  }

  // Try new API first (ArucoDetector)
  if (typeof cv.aruco_ArucoDetector === 'function' && typeof cv.aruco_DetectorParameters === 'function' && typeof cv.aruco_RefineParameters === 'function') {
    try {
      const params = new cv.aruco_DetectorParameters();
      const refine = new cv.aruco_RefineParameters(10, 3, true);
      const detector = new cv.aruco_ArucoDetector(dictionary, params, refine);
      const detect = (img, dict, corners, ids, unusedParams) => {
        const rejected = new cv.MatVector();
        try {
          detector.detectMarkers(img, corners, ids, rejected);
        } finally {
          rejected.delete();
        }
      };
      return { mode: 'detector', dictionary, params, refine, detector, detect, draw };
    } catch (err) {
      console.warn('ArucoDetector init failed, falling back to legacy API', err);
    }
  }

  // Try legacy API
  const paramFactories = [
    () => (typeof cv.DetectorParameters === 'function' ? new cv.DetectorParameters() : null),
    () => (typeof cv.aruco_DetectorParameters === 'function' ? new cv.aruco_DetectorParameters() : null)
  ];

  let params = null;
  for (const create of paramFactories) {
    if (!create) continue;
    try {
      const candidate = create();
      if (candidate) {
        params = candidate;
        break;
      }
    } catch (err) {}
  }

  const detectCandidates = [
    typeof cv.detectMarkers === 'function' ? (img, dict, corners, ids, detectorParams) => cv.detectMarkers(img, dict, corners, ids, detectorParams) : null,
    typeof cv.aruco_DetectMarkers === 'function' ? (img, dict, corners, ids, detectorParams) => cv.aruco_DetectMarkers(img, dict, corners, ids, detectorParams) : null
  ];

  let detect = null;
  for (const fn of detectCandidates) {
    if (fn) {
      detect = fn;
      break;
    }
  }

  if (!params || !detect) return null;
  return { mode: 'legacy', dictionary, params, detect, draw };
}

// ---------- Helper Functions ----------
function toGray(src) {
  const g = new cv.Mat();
  cv.cvtColor(src, g, cv.COLOR_RGBA2GRAY);
  return g;
}

function percentileFrom8U(mat8, p) {
  const hist = new Array(256).fill(0);
  const d = mat8.data;
  for (let i = 0; i < d.length; i++) hist[d[i]]++;
  const target = (p / 100) * d.length;
  let cum = 0;
  for (let v = 0; v < 256; v++) { cum += hist[v]; if (cum >= target) return v; }
  return 255;
}

// ---------- Task 1: Gradient and LoG ----------
function gradMagnitudeMat(src) {
  const gray = toGray(src);
  const dx = new cv.Mat(), dy = new cv.Mat();
  const mag = new cv.Mat(), mag8 = new cv.Mat();
  cv.GaussianBlur(gray, gray, new cv.Size(3,3), 0.8, 0.8);
  cv.Sobel(gray, dx, cv.CV_32F, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT);
  cv.Sobel(gray, dy, cv.CV_32F, 0, 1, 3, 1, 0, cv.BORDER_DEFAULT);
  cv.magnitude(dx, dy, mag);
  cv.normalize(mag, mag, 0, 255, cv.NORM_MINMAX);
  mag.convertTo(mag8, cv.CV_8U);
  gray.delete(); dx.delete(); dy.delete(); mag.delete();
  return mag8;
}

function gradAngleMat(src) {
  const gray = toGray(src);
  const dx = new cv.Mat(), dy = new cv.Mat();
  const mag = new cv.Mat(), ang = new cv.Mat();
  const mag8 = new cv.Mat(), angH = new cv.Mat();
  cv.GaussianBlur(gray, gray, new cv.Size(3,3), 0.8, 0.8);
  cv.Sobel(gray, dx, cv.CV_32F, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT);
  cv.Sobel(gray, dy, cv.CV_32F, 0, 1, 3, 1, 0, cv.BORDER_DEFAULT);
  cv.cartToPolar(dx, dy, mag, ang, true);
  cv.normalize(mag, mag, 0, 255, cv.NORM_MINMAX);
  mag.convertTo(mag8, cv.CV_8U);
  const mask = new cv.Mat(); cv.threshold(mag8, mask, 20, 255, cv.THRESH_BINARY);
  ang.convertTo(angH, cv.CV_8U, 0.5);
  const hueMax = new cv.Mat(angH.rows, angH.cols, cv.CV_8U); hueMax.setTo(new cv.Scalar(179));
  cv.min(angH, hueMax, angH); hueMax.delete();
  const sat = new cv.Mat(gray.rows, gray.cols, cv.CV_8U); sat.setTo(new cv.Scalar(255));
  const val = new cv.Mat(); cv.bitwise_and(mag8, mask, val);
  const mv = new cv.MatVector(); mv.push_back(angH); mv.push_back(sat); mv.push_back(val);
  const hsv = new cv.Mat(); cv.merge(mv, hsv);
  const rgb = new cv.Mat(); cv.cvtColor(hsv, rgb, cv.COLOR_HSV2RGB);
  gray.delete(); dx.delete(); dy.delete();
  mag.delete(); ang.delete(); mag8.delete(); angH.delete();
  sat.delete(); val.delete(); mask.delete(); mv.delete(); hsv.delete();
  return rgb;
}

function logMat(src) {
  const gray = toGray(src);
  const blur = new cv.Mat();
  const lap = new cv.Mat(), lap8 = new cv.Mat();
  cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 1.0, 1.0, cv.BORDER_DEFAULT);
  cv.Laplacian(blur, lap, cv.CV_32F, 3, 1, 0, cv.BORDER_DEFAULT);
  cv.convertScaleAbs(lap, lap8);
  gray.delete(); blur.delete(); lap.delete();
  return lap8;
}

// ---------- Task 2: Edge Detection (NMS + Hysteresis) ----------
function edgeSimple(src, returnBinary = true, edgeLow = 15, edgeHigh = 40, edgeAuto = true) {
  const gray = toGray(src);
  cv.GaussianBlur(gray, gray, new cv.Size(3,3), 0.8, 0.8);

  const dx = new cv.Mat(), dy = new cv.Mat(), mag = new cv.Mat(), ang = new cv.Mat();
  cv.Sobel(gray, dx, cv.CV_32F, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT);
  cv.Sobel(gray, dy, cv.CV_32F, 0, 1, 3, 1, 0, cv.BORDER_DEFAULT);
  cv.cartToPolar(dx, dy, mag, ang, true);

  const rows = mag.rows, cols = mag.cols;
  const nms = cv.Mat.zeros(rows, cols, cv.CV_32F);
  const M = mag.data32F, A = ang.data32F, N = nms.data32F;

  for (let y = 1; y < rows - 1; y++) {
    const r = y * cols;
    for (let x = 1; x < cols - 1; x++) {
      const i = r + x;
      let a = A[i];
      if (a < 0) a += 180;
      if (a >= 180) a -= 180;

      let o1 = 0, o2 = 0;
      if (a < 22.5 || a >= 157.5)      { o1 = -1;           o2 = +1; }
      else if (a < 67.5)               { o1 = -cols - 1;    o2 = +cols + 1; }
      else if (a < 112.5)              { o1 = -cols;        o2 = +cols; }
      else                             { o1 = -cols + 1;    o2 = +cols - 1; }

      const m = M[i];
      if (m >= M[i + o1] && m >= M[i + o2]) N[i] = m;
    }
  }

  const nms8 = new cv.Mat();
  cv.normalize(nms, nms, 0, 255, cv.NORM_MINMAX);
  nms.convertTo(nms8, cv.CV_8U);

  let high = edgeHigh, low = edgeLow;
  if (edgeAuto) {
    high = percentileFrom8U(nms8, 85);
    low  = Math.max(5, Math.round(0.4 * high));
  }

  const strong = new cv.Mat(), weak = new cv.Mat(), weakOnly = new cv.Mat();
  cv.threshold(nms8, strong, high, 255, cv.THRESH_BINARY);
  cv.threshold(nms8, weak,   low,  255, cv.THRESH_BINARY);
  cv.subtract(weak, strong, weakOnly);

  const connected = strong.clone();
  const kernel = cv.Mat.ones(3,3,cv.CV_8U);
  const dil = new cv.Mat(), add = new cv.Mat();

  for (let it = 0; it < 12; it++) {
    cv.dilate(connected, dil, kernel);
    cv.bitwise_and(dil, weakOnly, add);
    const nz = cv.countNonZero(add);
    if (nz === 0) break;
    cv.bitwise_or(connected, add, connected);
    cv.subtract(weakOnly, add, weakOnly);
  }

  const edges = connected;
  const kernel2 = cv.Mat.ones(3,3,cv.CV_8U);
  cv.dilate(edges, edges, kernel2);
  kernel2.delete();

  let out;
  if (returnBinary) {
    out = new cv.Mat();
    cv.cvtColor(edges, out, cv.COLOR_GRAY2RGB);
  } else {
    const base = new cv.Mat();
    cv.cvtColor(src, base, cv.COLOR_RGBA2RGB);
    const red = new cv.Mat(base.rows, base.cols, cv.CV_8UC3, new cv.Scalar(255,0,0,0));
    const edgeRGB = new cv.Mat();
    red.copyTo(edgeRGB, edges);
    out = new cv.Mat();
    cv.addWeighted(base, 0.6, edgeRGB, 1.0, 0, out);
    base.delete(); red.delete(); edgeRGB.delete();
  }

  const edgeCount = cv.countNonZero(edges);

  gray.delete(); dx.delete(); dy.delete(); mag.delete(); ang.delete();
  nms.delete(); nms8.delete(); strong.delete(); weak.delete(); weakOnly.delete();
  kernel.delete(); dil.delete(); add.delete(); edges.delete();

  return { out, count: edgeCount };
}

// ---------- Task 2: Harris Corners ----------
function cornersHarris(src, blockSize = 5, k = 0.04, th = 70) {
  try {
    const gray = toGray(src);
    cv.GaussianBlur(gray, gray, new cv.Size(3,3), 0.8, 0.8);
    const Ix = new cv.Mat(), Iy = new cv.Mat();
    cv.Sobel(gray, Ix, cv.CV_32F, 1, 0, 3, 1, 0);
    cv.Sobel(gray, Iy, cv.CV_32F, 0, 1, 3, 1, 0);
    const Ixx = new cv.Mat(), Iyy = new cv.Mat(), Ixy = new cv.Mat();
    cv.multiply(Ix, Ix, Ixx); cv.multiply(Iy, Iy, Iyy); cv.multiply(Ix, Iy, Ixy);
    Ix.delete(); Iy.delete();
    const win = new cv.Size(blockSize, blockSize);
    const Sxx = new cv.Mat(), Syy = new cv.Mat(), Sxy = new cv.Mat();
    cv.GaussianBlur(Ixx, Sxx, win, 1.0); cv.GaussianBlur(Iyy, Syy, win, 1.0); cv.GaussianBlur(Ixy, Sxy, win, 1.0);
    Ixx.delete(); Iyy.delete(); Ixy.delete();
    const det = new cv.Mat(), tmp = new cv.Mat();
    cv.multiply(Sxx, Syy, det); cv.multiply(Sxy, Sxy, tmp); cv.subtract(det, tmp, det);
    const trace = new cv.Mat(), trace2 = new cv.Mat(); cv.add(Sxx, Syy, trace); cv.multiply(trace, trace, trace2);
    const R = new cv.Mat(); cv.addWeighted(det, 1.0, trace2, -k, 0.0, R);
    Sxx.delete(); Syy.delete(); Sxy.delete(); det.delete(); tmp.delete(); trace.delete(); trace2.delete();
    const Rn = new cv.Mat(); cv.normalize(R, Rn, 0, 255, cv.NORM_MINMAX, cv.CV_32F);
    const R8 = new cv.Mat(); Rn.convertTo(R8, cv.CV_8U); R.delete(); Rn.delete();
    const out = new cv.Mat(); cv.cvtColor(src, out, cv.COLOR_RGBA2RGB);
    const thMask = new cv.Mat(); cv.threshold(R8, thMask, th, 255, cv.THRESH_BINARY);
    const dil = new cv.Mat(); const ker = cv.Mat.ones(3,3,cv.CV_8U); cv.dilate(R8, dil, ker); ker.delete();
    const eq = new cv.Mat(); cv.compare(R8, dil, eq, cv.CMP_EQ);
    const maxima = new cv.Mat(); cv.bitwise_and(eq, thMask, maxima);
    const pts = new cv.Mat(); cv.findNonZero(maxima, pts);
    let count = pts.rows || 0;
    if (count === 0) {
      cv.threshold(R8, thMask, 40, 255, cv.THRESH_BINARY);
      cv.bitwise_and(eq, thMask, maxima);
      cv.findNonZero(maxima, pts);
      count = pts.rows || 0;
    }
    const color = new cv.Scalar(0,255,0,255);
    for (let i = 0; i < count; i++) {
      const p = pts.intPtr(i, 0);
      cv.circle(out, new cv.Point(p[0], p[1]), 4, color, 2, cv.LINE_AA);
    }
    R8.delete(); thMask.delete(); dil.delete(); eq.delete(); maxima.delete(); pts.delete();
    gray.delete();
    return { out, count };
  } catch (e) {
    return { out: src.clone(), count: 0 };
  }
}

// ---------- Task 3: Boundary Detection ----------
function boundaryContour(src, closeK = 5, minAreaPct = 2, epsPct = 1.5, centerR = 40, edgeLow = 20, edgeHigh = 60, edgeAuto = true) {
  const gray = toGray(src);
  cv.GaussianBlur(gray, gray, new cv.Size(5,5), 0.8, 0.8);
  
  let low = edgeLow, high = edgeHigh;
  if (edgeAuto) {
    const mag8 = gradMagnitudeMat(src);
    const tmp = new cv.Mat();
    const otsu = cv.threshold(mag8, tmp, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
    tmp.delete(); mag8.delete();
    high = Math.max(30, Math.round(otsu));
    low  = Math.max(5, Math.round(0.5 * high));
  }
  
  const edges = new cv.Mat(); cv.Canny(gray, edges, low, high, 3, true);
  const ksize = Math.max(1, closeK | 0);
  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(ksize, ksize));
  const closed = new cv.Mat(); cv.morphologyEx(edges, closed, cv.MORPH_CLOSE, kernel);
  kernel.delete(); edges.delete(); gray.delete();
  
  const contours = new cv.MatVector(); const hierarchy = new cv.Mat();
  cv.findContours(closed, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  closed.delete(); hierarchy.delete();
  
  const rows = src.rows, cols = src.cols;
  const imgArea = rows * cols;
  const minArea = (minAreaPct / 100) * imgArea;
  const centerR_pct = centerR / 100;
  const cx0 = cols / 2, cy0 = rows / 2;
  const diag = Math.hypot(cols, rows);
  let bestIdx = -1, bestScore = -1, bestArea = 0;
  
  for (let i = 0; i < contours.size(); i++) {
    const cnt = contours.get(i);
    const area = cv.contourArea(cnt, false);
    if (area < minArea) { cnt.delete(); continue; }
    const m = cv.moments(cnt, false);
    if (m.m00 === 0) { cnt.delete(); continue; }
    const cx = m.m10 / m.m00, cy = m.m01 / m.m00;
    const dist = Math.hypot(cx - cx0, cy - cy0) / diag;
    const centerOK = dist <= centerR_pct;
    const score = area * (centerOK ? 1.2 : 1.0) * (1.0 - 0.6 * dist);
    if (score > bestScore) { bestScore = score; bestIdx = i; bestArea = area; }
    cnt.delete();
  }
  
  const out = new cv.Mat(); cv.cvtColor(src, out, cv.COLOR_RGBA2RGB);
  if (bestIdx < 0) {
    contours.delete();
    return { out, info: 'no contour', area: 0, perimeter: 0, points: [] };
  }
  
  const best = contours.get(bestIdx);
  const perim = cv.arcLength(best, true);
  const epsPct_val = epsPct / 100.0;
  const epsilon = Math.max(0.5, epsPct_val * perim);
  const approx = new cv.Mat(); cv.approxPolyDP(best, approx, epsilon, true);
  const mask = cv.Mat.zeros(rows, cols, cv.CV_8U);
  const mv = new cv.MatVector(); mv.push_back(approx);
  cv.fillPoly(mask, mv, new cv.Scalar(255)); mv.delete();
  
  const fillRGB = new cv.Mat(rows, cols, cv.CV_8UC3, new cv.Scalar(0, 255, 0, 0));
  const filled = new cv.Mat(); fillRGB.copyTo(filled, mask);
  const blended = new cv.Mat(); cv.addWeighted(out, 0.65, filled, 0.35, 0, blended);
  const approxVec = new cv.MatVector(); approxVec.push_back(approx);
  cv.polylines(blended, approxVec, true, new cv.Scalar(0, 255, 0, 255), 2, cv.LINE_AA);
  approxVec.delete();
  
  const points = [];
  for (let i = 0; i < approx.rows; i++) {
    const p = approx.intPtr(i, 0);
    points.push({x: p[0], y: p[1]});
  }
  
  out.delete(); fillRGB.delete(); filled.delete(); mask.delete(); best.delete(); contours.delete(); approx.delete();
  
  return { out: blended, info: `area=${Math.round(bestArea)} px, verts=${approx.rows}`, area: bestArea, perimeter: perim, points };
}

// ---------- Task 4: ArUco Segmentation ----------
function segAruco(src, arucoAPI, dictName = 'DICT_4X4_50', useCorners = true, showIds = true, dilate = 2) {
  const rgb = new cv.Mat(); cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
  const gray = new cv.Mat(); cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

  if (!arucoAPI) {
    gray.delete(); rgb.delete();
    return { out: rgb, info: 'aruco-missing', markerCount: 0, area: 0, perimeter: 0, points: [] };
  }

  const corners = new cv.MatVector();
  const ids = new cv.Mat();
  
  try {
    arucoAPI.detect(gray, arucoAPI.dictionary, corners, ids, arucoAPI.params);
  } catch (e) {
    corners.delete(); ids.delete(); gray.delete(); rgb.delete();
    return { out: rgb, info: 'detection-failed', markerCount: 0, area: 0, perimeter: 0, points: [] };
  }

  if (ids.rows === 0) {
    corners.delete(); ids.delete(); gray.delete(); rgb.delete();
    return { out: rgb, info: 'no-markers', markerCount: 0, area: 0, perimeter: 0, points: [] };
  }

  const detectedCount = ids.rows;
  const pts = [];

  if (!showIds) {
    try {
      if (arucoAPI && arucoAPI.draw) {
        arucoAPI.draw(rgb, corners, ids);
      }
    } catch (e) {
      console.warn('API draw failed:', e);
    }
  }

  const cornerCount = corners.size ? corners.size() : 0;
  for (let i = 0; i < cornerCount; i++) {
    const c = corners.get(i);
    const f = c.data32F || c.data32S || c.data;
    if (!f || f.length < 8) {
      c.delete();
      continue;
    }

    if (useCorners) {
      for (let k = 0; k < 4; k++) {
        pts.push({ x: Math.round(f[k*2]), y: Math.round(f[k*2 + 1]) });
      }
    } else {
      const cx = (f[0] + f[2] + f[4] + f[6]) * 0.25;
      const cy = (f[1] + f[3] + f[5] + f[7]) * 0.25;
      pts.push({ x: Math.round(cx), y: Math.round(cy) });
    }

    if (showIds) {
      const id = ids.intAt ? ids.intAt(i, 0) : (ids.data32S ? ids.data32S[i] : i);
      const tx = Math.round((f[0] + f[2] + f[4] + f[6]) * 0.25);
      const ty = Math.round((f[1] + f[3] + f[5] + f[7]) * 0.25);
      cv.putText(rgb, String(id), new cv.Point(tx, ty), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(0,255,255,255), 2, cv.LINE_AA);
    }

    c.delete();
  }
  corners.delete();
  ids.delete();
  gray.delete();

  if (pts.length < 3) {
    rgb.delete();
    return { out: src.clone(), info: 'few-points', markerCount: detectedCount, area: 0, perimeter: 0, points: [] };
  }

  // Order points around centroid
  const cx = pts.reduce((s,p)=>s+p.x,0)/pts.length;
  const cy = pts.reduce((s,p)=>s+p.y,0)/pts.length;
  pts.sort((a,b)=> Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx));
  
  const contour = new cv.Mat(pts.length, 1, cv.CV_32SC2);
  for (let i = 0; i < pts.length; i++) {
    const ptr = contour.intPtr(i, 0);
    ptr[0] = pts[i].x; ptr[1] = pts[i].y;
  }

  const rows = src.rows, cols = src.cols;
  let mask = cv.Mat.zeros(rows, cols, cv.CV_8U);
  const mv = new cv.MatVector(); mv.push_back(contour);
  cv.fillPoly(mask, mv, new cv.Scalar(255)); mv.delete();

  const dil = Math.max(0, dilate);
  if (dil > 0) {
    const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(dil, dil));
    const tmp = new cv.Mat(); cv.dilate(mask, tmp, k); mask.delete(); mask = tmp; k.delete();
  }

  const fillRGB = new cv.Mat(rows, cols, cv.CV_8UC3, new cv.Scalar(0, 255, 0, 0));
  const filled = new cv.Mat(); fillRGB.copyTo(filled, mask);
  const blended = new cv.Mat(); cv.addWeighted(rgb, 0.65, filled, 0.35, 0, blended);

  const outline = new cv.Mat(pts.length, 1, cv.CV_32SC2);
  for (let i = 0; i < pts.length; i++) {
    const ptr = outline.intPtr(i, 0);
    ptr[0] = pts[i].x; ptr[1] = pts[i].y;
  }
  const outlineVec = new cv.MatVector();
  outlineVec.push_back(outline);
  cv.polylines(blended, outlineVec, true, new cv.Scalar(0,255,0,255), 2, cv.LINE_AA);
  outlineVec.delete();
  outline.delete();

  const area = cv.contourArea(contour);
  const perimeter = cv.arcLength(contour, true);

  fillRGB.delete(); filled.delete(); mask.delete();
  rgb.delete(); contour.delete();

  return { out: blended, info: `aruco pts=${pts.length}`, markerCount: detectedCount, area, perimeter, points: pts };
}

// Export functions for use in templates
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    getArucoAPI,
    toGray,
    percentileFrom8U,
    gradMagnitudeMat,
    gradAngleMat,
    logMat,
    edgeSimple,
    cornersHarris,
    boundaryContour,
    segAruco
  };
}

