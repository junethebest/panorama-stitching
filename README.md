# Panorama Stitching

A Python implementation of a panorama stitching pipeline based on feature detection, feature matching, homography estimation, image warping, and image blending.

This project implements the main steps of traditional image stitching from scratch, including Harris corner detection, SIFT-like descriptor extraction, NCC-based feature matching, RANSAC-based homography estimation, and final panorama generation.

## Overview

Panorama stitching aims to combine two or more overlapping images into a wider-view image. In this project, the stitching pipeline is implemented using OpenCV, NumPy, and Matplotlib.

The main workflow is:

1. Detect feature points using a custom Harris corner detector.
2. Extract local SIFT-like descriptors around feature points.
3. Match features using normalized cross-correlation and nearest-neighbor ratio filtering.
4. Estimate the homography matrix using RANSAC.
5. Warp images into a common coordinate system.
6. Blend overlapping regions and generate the final panorama.

## Features

- Custom Harris corner detection
- SIFT-like local descriptor extraction
- Feature matching with NCC and ratio test
- Cross-check matching strategy
- RANSAC-based homography estimation
- Reprojection error analysis
- Image warping using perspective transformation
- Average and linear blending modes
- Intermediate visualization of corners, descriptors, matches, and RANSAC inliers
- Comparison with OpenCV-based stitching results

## Repository Structure

```text
.
├── data/
│   └── custom/
│       ├── indoor/
│       └── outdoor/
├── intermediate/
│   ├── feature_match/
│   ├── harris_corner/
│   ├── ransac/
│   └── sift_descriptor/
├── results/
│   ├── basic_experiment/
│   └── opencv_comparison/
│       └── indoor/
├── paronama_stitching.py
└── README.md
```

## Main File

| File | Description |
|---|---|
| `paronama_stitching.py` | Main Python script containing the panorama stitching pipeline and related visualization functions. |

> Note: The current file name is `paronama_stitching.py`. If you want to make the repository cleaner, you may rename it to `panorama_stitching.py`.

## Requirements

Recommended environment:

- Python 3.8 or later
- OpenCV
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install opencv-python numpy matplotlib
```

## How to Use

Clone this repository:

```bash
git clone https://github.com/junethebest/panorama-stitching.git
cd panorama-stitching
```

Run the Python script:

```bash
python paronama_stitching.py
```

If you use the stitching function directly in another script, a typical usage is:

```python
import cv2
from paronama_stitching import panorama_stitching, harris_params

img1 = cv2.imread("data/custom/indoor/left.jpg")
img2 = cv2.imread("data/custom/indoor/right.jpg")

result = panorama_stitching(img1, img2, harris_params)

cv2.imwrite("results/panorama_result.jpg", result)
```

Please adjust the image paths according to your own input image names.

## Method

### 1. Harris Corner Detection

The project first detects candidate feature points using a custom Harris corner detector. The detector computes image gradients, constructs the local second-moment matrix, calculates the Harris response, and applies thresholding and non-maximum suppression.

### 2. SIFT-like Descriptor Extraction

For each detected corner, a local patch is extracted. Gradient magnitudes and orientations are used to build a histogram-based descriptor. The descriptor is normalized to improve robustness against illumination changes.

### 3. Feature Matching

Feature descriptors from two images are matched using normalized cross-correlation. A ratio test and cross-check strategy are used to filter unreliable matches.

### 4. Homography Estimation with RANSAC

RANSAC is used to estimate a robust homography matrix from matched feature points. Outliers are rejected according to reprojection error.

### 5. Warping and Blending

The estimated homography is used to warp images into a common coordinate system. The overlapping region is then blended to obtain the final panorama.

## Results

The project saves both intermediate visualizations and final stitching results.

Intermediate results include:

- Harris corner visualization
- SIFT-like descriptor visualization
- Feature matching visualization
- RANSAC inlier/outlier visualization
- Reprojection error analysis

Final results are saved under:

```text
results/
```

If result images are available, they can be displayed in this README by adding:

```markdown
![Panorama Result](results/basic_experiment/your_result_image.jpg)
```

Replace `your_result_image.jpg` with the actual result image file name.

## Notes

This project focuses on understanding and implementing the core steps of classical panorama stitching. The result quality depends on image overlap, feature quality, camera motion, parallax, exposure differences, and the number of reliable matched points.

For scenes with strong parallax, moving objects, or low texture, the homography-based method may produce ghosting, distortion, or alignment errors.

## Possible Improvements

- Rename `paronama_stitching.py` to `panorama_stitching.py`
- Add command-line arguments for input and output paths
- Add a `requirements.txt` file
- Support stitching more than two images
- Improve blending with multi-band blending
- Add more result images to the README
- Separate the current large script into smaller modules

## License

This project is released under the MIT License.
