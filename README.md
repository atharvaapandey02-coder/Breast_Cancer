# Advanced Cell Boundary Detection using Active Contour Models (Snakes)

## Overview

This project presents an interactive cell boundary detection system
designed for microscopic biomedical images. It combines manual contour
annotation with automated refinement using the Active Contour (Snake)
model to extract quantitative morphological features of cells.

The system is implemented in Python using OpenCV, scikit-image, and
Tkinter, enabling researchers to perform semi-automated feature
extraction in an intuitive graphical interface.

------------------------------------------------------------------------

## Key Features

-   Interactive GUI built with Tkinter
-   Manual boundary drawing using mouse input
-   Active Contour (Snake) based boundary refinement
-   Zoom-in / Zoom-out functionality
-   Automatic feature computation:
    -   Area
    -   Perimeter
    -   Radius
    -   Smoothness
-   Statistical aggregation (Mean, Standard Error, Worst values)
-   Excel export of computed features
-   Support for magnification adjustment factor

------------------------------------------------------------------------

## Methodology

1.  Image Loading and Preprocessing
    -   Image loaded using OpenCV
    -   Converted to grayscale
    -   Gaussian smoothing applied for improved contour stability
    -   Sobel edge detection used for energy minimization
2.  Manual Boundary Initialization
    -   User manually draws initial contour
    -   Polygon stored as initial snake
3.  Active Contour Refinement
    -   scikit-image active_contour algorithm refines boundary
    -   Periodic boundary condition ensures closed contour
4.  Feature Extraction
    -   Area using cv2.contourArea
    -   Perimeter using cv2.arcLength
    -   Equivalent Radius derived from area
    -   Smoothness based on radial deviation from centroid
    -   Magnification correction applied
5.  Feature Export
    -   Features aggregated across refined cells
    -   Results appended to Output_Features.xlsx

------------------------------------------------------------------------

## Project Structure

├── Dataset_formation(delete, category, wisconsin).py ├── Output_Features.xlsx ├──
sample_images/ ├── README.md └── requirements.txt

------------------------------------------------------------------------

## Installation

Clone the repository:

    git clone <https://github.com/atharvaapandey02-coder/Breast_Cancer.git>
    cd <Breast_Cancer>

Install dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------

## How to Run

    python cell_boundary_detector.py

------------------------------------------------------------------------

## Output Format

The exported Excel file contains:

-   Serial no.
-   area_mean, area_se, area_worst
-   perimeter_mean, perimeter_se, perimeter_worst
-   radius_mean, radius_se, radius_worst
-   smoothness_mean, smoothness_worst
-   snake_refined_count

------------------------------------------------------------------------

## Applications

-   Breast cancer cell morphology analysis
-   Histopathological image quantification
-   Biomedical research feature extraction
-   Semi-automated annotation systems

------------------------------------------------------------------------

## Technologies Used

-   Python
-   OpenCV
-   scikit-image
-   NumPy
-   Pandas
-   Tkinter
-   Pillow

------------------------------------------------------------------------

## Author

Atharva Pandey

------------------------------------------------------------------------

## License

This project is intended for academic and research purposes.
