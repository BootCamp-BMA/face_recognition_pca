# Face Recognition using PCA (Principal Component Analysis)

This project implements a facial recognition system using PCA for dimensionality reduction.

**Table of Contents:**

*   [Introduction](#introduction)
*   [Installation](#installation)
*   [Requirements](#requirements)
*   [Data](#data)
*   [Usage](#usage)
    *   [Train the Model](#train)
    *   [Test the Model](#test)
*   [Training](#training)
*   [Testing](#testing)
*   [Results](#results)
*   [Contributors](#contributors)

**Introduction:**

This project utilizes PCA to build a simple facial recognition system. It uses image datasets in `.pgm` format.

**Installation:**

1.  Clone the repository:

    ```bash
    git clone [https://github.com/BootCamp-BMA/face_recognition_pca.git](https://github.com/BootCamp-BMA/face_recognition_pca.git)
    cd face_recognition_pca
    ```

2.  (Optional) Create a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

**Requirements:**

*   `numpy`
*   `scikit-learn`
*   `opencv-python`
*   `matplotlib`
*   `PIL` (Pillow)
*   `scipy`

**Data:**

*   Training data: `train` directory
*   Testing data: `test` directory
*   Each person has a folder named after them containing their images.

**Usage:**

**Train the Model** (using images in `train` directory):

```bash
python recongntionFacial.py --train