# myelin-segmentation

Python programs for automating myelinated axon segmentation in microscopy images.

---

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/KK201431873/myelin-segmentation.git
    cd myelin-segmentation
    ```

2. Install Python 3 if not already installed.

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

---

## How to Use

There are two main Python files:

- `gratio_tuner.py`: allows the user to tune the segmentation algorithm to an image and download those settings.
- `gratio_image_processor.py`: applies those settings to a batch of images and then allows the user to clean/export the data.

### Using `gratio_tuner.py`

1. Open a terminal in the project directory and run:

    ```bash
    python gratio_tuner.py
    ```

2. Select the image you want to tune the algorithm to.

3. Two windows will appear:
   - A thresholded version of your image.
   - A control panel with several sliders.

#### Adjusting the Sliders

- **Threshold**: Controls brightness threshold for identifying axon interiors.  
  _Higher = stricter, lower = more inclusive._

- **Radius**: Smooths edges and reduces noise.  
  _Higher = less detail._

- **Dilate**: Expands white pixel regions and fills gaps.

- **Erode**: Contracts white regions. Ideally, dilate and erode should be equal.

- **Minimum Size**: Minimum area (in thousands of pixels) for a contour to be recognized.

- **Maximum Size**: Maximum area (in thousands of pixels).

- **Convexity**: Filters out contours with low convexity.  
  _Convexity = Area / Convex Hull Area._

- **Circularity**: Filters out contours with low circularity.  
  _Circularity = 4π × Area / (Perimeter²)._

- **Toggle Contours**: Slide right to preview the algorithm’s detections.

#### Tuning Tips

- It's better to have more false positives than false negatives. Therefore, it may be ideal to keep convexity and circularity at 0.
- Start with all sliders at 0 except threshold. Tune threshold until the axon interiors are clearly isolated from the outside.
- Radius might not be useful in detailed images.
- Dilate and erode might not be useful in images that are crowded and detailed.
- You may not need to adjust min/max size often.

#### Save Your Settings

- Press **CTRL+S** to save current settings to a JSON file.
- Settings auto-save if the program is closed accidentally.

---

### Using `gratio_image_processor.py`

1. Run:

    ```bash
    python gratio_image_processor.py
    ```

2. A GUI will appear with five buttons and a textbox—this is where instructions and logs will appear.

#### Buttons Instructions

Click these buttons in order—

- **Select Settings**: Choose the JSON settings file you saved.
- **Select Images**: Select the images you want to segment using these settings.
- **Process Images**: Choose an output folder. A "STOP" button will appear to the right if you need to cancel the operation.
  - This generates `.pkl` files containing contours and measurements for each image.
- **Review Output**: Select the `.pkl` files you want to review.
  - _Note: This step just disables the contours; don't worry if you misclick._
- **Generate Data**: Select the `.pkl` files you want to generate a CSV file for. Then, choose an output folder to save your data to.

---

## Modifying the Algorithm

The segmentation algorithm is in the `process_image(...)` functions in both Python files. Feel free to edit the code in `gratio_tuner.py`, but make sure to copy your changes to `gratio_image_processor.py` to maintain consistency.

---
