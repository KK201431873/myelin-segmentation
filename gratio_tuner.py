import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Tk
from PIL import Image, ImageTk
import math
import json
import os
from pathlib import Path
import time

CONFIG_DIR = "__program_config__"
CONFIG_FILE = f"{CONFIG_DIR}/config_tuner.json"
os.makedirs(CONFIG_DIR, exist_ok=True)

# Initial parameters
def init_default_settings():
    global init_thresh, init_radius, init_dilate, init_erode, init_min_size, init_max_size, init_convex_thresh, init_circ_thresh
    init_thresh = 127
    init_radius = 0
    init_dilate = 0
    init_erode = 0
    init_min_size = 10000
    init_max_size = 1000000
    init_convex_thresh = 0
    init_circ_thresh = 0

root_dir = Path(__file__).parent
config_abs_path = os.path.join(root_dir, CONFIG_FILE.replace("/","\\"))
if os.path.exists(config_abs_path):
    with open(CONFIG_FILE, "r") as f:
        try:
            config = json.load(f)
            init_thresh = config["thresh"]
            init_radius = config["radius"]
            init_dilate = config["dilate"]
            init_erode = config["erode"]
            init_min_size = config["min_size"]
            init_max_size = config["max_size"]
            init_convex_thresh = config["convex_thresh"]
            init_circ_thresh = config["circ_thresh"]
        except Exception as e:
            init_default_settings()
else:
    init_default_settings()
finished_setup = False

def create_circular_kernel(radius):
    """Create a circular kernel mask"""
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x*x + y*y <= radius*radius
    return mask.astype(np.float32)

def convexness(contour, hull):
    contour_area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(hull)
    return contour_area/hull_area

def process_image(input_image, thresh_val, radius_val, dilate, erode, min_size, max_size, convex_thresh, circ_thresh):
    dilate = round(dilate * linear_correction_ratio)
    erode = round(erode * linear_correction_ratio)
    min_size = min_size * area_correction_ratio
    max_size = max_size * area_correction_ratio
    
    # Threshold image (binary)
    kernel = create_circular_kernel(radius_val)
    kernel_sum = np.sum(kernel)
    kernel /= kernel_sum
    img = cv2.filter2D(src=input_image, ddepth=-1, kernel=kernel)
    _, thresh = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)

    # Remove small black features
    inverted = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if cv2.contourArea(c) < 1000 * area_correction_ratio:
            cv2.drawContours(thresh, [c], -1, 255, cv2.FILLED)
    
    # # Remove small long white features
    # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # for c in contours:
    #     if cv2.contourArea(c) < 10000 * area_correction_ratio:
    #         cv2.drawContours(thresh, [c], -1, 0, cv2.FILLED)

    # Dilate image
    dilate_size = max(1, int(dilate))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size,dilate_size))
    dilated = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, dilate_kernel)

    # Dilate image
    erode_size = max(1, int(erode))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size,erode_size))
    eroded = cv2.morphologyEx(dilated, cv2.MORPH_ERODE, erode_kernel)

    if cv2.getTrackbarPos("Toggle Contours", control_window)==0:
        return eroded, []
    
    # Find contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # Filter contours by size and convexness
    filtered_contours = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for c in contours:
        # Check if contour touches edge of image
        if np.any(c[:, 0, 0] <= 1) or np.any(c[:, 0, 0] >= (w-2)) or \
           np.any(c[:, 0, 1] <= 1) or np.any(c[:, 0, 1] >= (h-2)):
            continue
        
        # Check if inner edge is black
        bx, by, bw, bh = cv2.boundingRect(c)
        eroded_roi = eroded[by:by+bh, bx:bx+bw]
        mask_shape = (bh, bw)
        full_mask = np.zeros(mask_shape, dtype=np.uint8)
        shifted_c = c - [bx, by]
        cv2.drawContours(full_mask, [shifted_c], -1, color=255, thickness=cv2.FILLED)
        eroded_mask1 = cv2.erode(full_mask, kernel, iterations=1)
        eroded_mask2 = cv2.erode(eroded_mask1, kernel, iterations=1)
        inner_edge_mask = cv2.subtract(eroded_mask1, eroded_mask2) # 1 pixel thick inner edge
        inner_pixels = eroded_roi[inner_edge_mask == 255]
        if len(inner_pixels) == 0 or np.mean(inner_pixels) < 128:
            continue
        
        # Check size
        c_area = cv2.contourArea(c)
        if not (min_size <= c_area <= max_size):
            continue
        
        # Check convexness
        hull = cv2.convexHull(c, returnPoints=True)
        convex = convexness(c, hull)
        if convex < convex_thresh:
            continue
        
        # Check circularity
        circularity = 4 * math.pi * cv2.contourArea(c) / (cv2.arcLength(c, closed=True) ** 2) if cv2.arcLength(c, closed=True) != 0 else 0
        if circularity < circ_thresh:
            continue

        # Check if scale label interferes with the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x_limit = 290/1365*eroded.shape[0]
            y_limit = eroded.shape[1]-165/1365*eroded.shape[1]
            if cx<=x_limit and cy>= y_limit:
                continue

        filtered_contours.append(c)
    
    # Create output color image for visualization
    out_img = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    
    # Create binary mask of all contours
    all_contours_mask = np.zeros_like(eroded)
    cv2.drawContours(all_contours_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

    # return all_contours_mask, []
    
    data = [] # (ID, gratio, circularity, inner_diameter, outer_diameter, myelin_thickness)
    for i, contour in enumerate(filtered_contours):
        # Create mask of this contour
        contour_mask = np.zeros_like(eroded)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Compute center of mass of the contour
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        # Define bounding box around center of mass to crop candidate area
        search_radius = int(cv2.arcLength(contour, True) / (2*math.pi) + input_image.shape[0]/8)
        x_min = max(cx - search_radius, 0)
        x_max = min(cx + search_radius + 1, eroded.shape[1])
        y_min = max(cy - search_radius, 0)
        y_max = min(cy + search_radius + 1, eroded.shape[0])

        # Crop exclusion mask and eroded image
        cropped_eroded = eroded[y_min:y_max, x_min:x_max]
        cropped_contour_mask = contour_mask[y_min:y_max, x_min:x_max]

        # Build exclusion mask inside this cropped area
        exclusion_raw = ((cropped_contour_mask == 0) & (cropped_eroded == 255)).astype(np.uint8)
        exclusion_mask = cv2.Canny(exclusion_raw,0,0)
        distance = cv2.distanceTransform(255 - cropped_contour_mask, distanceType=cv2.DIST_L2, maskSize=5)[exclusion_mask > 0]

        # print(max([max(row) for row in distance]))
        # return cv2.bitwise_and((distance/max([max(row) for row in distance])*255).astype(np.uint8), exclusion_mask)
        
        # Flatten the array and filter out zeros
        nonzero_vals = distance[distance > 0]

        # Safety check: fewer than n nonzero values
        n = 2*len(contour)
        if len(nonzero_vals) < n:
            smallest = np.sort(nonzero_vals)
        else:
            # Get the n smallest nonzero values (unsorted)
            smallest = np.partition(nonzero_vals, n - 1)[:n]

        # Get thickness via percentile
        thickness = np.percentile(np.array(smallest), 30)

        ### generate visualization ###
        # Use distance transform
        distance = cv2.distanceTransform(255 - cropped_contour_mask, distanceType=cv2.DIST_L2, maskSize=5)

        # Draw outer and inner contours
        offset_mask = (distance <= thickness).astype(np.uint8) * 255
        offset_mask = offset_mask.astype(np.uint8)
        outer_contour, _ = cv2.findContours(offset_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        distance_inner = cv2.distanceTransform(offset_mask, cv2.DIST_L2, maskSize=5)
        offset_mask_eroded = (distance_inner > thickness).astype(np.uint8) * 255
        inner_contour, _ = cv2.findContours(offset_mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        draw_scale = int(8*linear_correction_ratio)

        # Calculate g-ratio
        contour_perimeter = cv2.arcLength(inner_contour[0], True)
        radius = contour_perimeter / (2*math.pi)
        g_ratio = radius / (radius + thickness)

        # Calculate circularity
        circularity = 4 * math.pi * cv2.contourArea(inner_contour[0]) / (contour_perimeter ** 2) if contour_perimeter != 0 else 0
        
        # Draw contour and label thickness on output image
        cv2.drawContours(out_img, inner_contour + np.array([[[x_min, y_min]]]), -1, (0, 255, 0), draw_scale)
        cv2.drawContours(out_img, outer_contour + np.array([[[x_min, y_min]]]), -1, (0, 255, 0), draw_scale)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"]-15*draw_scale)
            cy = int(M["m01"] / M["m00"]+6*draw_scale)
            line_spacing = 6*draw_scale
            cv2.putText(out_img, f"#{i+1}", (cx, cy-line_spacing), 
                        cv2.FONT_HERSHEY_PLAIN, draw_scale, (0, 0, 0), draw_scale, cv2.LINE_AA)
            cv2.putText(out_img, f"{g_ratio:.2f}", (cx, cy+line_spacing), 
                        cv2.FONT_HERSHEY_PLAIN, draw_scale, (255, 0, 0), draw_scale, cv2.LINE_AA)
            
        inner_diameter = 2*radius
        outer_diameter = inner_diameter + 2*thickness
        data.append((i+1, float(g_ratio), float(circularity), float(inner_diameter), float(outer_diameter), float(thickness)))
        # print(f"myelin thickness {thickness} | axon radius {radius} | g_ratio {g_ratio}")
    
    return out_img, data

# Open file dialog
def select_image_file():
    global root
    root.withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
    )
    return file_path

# Callback for trackbar
def update(val=0):
    global root, label, tk_image
    if not finished_setup or not root:
        return
    # Read all slider values
    thresh = cv2.getTrackbarPos("Threshold", control_window)
    radius = cv2.getTrackbarPos("Radius", control_window)
    dilate = cv2.getTrackbarPos("Dilate", control_window)
    erode = cv2.getTrackbarPos("Erode", control_window)
    min_size = cv2.getTrackbarPos("Min Size (K)", control_window)*1000
    max_size = cv2.getTrackbarPos("Max Size (K)", control_window)*1000
    convex_thresh = cv2.getTrackbarPos("Convexity %", control_window) / 100.0
    circ_thresh = cv2.getTrackbarPos("Circularity %", control_window) / 100.0

    # Process and show
    out_img, _ = process_image(img_gray, thresh, radius, dilate, erode, min_size, max_size, convex_thresh, circ_thresh)
    scale = 0.64
    out_img = cv2.resize(out_img, None, fx=scale, fy=scale)
    
    pil_image = Image.fromarray(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    tk_image = ImageTk.PhotoImage(pil_image)
    label.config(image=tk_image)

# Window names
control_window = "Controls"
image_window = "Processed Image"

# Main
root = Tk()
file_path = select_image_file()
if not file_path:
    print("No file selected.")
    exit()

# Load image
img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise ValueError("Image not loaded. Check the path!")
divisor = 3
img_gray = cv2.resize(img_gray, None, fx=1/divisor, fy=1/divisor)

h, w = img_gray.shape
image_size = img_gray.shape[0]*img_gray.shape[1]
area_correction_ratio = image_size / (3608 * 4096) 
linear_correction_ratio = math.sqrt(area_correction_ratio)

# Create window and sliders
cv2.namedWindow(control_window)
cv2.resizeWindow(control_window, 500, 400)

cv2.createTrackbar("Threshold", control_window, init_thresh, 255, update)
cv2.createTrackbar("Radius", control_window, init_radius, 20, update)
cv2.createTrackbar("Dilate", control_window, init_dilate, 50, update)
cv2.createTrackbar("Erode", control_window, init_erode, 50, update)
cv2.createTrackbar("Min Size (K)", control_window, init_min_size//1000, 200, update)
cv2.createTrackbar("Max Size (K)", control_window, init_max_size//1000, 1000, update)
cv2.createTrackbar("Convexity %", control_window, int(100*init_convex_thresh), 100, update)
cv2.createTrackbar("Circularity %", control_window, int(100*init_circ_thresh), 100, update)
cv2.createTrackbar("Toggle Contours", control_window, 0, 1, update)

def auto_save_config():
    save_config(CONFIG_FILE)

def user_save_config(event=None):
    default_filename = f"{file_name}_settings.json"
    filepath = filedialog.asksaveasfilename(
        initialdir=os.getcwd(),
        initialfile=default_filename,
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")]
    )
    print(f"Settings saved to: {filepath}")
    save_config(filepath)

def save_config(filepath):
    if filepath:
        try:
            with open(filepath, "w") as f:
                settings = {
                    "thresh" : cv2.getTrackbarPos("Threshold", control_window),
                    "radius" : cv2.getTrackbarPos("Radius", control_window),
                    "dilate" : cv2.getTrackbarPos("Dilate", control_window),
                    "erode" : cv2.getTrackbarPos("Erode", control_window),
                    "min_size" : cv2.getTrackbarPos("Min Size (K)", control_window)*1000,
                    "max_size" : cv2.getTrackbarPos("Max Size (K)", control_window)*1000,
                    "convex_thresh" : cv2.getTrackbarPos("Convexity %", control_window) / 100.0,
                    "circ_thresh" : cv2.getTrackbarPos("Circularity %", control_window) / 100.0,
                }
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Failed to save file: {e}")

def close(event=None):
    auto_save_config()
    root.quit()
    root.destroy()
    cv2.destroyAllWindows()

root.title(image_window)
root.protocol("WM_DELETE_WINDOW", close)
root.bind("<Control-s>", user_save_config)
root.bind("<Escape>", close)
label = tk.Label(root)
label.pack()

# Initial display
finished_setup = True
root.after(0, update)
root.after(0, root.wm_deiconify)
print("Press ESCAPE to exit.")
print("Press CTRL+S to save settings.")
file_name = file_path.split("/")[-1].split(".")[0]
root.mainloop()
