import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
from pathlib import Path
import cv2
import numpy as np
import math
import keyboard
import pickle
import threading
import time
from datetime import datetime
from screeninfo import get_monitors

CONFIG_DIR = "__program_config__"
root_dir = Path(__file__).parent
CONFIG_FILE = os.path.join(root_dir, CONFIG_DIR, "config_imgproc.json")
os.makedirs(CONFIG_DIR, exist_ok=True)
MONITOR = get_monitors()[0]

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("G-ratio Image Processor")
        self.root.geometry("960x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        # Settings
        self.settings_frame = tk.Frame(root)
        self.settings_frame.pack(pady=5, fill="x", padx=10)
        tk.Button(self.settings_frame, text="Select Settings", command=self.select_settings).pack(side="left")
        self.settings_label = tk.Label(self.settings_frame, text="No file selected", anchor="w")
        self.settings_label.pack(side="left", padx=10, fill="x", expand=True)

        # Images
        self.images_frame = tk.Frame(root)
        self.images_frame.pack(pady=5, fill="x", padx=10)
        tk.Button(self.images_frame, text="Select Images", command=self.select_images).pack(side="left")
        self.images_label = tk.Label(self.images_frame, text="No images selected", anchor="w")
        self.images_label.pack(side="left", padx=10, fill="x", expand=True)

        # Process images
        self.imgproc_frame = tk.Frame(root)
        self.imgproc_frame.pack(pady=5, fill="x", padx=10)
        tk.Button(self.imgproc_frame, text="Process Images", command=self.process_images).pack(side="left")
        self.imgproc_label = tk.Label(self.imgproc_frame, text="Inactive", anchor="w")
        self.imgproc_label.pack(side="left", padx=10, fill="x", expand=True)

        # STOP button (initially hidden)
        self.stop_button = tk.Button(self.imgproc_frame, text="STOP", fg="white", bg="red", command=self.stop_processing)
        self.stop_button.pack(side="left", padx=10)
        self.stop_button.pack_forget()  # Hide by default

        # Review output
        self.review_frame = tk.Frame(root)
        self.review_frame.pack(pady=5, fill="x", padx=10)
        tk.Button(self.review_frame, text="Review Output", command=self.review_output).pack(side="left")

        # Generate Data
        self.gen_data_frame = tk.Frame(root)
        self.gen_data_frame.pack(pady=5, fill="x", padx=10)
        tk.Button(self.gen_data_frame, text="Generate Data", command=self.generate_data).pack(side="left")

        # Settings list display
        self.output_frame = tk.Frame(root)
        self.output_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.output_text = tk.Text(self.output_frame, height=10, state="disabled", wrap="word")
        self.output_text.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(self.output_frame, command=self.output_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.output_text.configure(yscrollcommand=scrollbar.set)
        self.read_settings({})

        self.settings = None
        self.settings_path = None
        self.image_paths = []
        self.save_directory = None
        self.load_config()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                try:
                    config = json.load(f)
                    if "settings_path" in config.keys() and config["settings_path"] and os.path.exists(config["settings_path"]):
                        self.settings_path = config["settings_path"]
                        self.load_settings(self.settings_path)
                        self.read_settings(self.settings)
                    else:
                        self.settings = None
                        self.settings_path = None
                    if "image_paths" in config.keys() and len(config["image_paths"])>0:
                        self.image_paths = config["image_paths"]
                        self.image_paths = [x for x in self.image_paths if os.path.exists(x)]
                        if len(self.image_paths)>0:
                            self.load_images()
                    return
                except Exception as e:
                    print(e)
                    pass
        
        self.settings = None
        self.settings_path = None
        self.image_paths = []

    def select_settings(self):
        path = filedialog.askopenfilename(
            title="Select Settings JSON",
            filetypes=[("JSON Files", "*.json")]
        )
        if path:
            self.load_settings(path)
        if self.settings:
            self.read_settings(self.settings)
    
    def load_settings(self, path):
        try:
            with open(path, 'r') as f:
                self.settings = json.load(f)
                self.settings_path = path
            filename = os.path.basename(path)
            self.settings_label.config(text=filename)
        except Exception as e:
            self.settings = None
            self.settings_label.config(text="No file selected")
            messagebox.showerror("Error", f"Failed to load settings:\n{e}")

    def read_settings(self, settings):
        keys = ["thresh","radius","dilate","erode","min_size","max_size","convex_thresh","circ_thresh"]
        if all(key in settings for key in keys):
            self.init_thresh = settings["thresh"]
            self.init_radius = settings["radius"]
            self.init_dilate = settings["dilate"]
            self.init_erode = settings["erode"]
            self.init_min_size = settings["min_size"]
            self.init_max_size = settings["max_size"]
            self.init_convex_thresh = settings["convex_thresh"]
            self.init_circ_thresh = settings["circ_thresh"]

        res_text = "Settings:"
        for key in keys:
            res_text += f"\n|   {key}: {settings[key] if key in settings else '-'}"
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("end", res_text)
        self.output_text.config(state="disabled")

    def select_images(self):
        files = filedialog.askopenfilenames(
            title="Select Image Files",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if files:
            self.image_paths = list(files)
        if self.image_paths:
            self.load_images()
    
    def load_images(self):
        filenames = [os.path.basename(p) for p in self.image_paths]

        # Truncate if too long
        display_text = ", ".join(filenames[:3])
        if len(filenames) > 3:
            display_text += f", ... ({len(filenames)} total)"

        self.images_label.config(text=display_text)

        # Set textbox text
        selected_images_text = "Selected Images:"
        for filename in filenames:
            selected_images_text += f"\n|   {filename}"
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("end", selected_images_text)
        self.output_text.config(state="disabled")

    def process_images(self):
        if not self.settings:
            messagebox.showwarning("Missing Settings", "Please select a settings JSON file first.")
            return
        if not self.image_paths:
            messagebox.showwarning("No Images", "Please select one or more image files.")
            return
        
        selected_dir = filedialog.askdirectory(title="Select Output Folder")
        if not selected_dir:
            return

        self.save_directory = Path(selected_dir)
        
        self.stop_requested = False
        self.root.after(0, self.stop_button.pack)
        threading.Thread(target=self._process_images_thread, daemon=True).start()

    def _process_images_thread(self):
        start_time = time.time()
        self.update_progress(-1, 0)
        for i, img_path in enumerate(self.image_paths):
            print(f"Processing {img_path}")

            # Read image
            divisor = 3
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, None, fx=1/divisor, fy=1/divisor)
            if img is None:
                print(f"Skipping unreadable file: {img_path}")
                continue

            # Process the image
            contour_data = self.process_image(
                img, 
                self.init_thresh, 
                self.init_radius, 
                self.init_dilate, 
                self.init_erode, 
                self.init_min_size, 
                self.init_max_size, 
                self.init_convex_thresh, 
                self.init_circ_thresh
            )
            
            if self.stop_requested:
                elapsed_time = time.time() - start_time
                self.update_progress(i-1, -elapsed_time)
                break

            img_filename = os.path.basename(img_path)
            file_data = {
                "img_filename": img_filename,
                "image_data": cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                "contour_data": contour_data,
                "selected_states": {ID:True for ID,*_ in contour_data}
            }
            
            img_name = os.path.basename(img_path).split(".")[0]
            imgproc_out_path = self.save_directory / f"{img_name}_imgproc_out.pkl"
            with open(imgproc_out_path, "wb") as f:
                pickle.dump(file_data, f)

            elapsed_time = time.time() - start_time
            self.update_progress(i, elapsed_time)

        self.root.after(0, self.stop_button.pack_forget)

        if self.stop_requested:
            self.root.after(0, lambda: messagebox.showinfo("Processing Stopped", "Image processing was stopped."))
        else:
            self.root.after(0, lambda: messagebox.showinfo("Processing Complete", "Finished processing images."))
    
    def stop_processing(self):
        self.stop_requested = True

    def review_output(self):
        files = filedialog.askopenfilenames(
            title="Select Imgproc Output Files",
            filetypes=[("Python Pickle Files", "*.pkl")]
        )
        self.show_review_instructions()
        self.root.update_idletasks()
        if files:
            files = list(files)
            total = len(files)
            running = True
            i = 0
            while running:
                file_path = files[i]
                filename = os.path.basename(file_path)
                exit_code = self.interactive_reviewer(
                    file_path,
                    display_scale=0.64,
                    window_title=f"Review - {filename} ({i+1}/{total} images)"
                )
                if exit_code==0:
                    running = False
                    break
                i = (i+exit_code+total) % total
                    
    
    def generate_data(self):
        files = filedialog.askopenfilenames(
            title="Select Imgproc Output Files",
            filetypes=[("Python Pickle Files", "*.pkl")]
        )
        if not files:
            return
        
        files = list(files)
        formatted_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_imgs, csv_lines = self.get_csv_lines(files, formatted_datetime=formatted_datetime)
        
        default_filename = f"gratios_data_{formatted_datetime}.csv"
        csv_filepath = filedialog.asksaveasfilename(
            initialdir=os.getcwd(),
            initialfile=default_filename,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not csv_filepath:
            return
        
        parent_dir = os.path.dirname(csv_filepath)
        parent_dirname = os.path.basename(parent_dir)

        csv_filename = os.path.basename(csv_filepath).split(".")[0]
        csv_dir = os.path.dirname(csv_filepath) + f"/{csv_filename}/"
        csv_filepath = csv_dir + os.path.basename(csv_filepath)
        os.makedirs(csv_dir, exist_ok=True)
        if csv_filepath:
            with open(csv_filepath, mode='w') as f:
                f.writelines(csv_lines)

        image_directory = os.path.dirname(csv_filepath) + f"/images_{formatted_datetime}"
        os.makedirs(image_directory, exist_ok=True)
        for filename, image in out_imgs:
            out_img_path = f"{image_directory}/{filename}"
            print(f"Writing {str(out_img_path)}")
            cv2.imwrite(str(out_img_path), image)

        generated_data_text = f"Generated in folder '{parent_dirname}/':"
        generated_data_text += f"\n{csv_filename}/"
        generated_data_text += f"\n|   {csv_filename}.csv"
        generated_data_text += f"\n|   images_{formatted_datetime}/"
        for filename, _ in out_imgs:
            generated_data_text += f"\n|   |   {filename}"
        self.root.after(0, self.output_text.config, {"state":"normal"})
        self.root.after(0, lambda: self.output_text.delete("1.0", "end"))
        self.root.after(0, lambda: self.output_text.insert("end", generated_data_text))
        self.root.after(0, self.output_text.config, {"state":"disabled"})
        
        os.startfile(os.path.normpath(csv_dir))
    
    def on_exit(self):
        try:
            with open(CONFIG_FILE, "w") as f:
                config = {
                    "settings_path": self.settings_path,
                    "image_paths": self.image_paths
                }
                json.dump(config, f, indent=4)
                self.root.destroy()  # Actually close the window
        except Exception as e:
            print(f"Failed to save file: {e}")

    def update_progress(self, i, elapsed_time):
        # Update progress label from the GUI thread
        total = len(self.image_paths)
        progress_text = f"{i+1}/{total}"
        if i+1==total:
            progress_text += f" (Finished in {int(elapsed_time)} seconds)"
        elif i>-1:
            seconds_per_image = elapsed_time / (i+1)
            if elapsed_time < 0:
                elapsed_time *= -1
                seconds_per_image = elapsed_time / (i+2)
            seconds_remaining = seconds_per_image * (total - i - 1)
            progress_text += f" ({self.format_remaining_time(seconds_remaining)} remaining)"

        self.root.after(0, self.imgproc_label.config, {"text":progress_text})
        
        remaining_images_text = "Remaining:"
        for image_path in self.image_paths[i+1:]:
            image_name = os.path.basename(image_path)
            remaining_images_text += f"\n|   {image_name}"
        self.root.after(0, self.output_text.config, {"state":"normal"})
        self.root.after(0, lambda: self.output_text.delete("1.0", "end"))
        self.root.after(0, lambda: self.output_text.insert("end", remaining_images_text))
        self.root.after(0, self.output_text.config, {"state":"disabled"})
        
    def format_remaining_time(self, seconds_remaining):
        seconds_remaining = int(round(seconds_remaining))
        if seconds_remaining < 60:
            return f"{seconds_remaining} second{'s' if seconds_remaining != 1 else ''}"
        else:
            minutes = seconds_remaining // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''}"

    def show_review_instructions(self):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")

        instructions = [
            ("Instructions for Image Review:\n", None),
            ("|   ", None),
            ("GREEN", "green"),
            (" = selected, ", None),
            ("RED", "red"),
            (" = deselected (excluded when generating dataset)\n", None),
            ("|   CLICK near a ", None),
            ("BLACK", "black"),
            (" dot to select/deselect its contour\n", None),
            ("|   Hold SHIFT to hide ", None),
            ("RED", "red"),
            (" contours\n", None),
            ("|   Hold SHIFT+SPACE to hide all contours\n", None),
            ("|   Use LEFT and RIGHT ARROW keys to change images\n", None),
            ("|   Press ESCAPE to save and quit", None)
        ]

        for text, color in instructions:
            if color:
                self.output_text.insert("end", text, color)
            else:
                self.output_text.insert("end", text)

        self.output_text.config(state="disabled")

        # Define styles
        self.output_text.tag_configure("green", foreground="green", font=("TkDefaultFont", 10, "bold"))
        self.output_text.tag_configure("red", foreground="red", font=("TkDefaultFont", 10, "bold"))
        self.output_text.tag_configure("black", foreground="black", font=("TkDefaultFont", 10, "bold"))

    def get_csv_lines(self, files, formatted_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")):
        data_lists = []
        out_imgs = []
        for file_path in files:
            try:
                with open(file_path, "rb") as f:
                    file_data = pickle.load(f)
            except FileNotFoundError:
                messagebox.showerror("Error", "No processed data file found.")
                return
            # contour_data: (ID, inner_contour, outer_contour, g_ratio, circularity, thickness, inner_diameter, outer_diameter)
            img_filename, out_img, contour_data, selected_states = [item[1] for item in file_data.items()]
            h, w = out_img.shape[:2]
            image_size = h*w
            area_correction_ratio = image_size / (3608 * 4096)
            linear_correction_ratio = math.sqrt(area_correction_ratio)
            data = []

            included_ids = [ID for ID, keep in selected_states.items() if keep]
            filtered_contour_data = [d for d in contour_data if d[0] in included_ids]
            reindexed_contour_data = [tuple([i+1]+list(datum)[1:]) for i,datum in enumerate(filtered_contour_data)]

            for ID, inner_contour, outer_contour, g_ratio, circularity, thickness, inner_diameter, outer_diameter in reindexed_contour_data:
                M = cv2.moments(inner_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                color = (0, 255, 0)
                cv2.drawContours(out_img, [inner_contour], -1, color, 2)
                cv2.drawContours(out_img, [outer_contour], -1, color, 2)
            for ID, inner_contour, outer_contour, g_ratio, circularity, thickness, inner_diameter, outer_diameter in reindexed_contour_data:
                M = cv2.moments(inner_contour)
                draw_scale = int(8*linear_correction_ratio)
                cx = int(M["m10"] / M["m00"]-15*draw_scale)
                cy = int(M["m01"] / M["m00"]+6*draw_scale)
                line_spacing = 6*draw_scale
                cv2.putText(out_img, f"#{ID}", (cx, cy-line_spacing), 
                            cv2.FONT_HERSHEY_PLAIN, draw_scale, (0, 0, 0), draw_scale, cv2.LINE_AA)
                cv2.putText(out_img, f"{g_ratio:.2f}", (cx, cy+line_spacing), 
                            cv2.FONT_HERSHEY_PLAIN, draw_scale, (255, 0, 0), draw_scale, cv2.LINE_AA)
                data.append((ID, float(g_ratio), float(circularity), float(inner_diameter), float(outer_diameter), float(thickness)))
            
            name, extension = img_filename.split(".")
            out_imgs.append((f"{name}_gratios_{formatted_datetime}.{extension}", out_img))
            data_lists.append((img_filename, data))
            

        csv_lines = []
        csv_lines.append(f"NOTE: All length measurements are in pixels.\n")
        csv_lines.append("\n")

        csv_lines.append(f"Image,Axons found\n")
        for filename,data in data_lists:
            csv_lines.append(f"{filename},{len(data)}\n")
        csv_lines.append(f"Total,{sum([len(data) for _,data in data_lists])}\n")
        csv_lines.append("\n")

        for filename,data in data_lists:
            csv_lines.append(f"{filename}\n")
            csv_lines.append("Axon #,G-ratio,Circularity,Inner diameter,Outer diameter,Myelin Thickness\n")
            for axon_id, gratio, circularity, inner_dia, outer_dia, thickness in data:
                csv_lines.append(f"{axon_id},{gratio:.4f},{circularity:.4f},{inner_dia:.4f},{outer_dia:.4f},{thickness:.4f}\n")
            csv_lines.append("\n")

        return out_imgs, csv_lines


























    def interactive_reviewer(self, data_path, display_scale=0.5, window_title="Review Segmentation"):
        try:
            with open(data_path, "rb") as f:
                file_data = pickle.load(f)
        except FileNotFoundError:
            messagebox.showerror("Error", "No processed data file found.")
            return
        
        # contour_data: (ID, inner_contour, outer_contour, g_ratio, circularity, thickness, inner_diameter, outer_diameter)
        img_filename, image, contour_data, selected_states = [item[1] for item in file_data.items()]

        render_data = []
        for datum in contour_data:
            M = cv2.moments(datum[1])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                render_data.append((datum[0], cx, cy, datum[1], datum[2]))

        h, w = image.shape[:2]
        image_size = h*w
        area_correction_ratio = image_size / (3608 * 4096)
        linear_correction_ratio = math.sqrt(area_correction_ratio)

        display_scale = 0.85*MONITOR.height/h

        preview_mode = False  # If True, hide excluded
        original_image_mode = False

        def draw():
            display_img = image.copy()
            for ID, cx, cy, inner_contour, outer_contour in render_data:
                if (preview_mode and not selected_states[ID]) or original_image_mode:
                    continue  # Hide excluded in preview
                color = (0, 255, 0) if selected_states[ID] else (0, 0, 255)
                cv2.drawContours(display_img, [inner_contour], -1, color, 2)
                cv2.drawContours(display_img, [outer_contour], -1, color, 2)
                cv2.circle(display_img, (cx, cy), 4, (0, 0, 0), -1)
                draw_scale = int(8*linear_correction_ratio)
                cx = int(cx-15*draw_scale)
                cv2.putText(display_img, f"#{ID}", (cx, cy), 
                            cv2.FONT_HERSHEY_PLAIN, draw_scale, (0, 0, 0), draw_scale, cv2.LINE_AA)
            resized = cv2.resize(display_img, (0, 0), fx=display_scale, fy=display_scale)
            return resized

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                closest_valid_contour = (-1, 99999999) # ID, distance
                x_full = int(x / display_scale)
                y_full = int(y / display_scale)
                for ID, cx, cy, inner_contour, *_ in render_data:
                    if preview_mode and not selected_states[ID]:
                        continue
                    radius = 0.5*cv2.arcLength(inner_contour, closed=True)/(2*math.pi)
                    distance = math.hypot(x_full - cx, y_full - cy) 
                    if distance < radius and distance < closest_valid_contour[1]:
                        closest_valid_contour = (ID, distance)
                closest_ID = closest_valid_contour[0]
                if closest_ID != -1:
                    selected_states[closest_ID] = not selected_states[closest_ID]

        cv2.namedWindow(window_title)
        cv2.moveWindow(window_title, int(MONITOR.width/2-display_scale*w/2), 0)
        cv2.setMouseCallback(window_title, on_mouse)

        start_time = time.time()
        exit_code = 0
        while True:
            preview_mode = keyboard.is_pressed('shift')
            original_image_mode = keyboard.is_pressed('shift') and keyboard.is_pressed('space')
            disp = draw()
            cv2.imshow(window_title, disp)

            elapsed_time = time.time() - start_time
            if elapsed_time < 0.25:
                continue

            # Quit condition
            keep_running = True
            key = cv2.waitKey(30) & 0xFF
            if key == 27: # Escape
                keep_running = False
            if keyboard.is_pressed('left'):
                keep_running = False
                exit_code -= 1
            if keyboard.is_pressed('right'):
                keep_running = False
                exit_code += 1
            if not keep_running:
                break

        cv2.destroyAllWindows()
        
        # Update data file
        file_data = {
            "img_filename": img_filename,
            "image_data": image,
            "contour_data": contour_data,
            "selected_states": selected_states
        }
        with open(data_path, "wb") as f:
            pickle.dump(file_data, f)
        
        return exit_code


















    def create_circular_kernel(self, radius):
        """Create a circular kernel mask"""
        size = 2 * radius + 1
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x*x + y*y <= radius*radius
        return mask.astype(np.float32)

    def convexness(self, contour, hull):
        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull)
        return contour_area/hull_area

    def process_image(self, input_image, thresh_val, radius_val, dilate, erode, min_size, max_size, convex_thresh, circ_thresh):
        h, w = input_image.shape
        image_size = input_image.shape[0]*input_image.shape[1]
        area_correction_ratio = image_size / (3608 * 4096)
        linear_correction_ratio = math.sqrt(area_correction_ratio)
        dilate = round(dilate * linear_correction_ratio)
        erode = round(erode * linear_correction_ratio)
        min_size = min_size * area_correction_ratio
        max_size = max_size * area_correction_ratio
        
        # Threshold image (binary)
        kernel = self.create_circular_kernel(radius_val)
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
            convex = self.convexness(c, hull)
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
        
        # # Create output color image for visualization
        # out_img = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
        
        # Create binary mask of all contours
        all_contours_mask = np.zeros_like(eroded)
        cv2.drawContours(all_contours_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

        # return all_contours_mask, []
        
        contour_data = []
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
            # cv2.drawContours(out_img, inner_contour + np.array([[[x_min, y_min]]]), -1, (0, 255, 0), draw_scale)
            # cv2.drawContours(out_img, outer_contour + np.array([[[x_min, y_min]]]), -1, (0, 255, 0), draw_scale)
            # M = cv2.moments(contour)
            # if M["m00"] != 0:
            #     cx = int(M["m10"] / M["m00"]-15*draw_scale)
            #     cy = int(M["m01"] / M["m00"]+6*draw_scale)
            #     line_spacing = 6*draw_scale
            #     cv2.putText(out_img, f"#{i+1}", (cx, cy-line_spacing), 
            #                 cv2.FONT_HERSHEY_PLAIN, draw_scale, (0, 0, 0), draw_scale, cv2.LINE_AA)
            #     cv2.putText(out_img, f"{g_ratio:.2f}", (cx, cy+line_spacing), 
            #                 cv2.FONT_HERSHEY_PLAIN, draw_scale, (255, 0, 0), draw_scale, cv2.LINE_AA)
                
            inner_diameter = 2*radius
            outer_diameter = inner_diameter + 2*thickness

            inner_contour += np.array([[[x_min, y_min]]])
            outer_contour += np.array([[[x_min, y_min]]])
            contour_data.append((i+1, inner_contour[0], outer_contour[0], g_ratio, circularity, thickness, inner_diameter, outer_diameter))
            # print(f"myelin thickness {thickness} | axon radius {radius} | g_ratio {g_ratio}")
        
        return contour_data #out_img, contour_data







if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()