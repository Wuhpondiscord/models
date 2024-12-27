
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser, scrolledtext
import threading
import time
import sys
import os
import logging
from screeninfo import get_monitors
import subprocess
from ultralytics import YOLO
import mouse_anywhere as mouse
from yolo_overlay import YOLOOverlay
import ctypes
import mss
from PIL import Image
from queue import Queue, Empty
import keyboard
import random
import torch  
import requests  

class DetectionBox(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("color", ctypes.c_uint32),
        ("label", ctypes.c_char * 50),
        ("lastSeen", ctypes.c_uint32),
        ("paused", ctypes.c_int)
    ]

class AimbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Overlay & Mouse Anywhere Integration")
        self.root.geometry("1000x800")

        self.root.configure(bg="#000000")  

        self.setup_logging()

        self.exit_hotkey = 'f9'
        self.exit_hotkey_var = tk.StringVar(value=self.exit_hotkey)

        try:
            mouse.initialize()
            mouse.set_config(
                strength=50,
                hold_time_ms=300,
                mouse_speed=5,
                easing_type=3,  
                smooth_movement=True
            )
            mouse.set_logging_level(2)
            self.logger.info("Mouse Anywhere initialized successfully.")
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize Mouse Anywhere: {e}")
            self.logger.error(f"Failed to initialize Mouse Anywhere: {e}")
            sys.exit(1)

        self.detection_queue = Queue()
        self.overlay = None
        self.classes = {}
        self.class_colors = {}
        self.class_priorities = {}

        self.stop_event = threading.Event()
        self.class_colors_lock = threading.Lock()
        self.running = False
        self.detection_thread = None
        self.aimbot_thread = None

        self.ads_delay = 0.2

        self.color_buttons = {}
        self.ignore_checkbuttons = {}
        self.ignore_vars = {}
        self.priority_spinboxes = {}
        self.priority_vars = {}

        self.exit_hotkey_handler = None
        self.f_key_handler = None

        self.enable_refined_movement = False

        self.enable_partial_capture = False
        self.partial_coords = (0, 0, 800, 600)  
        self.enable_downscale_capture = False
        self.downscale_factor = 2

        self.default_device = "GPU" if torch.cuda.is_available() else "CPU"

        self.configure_style()

        self.create_widgets()
        self.start_hotkey_listener()

    def setup_logging(self):
        logging.basicConfig(
            filename='aimbot_app.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger()
        self.logger.info("Logging initialized.")

    def configure_style(self):
        """
        Creates or updates a custom ttk style that enforces a black background,
        white text, and purple accent color throughout.
        """
        style = ttk.Style()

        style.theme_use('clam')

        background_color = "#000000"  
        foreground_color = "#FFFFFF"  
        accent_color = "#8A2BE2"      

        style.configure(
            "TFrame",
            background=background_color,
            foreground=foreground_color
        )

        style.configure(
            "Black.TLabelframe",
            background=background_color,
            foreground=foreground_color,
            borderwidth=0,
            relief="flat"  
        )
        style.configure(
            "Black.TLabelframe.Label",
            background=background_color,
            foreground=foreground_color
        )

        style.configure(
            "TLabel",
            background=background_color,
            foreground=foreground_color
        )

        style.configure(
            "TButton",
            background=accent_color,
            foreground="#FFFFFF",
            padding=6
        )
        style.map(
            "TButton",
            background=[
                ("active", "#6A1CBF"),   
                ("pressed", "#5E1898")
            ]
        )

        style.configure(
            "TCheckbutton",
            background=background_color,
            foreground=foreground_color
        )
        style.map(
            "TCheckbutton",
            background=[
                ("active", background_color),
                ("selected", background_color)
            ]
        )

        style.configure(
            "TCombobox",
            fieldbackground=background_color,
            foreground=foreground_color,
            background=background_color
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", background_color)],
            foreground=[("readonly", foreground_color)]
        )

        style.configure(
            "TSpinbox",
            fieldbackground=background_color,
            foreground=foreground_color,
            background=background_color
        )

        style.configure(
            "TEntry",
            fieldbackground=background_color,
            foreground=foreground_color
        )

        style.element_create('Purple.Horizontal.Scale.trough', 'from', 'clam')
        style.element_create('Purple.Horizontal.Scale.slider', 'from', 'clam')

        style.layout('Purple.Horizontal.TScale',
                     [('Horizontal.Scale.trough',
                       {'sticky': 'nswe'}),
                      ('Horizontal.Scale.slider',
                       {'side': 'left', 'sticky': ''})])

        style.configure('Purple.Horizontal.TScale',
                        background=accent_color,
                        troughcolor=accent_color)

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        self.controls_frame = ttk.Frame(self.notebook)
        self.settings_frame = ttk.Frame(self.notebook)
        self.logging_frame = ttk.Frame(self.notebook)

        self.help_frame = tk.Frame(self.notebook, bg="#000000")

        self.notebook.add(self.controls_frame, text='Controls')
        self.notebook.add(self.settings_frame, text='Settings')
        self.notebook.add(self.logging_frame, text='Logging')
        self.notebook.add(self.help_frame, text='Help')  

        self.create_controls_tab()
        self.create_settings_tab()
        self.create_logging_tab()
        self.create_help_tab()  

        self.status_var = tk.StringVar(value="Status: Idle")

        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#000000",
            fg="#FFFFFF",
            font=("Arial", 12)
        )
        self.status_label.pack(side="bottom", fill="x")

    def create_controls_tab(self):
        reminder_label = ttk.Label(
            self.controls_frame,
            text=(
                "Reminder: Some games (like Roblox) might not allow direct SetCursorPos in true fullscreen.\n"
                "Use windowed/borderless mode to avoid forced mouse resets."
            ),
            foreground="red",
        )
        reminder_label.pack(pady=10, padx=10)

        self.start_btn = ttk.Button(
            self.controls_frame,
            text="Start Aimbot",
            command=self.start_aimbot,
            state="disabled"
        )
        self.start_btn.pack(pady=20, padx=20, fill='x')

        self.stop_btn = ttk.Button(
            self.controls_frame,
            text="Stop Aimbot",
            command=self.stop_aimbot,
            state="disabled"
        )
        self.stop_btn.pack(pady=20, padx=20, fill='x')

        self.test_mouse_btn = ttk.Button(
            self.controls_frame,
            text="Test Mouse Move",
            command=self.test_mouse_move
        )
        self.test_mouse_btn.pack(pady=10, padx=20, fill='x')

    def create_settings_tab(self):

        yolo_group = ttk.LabelFrame(
            self.settings_frame,
            text="YOLO Overlay Settings",
            padding=(20, 10),
            style="Black.TLabelframe"
        )
        yolo_group.pack(fill='both', expand=True, padx=10, pady=10)

        yolo_inner_frame = ttk.Frame(yolo_group, style="Black.TLabelframe")
        yolo_inner_frame.pack(fill='both', expand=True, padx=10, pady=10)

        ttk.Label(yolo_inner_frame, text="Confidence Threshold:").grid(
            row=0, column=0, sticky='w', pady=5
        )
        self.conf_threshold_var = tk.DoubleVar(value=0.5)
        ttk.Scale(
            yolo_inner_frame,
            from_=0.0, to=1.0,
            orient='horizontal',
            variable=self.conf_threshold_var,
            command=self.update_conf_threshold,
            style='Purple.Horizontal.TScale'
        ).grid(row=0, column=1, pady=5, sticky='ew')
        yolo_inner_frame.columnconfigure(1, weight=1)

        ttk.Label(yolo_inner_frame, text="Maximum Detections:").grid(
            row=1, column=0, sticky='w', pady=5
        )
        self.max_detections_var = tk.IntVar(value=100)
        self.max_detections_scale = ttk.Scale(
            yolo_inner_frame, from_=10, to=200, orient='horizontal',
            variable=self.max_detections_var,
            command=self.update_max_detections,
            style='Purple.Horizontal.TScale'
        )
        self.max_detections_scale.grid(row=1, column=1, pady=5, sticky='ew')

        self.toggle_max_detections_var = tk.BooleanVar(value=True)
        self.toggle_max_detections_chk = ttk.Checkbutton(
            yolo_inner_frame,
            text="Enable Maximum Detections",
            variable=self.toggle_max_detections_var,
            command=self.toggle_max_detections
        )
        self.toggle_max_detections_chk.grid(row=1, column=2, padx=5, pady=5, sticky='w')

        ttk.Label(yolo_inner_frame, text="Easing Type:").grid(
            row=2, column=0, sticky='w', pady=5
        )
        self.easing_type_var = tk.StringVar(value="Sinusoidal")
        easing_options = ["Linear", "Quadratic", "Sinusoidal", "Cubic", "Exponential"]
        self.easing_combobox = ttk.Combobox(
            yolo_inner_frame,
            values=easing_options,
            textvariable=self.easing_type_var,
            state='readonly'
        )
        self.easing_combobox.current(2)
        self.easing_combobox.grid(row=2, column=1, pady=5, sticky='ew')
        ttk.Button(
            yolo_inner_frame, text="Apply Easing Type", command=self.apply_easing_type
        ).grid(row=2, column=2, padx=5, pady=5)

        ttk.Label(yolo_inner_frame, text="Select Monitor:").grid(
            row=3, column=0, sticky='w', pady=5
        )
        monitors = get_monitors()
        if not monitors:
            messagebox.showerror("Error", "No monitors detected.")
            self.logger.error("No monitors detected.")
            self.root.destroy()
            return
        monitor_names = [f"{m.name} ({m.width}x{m.height})" for m in monitors]
        self.monitor_dropdown = ttk.Combobox(yolo_inner_frame, values=monitor_names, state="readonly")
        if monitor_names:
            self.monitor_dropdown.current(0)
        self.monitor_dropdown.grid(row=3, column=1, pady=5, sticky='ew')

        ttk.Label(yolo_inner_frame, text="Selected Model:").grid(
            row=4, column=0, sticky='w', pady=5
        )
        self.selected_model_var = tk.StringVar(value="No model selected.")
        self.selected_model_label = ttk.Label(
            yolo_inner_frame,
            textvariable=self.selected_model_var,
            wraplength=300
        )
        self.selected_model_label.grid(row=4, column=1, pady=5, sticky='w')

        ttk.Label(yolo_inner_frame, text="Presets:").grid(
            row=5, column=0, sticky='w', pady=5
        )
        preset_frame = ttk.Frame(yolo_inner_frame, style="Black.TLabelframe")
        preset_frame.grid(row=5, column=1, pady=5, sticky='w')
        ttk.Button(preset_frame, text="Default", command=lambda: self.apply_preset(1)).pack(side='left', padx=5)
        ttk.Button(preset_frame, text="Fast", command=lambda: self.apply_preset(2)).pack(side='left', padx=5)
        ttk.Button(preset_frame, text="Smooth", command=lambda: self.apply_preset(3)).pack(side='left', padx=5)

        ttk.Label(yolo_inner_frame, text="Detection FPS:").grid(
            row=6, column=0, sticky='w', pady=5
        )
        self.fps_var = tk.IntVar(value=5)  
        self.fps_scale = ttk.Scale(
            yolo_inner_frame, from_=1, to=30, orient='horizontal',
            variable=self.fps_var, command=self.update_fps,
            style='Purple.Horizontal.TScale'
        )
        self.fps_scale.grid(row=6, column=1, pady=5, sticky='ew')

        partial_frame = ttk.LabelFrame(
            yolo_inner_frame,
            text="Partial/Downscale Capture",
            padding=(10, 5),
            style="Black.TLabelframe"
        )
        partial_frame.grid(row=7, column=0, columnspan=3, padx=5, pady=5, sticky='ew')

        self.partial_capture_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            partial_frame,
            text="Enable Partial Capture",
            variable=self.partial_capture_var,
            command=self.toggle_partial_capture
        ).pack(anchor='w', pady=2)

        self.downscale_capture_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            partial_frame,
            text="Enable Downscale Capture",
            variable=self.downscale_capture_var,
            command=self.toggle_downscale_capture
        ).pack(anchor='w', pady=2)

        bbox_colors_group = ttk.LabelFrame(
            yolo_inner_frame,
            text="Bounding Box Colors",
            padding=(10, 5),
            style="Black.TLabelframe"
        )
        bbox_colors_group.grid(row=8, column=0, columnspan=3, padx=5, pady=5, sticky='ew')
        bbox_colors_group.columnconfigure(1, weight=1)
        self.bbox_colors_frame = bbox_colors_group

        ignore_group = ttk.LabelFrame(
            yolo_inner_frame,
            text="Ignore Classes",
            padding=(10, 5),
            style="Black.TLabelframe"
        )
        ignore_group.grid(row=9, column=0, columnspan=3, padx=5, pady=5, sticky='ew')
        self.ignore_frame = ignore_group

        priority_group = ttk.LabelFrame(
            yolo_inner_frame,
            text="Class Priority",
            padding=(10, 5),
            style="Black.TLabelframe"
        )
        priority_group.grid(row=10, column=0, columnspan=3, padx=5, pady=5, sticky='ew')
        self.priority_frame = priority_group

        ttk.Button(yolo_inner_frame, text="Select YOLOv8 Model", command=self.select_model).grid(
            row=11, column=0, columnspan=3, pady=10, sticky='ew'
        )

        mouse_group = ttk.LabelFrame(
            self.settings_frame,
            text="Mouse Anywhere Settings",
            padding=(20, 10),
            style="Black.TLabelframe"
        )
        mouse_group.pack(fill='both', expand=True, padx=10, pady=10)

        mouse_inner_frame = ttk.Frame(mouse_group, style="Black.TLabelframe")
        mouse_inner_frame.pack(fill='both', expand=True, padx=10, pady=10)

        ttk.Label(mouse_inner_frame, text="Strength:").grid(
            row=0, column=0, sticky='w', pady=5
        )
        self.strength_var = tk.IntVar(value=50)
        ttk.Scale(
            mouse_inner_frame, from_=1, to=100, orient='horizontal',
            variable=self.strength_var, command=self.update_strength,
            style='Purple.Horizontal.TScale'
        ).grid(row=0, column=1, pady=5, sticky='ew')

        ttk.Label(mouse_inner_frame, text="Hold Time (ms):").grid(
            row=1, column=0, sticky='w', pady=5
        )
        self.hold_time_var = tk.IntVar(value=300)
        ttk.Scale(
            mouse_inner_frame, from_=50, to=1000, orient='horizontal',
            variable=self.hold_time_var, command=self.update_hold_time,
            style='Purple.Horizontal.TScale'
        ).grid(row=1, column=1, pady=5, sticky='ew')

        ttk.Label(mouse_inner_frame, text="Mouse Speed (ms/step):").grid(
            row=2, column=0, sticky='w', pady=5
        )
        self.mouse_speed_var = tk.IntVar(value=5)
        ttk.Scale(
            mouse_inner_frame, from_=1, to=20, orient='horizontal',
            variable=self.mouse_speed_var, command=self.update_mouse_speed,
            style='Purple.Horizontal.TScale'
        ).grid(row=2, column=1, pady=5, sticky='ew')

        self.smooth_movement_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            mouse_inner_frame,
            text="Enable Smooth Movement",
            variable=self.smooth_movement_var,
            command=self.toggle_smooth_movement
        ).grid(row=3, column=0, columnspan=2, pady=5, sticky='w')

        ads_group = ttk.LabelFrame(
            mouse_inner_frame,
            text="ADS Settings",
            padding=(10, 5),
            style="Black.TLabelframe"
        )
        ads_group.grid(row=4, column=0, columnspan=2, padx=5, pady=10, sticky='ew')

        self.ads_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ads_group,
            text="Enable ADS (Right-Click)",
            variable=self.ads_var,
            command=self.toggle_ads
        ).pack(anchor='w', pady=5)

        ttk.Label(ads_group, text="ADS to Shoot Delay (ms):").pack(
            anchor='w', pady=5
        )
        self.ads_delay_var = tk.IntVar(value=200)
        self.ads_delay_entry = ttk.Entry(ads_group, textvariable=self.ads_delay_var, width=10)
        self.ads_delay_entry.pack(anchor='w', pady=5)
        ttk.Button(ads_group, text="Set Delay", command=self.set_ads_delay).pack(anchor='w', pady=5)

        self.refined_movement_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            mouse_inner_frame,
            text="Enable Refined Movement (Anti-Cheat Noise)",
            variable=self.refined_movement_var,
            command=self.toggle_refined_movement
        ).grid(row=5, column=0, columnspan=2, pady=5, sticky='w')

        device_group = ttk.LabelFrame(
            mouse_inner_frame,
            text="Compute Device",
            padding=(10, 5),
            style="Black.TLabelframe"
        )
        device_group.grid(row=6, column=0, columnspan=2, padx=5, pady=10, sticky='ew')
        ttk.Label(device_group, text="Device:").pack(anchor='w')

        self.device_var = tk.StringVar(value=self.default_device)  
        device_dropdown = ttk.Combobox(device_group, values=["CPU", "GPU"], textvariable=self.device_var, state="readonly")
        device_dropdown.pack(anchor='w', pady=5)

        ttk.Button(device_group, text="Apply Device", command=self.apply_compute_device).pack(anchor='w', pady=5)

        mouse_inner_frame.columnconfigure(1, weight=1)

        hotkey_group = ttk.LabelFrame(
            self.settings_frame,
            text="Kill Hotkey Settings",
            padding=(20, 10),
            style="Black.TLabelframe"
        )
        hotkey_group.pack(fill='both', expand=True, padx=10, pady=10)

        ttk.Label(hotkey_group, text="Current Exit Hotkey:").grid(
            row=0, column=0, sticky='w', pady=5
        )
        self.current_exit_hotkey_display = ttk.Label(
            hotkey_group,
            textvariable=self.exit_hotkey_var,
            foreground="yellow",
            font=("Arial", 12, "bold"),
            background="#000000"
        )
        self.current_exit_hotkey_display.grid(row=0, column=1, pady=5, sticky='w')

        ttk.Button(
            hotkey_group,
            text="Set New Exit Hotkey",
            command=self.set_new_exit_hotkey
        ).grid(row=1, column=0, columnspan=2, pady=10, sticky='ew')

    def create_logging_tab(self):
        ttk.Label(
            self.logging_frame,
            text="Set Logging Level:"
        ).grid(row=0, column=0, sticky='w', pady=10, padx=10)
        self.log_level_var = tk.StringVar(value="Info")
        log_levels = ["None", "Error", "Info", "Debug"]
        self.log_dropdown = ttk.Combobox(
            self.logging_frame,
            values=log_levels,
            textvariable=self.log_level_var,
            state='readonly'
        )
        self.log_dropdown.current(2)
        self.log_dropdown.grid(row=0, column=1, pady=10, padx=10, sticky='ew')
        ttk.Button(
            self.logging_frame,
            text="Apply Log Level",
            command=self.apply_log_level
        ).grid(row=0, column=2, pady=10, padx=10)

        ttk.Button(
            self.logging_frame,
            text="View Log",
            command=self.view_log
        ).grid(row=1, column=0, columnspan=3, pady=10, padx=10, sticky='ew')

        self.logging_frame.columnconfigure(1, weight=1)

    def create_help_tab(self):
        help_label = tk.Label(
            self.help_frame,
            text="Aimbot Help",
            font=("Arial", 16, "bold"),
            bg="#000000",
            fg="#FFFFFF"
        )
        help_label.pack(pady=10)

        self.help_text = scrolledtext.ScrolledText(
            self.help_frame,
            wrap=tk.WORD,
            width=100,
            height=40,
            state='disabled',
            bg="#000000",
            fg="#FFFFFF",
            font=("Consolas", 10)
        )
        self.help_text.pack(padx=10, pady=10, fill='both', expand=True)

        threading.Thread(target=self.load_help_content, daemon=True).start()

    def load_help_content(self):
        help_url = "https://raw.githubusercontent.com/Wuhpondiscord/models/refs/heads/main/examples/python/aimbot-help.txt"
        try:
            response = requests.get(help_url)
            response.raise_for_status()
            help_content = response.text
        except requests.RequestException as e:
            help_content = f"Failed to load help content:\n{e}"
            self.logger.error(f"Help content load error: {e}")

        self.help_text.config(state='normal')
        self.help_text.insert(tk.END, help_content)
        self.help_text.config(state='disabled')

    def toggle_partial_capture(self):
        self.enable_partial_capture = self.partial_capture_var.get()
        if self.enable_partial_capture:
            messagebox.showinfo(
                "Partial Capture",
                "Capturing only a sub-region of the screen. Adjust partial_coords in code if needed."
            )
            self.logger.info("Partial screen capture enabled.")
        else:
            self.logger.info("Partial screen capture disabled.")

    def toggle_downscale_capture(self):
        self.enable_downscale_capture = self.downscale_capture_var.get()
        if self.enable_downscale_capture:
            messagebox.showinfo(
                "Downscale Capture",
                "Downscaling captured image before YOLO inference."
            )
            self.logger.info("Downscale capture enabled.")
        else:
            self.logger.info("Downscale capture disabled.")

    def apply_partial_or_downscale(self, img):
        """If partial capture is enabled, crop. If downscale is enabled, resize."""
        if self.enable_partial_capture:
            x, y, w, h = self.partial_coords
            img = img.crop((x, y, x + w, y + h))

        if self.enable_downscale_capture and self.downscale_factor > 1:
            new_w = img.width // self.downscale_factor
            new_h = img.height // self.downscale_factor
            if new_w <= 0 or new_h <= 0:
                new_w, new_h = img.width, img.height  
            img = img.resize((new_w, new_h), Image.BILINEAR)

        return img

    def select_model_path(self):
        model_path = filedialog.askopenfilename(
            title="Select YOLOv8 Model",
            filetypes=[("PyTorch Model", "*.pt")]
        )
        if model_path:
            self.logger.debug(f"Model selected: {model_path}")
            return model_path
        else:
            self.logger.debug("No model selected.")
            return None

    def select_model(self):
        model_path = self.select_model_path()
        if model_path:
            try:
                if self.running:
                    self.stop_aimbot()
                if self.overlay:
                    self.overlay.stop()
                    self.logger.info("Stopped existing YOLO Overlay for model change.")

                monitor_index = self.monitor_dropdown.current()
                available_monitors = get_monitors()
                if monitor_index >= len(available_monitors) or monitor_index < 0:
                    self.logger.warning(
                        f"Invalid monitor index {monitor_index}. Defaulting to primary (0)."
                    )
                    monitor_index = 0

                self.overlay = YOLOOverlay(
                    model_path=model_path,
                    dll_path=None,
                    max_detections=self.max_detections_var.get() if self.toggle_max_detections_var.get() else None,
                    conf_threshold=self.conf_threshold_var.get(),
                    monitor_index=monitor_index
                )
                self.logger.info(f"YOLO Overlay re-initialized with: {model_path}")
                messagebox.showinfo("Model Selected", f"Loaded YOLOv8: {model_path}")

                self.apply_compute_device(force=True)

                if hasattr(self.overlay, 'model') and hasattr(self.overlay.model, 'names'):
                    self.classes = self.overlay.model.names
                    if not self.classes:
                        raise ValueError("Model has no class names.")
                    self.logger.info(f"Classes: {self.classes}")
                else:
                    self.classes = {}
                    self.logger.warning("No 'model.names' attribute found.")
                    messagebox.showwarning("Warning", "Could not retrieve class names from YOLO model.")

                self.class_colors = {cls.lower(): "#00FF00" for cls in self.classes.values()}
                self.class_priorities = {cls.lower(): 1 for cls in self.classes.values()}

                self.update_selected_model_label(model_path)
                self.create_class_controls()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
                self.logger.error(f"Load model error: {e}")

    def update_selected_model_label(self, model_path):
        self.selected_model_var.set(model_path)
        self.logger.debug(f"Updated selected model label: {model_path}")
        self.start_btn.config(state="normal")
        self.create_class_controls()

    def create_class_controls(self):

        if hasattr(self, 'bbox_colors_frame') and self.bbox_colors_frame:
            for btn in self.color_buttons.values():
                btn.destroy()
            self.color_buttons.clear()
            for cls in self.classes.values():
                cls_lower = cls.lower()
                ttk.Label(
                    self.bbox_colors_frame,
                    text=f"{cls.capitalize()}:"
                ).pack(anchor='w', pady=2)
                self.class_colors.setdefault(cls_lower, "#00FF00")
                btn = ttk.Button(
                    self.bbox_colors_frame,
                    text="Choose Color",
                    command=lambda c=cls_lower: self.choose_color(c)
                )
                btn.pack(anchor='w', pady=2, padx=5)
                self.color_buttons[cls_lower] = btn

        if hasattr(self, 'ignore_frame') and self.ignore_frame:
            for chk in self.ignore_checkbuttons.values():
                chk.destroy()
            self.ignore_checkbuttons.clear()
            for cls in self.classes.values():
                cls_lower = cls.lower()
                var = tk.BooleanVar(value=False)
                chk = ttk.Checkbutton(self.ignore_frame, text=cls.capitalize(), variable=var)
                chk.pack(side='left', padx=5)
                self.ignore_vars[cls_lower] = var
                self.ignore_checkbuttons[cls_lower] = chk

        if hasattr(self, 'priority_frame') and self.priority_frame:
            for spin in self.priority_spinboxes.values():
                spin.destroy()
            self.priority_spinboxes.clear()
            for cls in self.classes.values():
                cls_lower = cls.lower()
                ttk.Label(self.priority_frame, text=f"{cls.capitalize()}:").pack(side='left', padx=5)
                var = tk.IntVar(value=1)
                spin = ttk.Spinbox(self.priority_frame, from_=1, to=10, width=5, textvariable=var)
                spin.pack(side='left', padx=5)
                self.priority_vars[cls_lower] = var
                self.priority_spinboxes[cls_lower] = spin

    def refresh_class_controls(self):
        for widget in self.settings_frame.winfo_children():
            if isinstance(widget, ttk.LabelFrame) and widget['text'] == "YOLO Overlay Settings":
                widget.destroy()
        self.create_settings_tab()

    def apply_compute_device(self, force=False):
        device_choice = self.device_var.get()
        if not self.overlay or not hasattr(self.overlay, 'model'):
            if force:
                self.logger.warning("Overlay not loaded yet, but device choice applied on next model load.")
            return

        try:
            if device_choice == "GPU":
                if torch.cuda.is_available():
                    self.overlay.model.to('cuda')
                    self.logger.info("Moved model to GPU (CUDA).")
                    messagebox.showinfo("Compute Device", "Using GPU (CUDA).")
                else:
                    self.logger.warning("GPU selected but no CUDA available. Falling back to CPU.")
                    self.overlay.model.to('cpu')
                    messagebox.showinfo("Compute Device", "CUDA not found. Falling back to CPU.")
            else:
                self.overlay.model.to('cpu')
                self.logger.info("Using CPU for YOLO inference.")
                messagebox.showinfo("Compute Device", "Using CPU.")
        except Exception as e:
            self.logger.error(f"Failed to set compute device: {e}")
            messagebox.showerror("Compute Device Error", f"Error setting device: {e}")

    def start_aimbot(self):
        if not self.running:
            if not self.overlay:
                messagebox.showerror("Error", "No YOLOv8 model loaded. Please select a model first.")
                self.logger.error("Cannot start—no model.")
                return

            self.ignored_classes = [cls for cls, var in self.ignore_vars.items() if var.get()]
            self.class_priority = {cls: var.get() for cls, var in self.priority_vars.items()}
            self.class_priority_sorted = sorted(self.class_priority.items(), key=lambda item: item[1], reverse=True)

            if self.ignored_classes:
                self.logger.info(f"Ignored Classes: {self.ignored_classes}")
            else:
                self.logger.info("No classes ignored.")

            self.running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_var.set("Status: Aimbot Running")
            self.logger.info("Aimbot started.")
            self.stop_event.clear()

            self.detection_thread = threading.Thread(
                target=process_detections,
                args=(
                    self.overlay.model,
                    self.overlay.overlay_dll,
                    self.overlay.monitor_width,
                    self.overlay.monitor_height,
                    self.overlay.id_gen,
                    self.overlay.conf_threshold,
                    self.detection_queue,
                    self.logger,
                    self.class_colors,
                    self.class_colors_lock,
                    self.stop_event,
                    self.apply_partial_or_downscale,

                    self.fps_var
                ),
                daemon=True
            )
            self.detection_thread.start()

            self.aimbot_thread = threading.Thread(
                target=self.aimbot_loop,
                daemon=True
            )
            self.aimbot_thread.start()

    def stop_aimbot(self):
        if self.running:
            self.running = False
            self.stop_event.set()

            if self.detection_thread:
                self.detection_thread.join()
            if self.aimbot_thread:
                self.aimbot_thread.join()

            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.status_var.set("Status: Aimbot Stopped")
            self.logger.info("Aimbot stopped.")

    def aimbot_loop(self):
        """Consumes detection data from the queue and performs mouse moves/clicks."""
        try:
            while self.running:
                try:

                    detections = self.detection_queue.get(timeout=0.05)
                    if not detections:
                        continue

                    detections = [
                        det for det in detections
                        if det.label.decode('utf-8').lower() not in self.ignored_classes
                    ]
                    if not detections:
                        continue

                    detections_sorted = sorted(
                        detections,
                        key=lambda det: self.class_priority.get(det.label.decode('utf-8').lower(), 1),
                        reverse=True
                    )

                    target = detections_sorted[0]
                    x, y, w, h = target.x, target.y, target.width, target.height
                    center_x = x + w // 2
                    center_y = y + h // 2

                    if self.enable_refined_movement:
                        offset_x = random.randint(-2, 2)
                        offset_y = random.randint(-2, 2)
                        center_x += offset_x
                        center_y += offset_y
                        self.logger.debug(f"Refined offset: ({offset_x}, {offset_y})")

                    mouse.set_cursor_abs(center_x, center_y)
                    self.logger.debug(f"Moved mouse to ({center_x}, {center_y}).")

                    if self.ads_var.get():
                        mouse.click(2)
                        self.logger.debug("Performed ADS (Right Click).")
                        time.sleep(getattr(self, 'ads_delay', 0.2))

                    mouse.click(1)
                    self.logger.debug("Left mouse click fired.")

                except Empty:
                    continue

                time.sleep(0.001)
        except Exception as e:
            messagebox.showerror("Aimbot Error", f"An error occurred: {e}")
            self.logger.error(f"Aimbot loop error: {e}")
            self.stop_aimbot()

    def update_conf_threshold(self, event):
        try:
            conf_threshold = self.conf_threshold_var.get()
            if self.overlay:
                self.overlay.conf_threshold = conf_threshold
                self.logger.info(f"Confidence threshold -> {conf_threshold}")
        except Exception as e:
            self.logger.error(f"Update conf threshold error: {e}")

    def update_max_detections(self, event):
        try:
            if self.toggle_max_detections_var.get():
                max_detections = self.max_detections_var.get()
                if self.overlay:
                    self.overlay.max_detections = max_detections
                self.logger.info(f"Max detections -> {max_detections}")
            else:
                if self.overlay:
                    self.overlay.max_detections = None
                self.logger.info("Max detections disabled")
        except Exception as e:
            self.logger.error(f"Update max detections error: {e}")

    def toggle_max_detections(self):
        if not self.toggle_max_detections_var.get():
            self.max_detections_scale.state(['disabled'])
            try:
                if self.overlay:
                    self.overlay.max_detections = None
                self.logger.info("Max detections disabled.")
            except Exception as e:
                self.logger.error(f"Disable max detection error: {e}")
        else:
            self.max_detections_scale.state(['!disabled'])
            try:
                max_detections = self.max_detections_var.get()
                if self.overlay:
                    self.overlay.max_detections = max_detections
                self.logger.info(f"Max detections enabled: {max_detections}")
            except Exception as e:
                self.logger.error(f"Enable max detection error: {e}")

    def apply_easing_type(self):
        easing_map = {
            "Linear": 1,
            "Quadratic": 2,
            "Sinusoidal": 3,
            "Cubic": 4,
            "Exponential": 5
        }
        easing_type = easing_map.get(self.easing_type_var.get(), 3)
        try:
            mouse.set_config(
                strength=self.strength_var.get(),
                hold_time_ms=self.hold_time_var.get(),
                mouse_speed=self.mouse_speed_var.get(),
                easing_type=easing_type,
                smooth_movement=self.smooth_movement_var.get()
            )
            self.logger.info(f"Easing -> {self.easing_type_var.get()}")
            messagebox.showinfo("Easing Type", f"Set to {self.easing_type_var.get()}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set easing type: {e}")
            self.logger.error(f"Easing type error: {e}")

    def apply_preset(self, preset_type):
        try:
            mouse.apply_preset(preset_type)
            presets = {1: "Default", 2: "Fast", 3: "Smooth"}
            preset_name = presets.get(preset_type, "Unknown")
            self.logger.info(f"Applied preset: {preset_name}")
            messagebox.showinfo("Preset", f"Applied {preset_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply preset: {e}")
            self.logger.error(f"Preset error: {e}")

    def update_strength(self, event):
        try:
            self.set_mouse_config()
        except Exception as e:
            self.logger.error(f"Update strength error: {e}")

    def update_hold_time(self, event):
        try:
            self.set_mouse_config()
        except Exception as e:
            self.logger.error(f"Update hold time error: {e}")

    def update_mouse_speed(self, event):
        try:
            self.set_mouse_config()
        except Exception as e:
            self.logger.error(f"Update mouse speed error: {e}")

    def set_mouse_config(self):
        strength = self.strength_var.get()
        hold_time = self.hold_time_var.get()
        mouse_speed = self.mouse_speed_var.get()
        easing_type = self.get_easing_type()
        smooth = self.smooth_movement_var.get()
        mouse.set_config(
            strength=strength,
            hold_time_ms=hold_time,
            mouse_speed=mouse_speed,
            easing_type=easing_type,
            smooth_movement=smooth
        )
        self.logger.info(
            f"Mouse config -> strength={strength}, hold={hold_time}, speed={mouse_speed}, "
            f"easing={easing_type}, smooth={smooth}"
        )

    def get_easing_type(self):
        easing_map = {
            "Linear": 1,
            "Quadratic": 2,
            "Sinusoidal": 3,
            "Cubic": 4,
            "Exponential": 5
        }
        return easing_map.get(self.easing_type_var.get(), 3)

    def toggle_smooth_movement(self):
        try:
            self.set_mouse_config()
            status = "enabled" if self.smooth_movement_var.get() else "disabled"
            self.logger.info(f"Smooth movement {status}")
            messagebox.showinfo("Smooth Movement", f"{status.capitalize()}")
        except Exception as e:
            self.logger.error(f"Toggle smooth movement error: {e}")

    def toggle_ads(self):
        try:
            ads_enabled = self.ads_var.get()
            if ads_enabled:
                self.logger.info("ADS enabled.")
                messagebox.showinfo("ADS", "Right-click ADS enabled.")
            else:
                self.logger.info("ADS disabled.")
                messagebox.showinfo("ADS", "ADS disabled.")
        except Exception as e:
            self.logger.error(f"Toggle ADS error: {e}")

    def set_ads_delay(self):
        try:
            delay = self.ads_delay_var.get()
            if delay < 0:
                raise ValueError("Delay cannot be negative.")
            self.ads_delay = delay / 1000.0
            self.logger.info(f"ADS delay -> {delay} ms")
            messagebox.showinfo("ADS Delay", f"Set to {delay} ms.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set ADS delay: {e}")
            self.logger.error(f"Set ADS delay error: {e}")

    def toggle_refined_movement(self):
        self.enable_refined_movement = self.refined_movement_var.get()
        if self.enable_refined_movement:
            msg = (
                "Refined Movement: small random offsets to reduce anti-cheat detection.\n"
                "Now enabled."
            )
            messagebox.showinfo("Refined Movement", msg)
            self.logger.info("Refined Movement on.")
        else:
            self.logger.info("Refined Movement off.")

    def choose_color(self, cls_lower):
        color_code = colorchooser.askcolor(title=f"Choose color for {cls_lower.capitalize()}")
        if color_code and color_code[1]:
            with self.class_colors_lock:
                self.class_colors[cls_lower] = color_code[1]
            self.logger.info(f"Color for '{cls_lower}' -> {color_code[1]}")
            messagebox.showinfo("Color", f"Color set to {color_code[1]}")

    def update_fps(self, event):
        new_fps = self.fps_var.get()
        self.logger.info(f"Updated Detection FPS -> {new_fps}")

    def start_hotkey_listener(self):
        try:
            self.f_key_handler = keyboard.add_hotkey('f', self.on_f_key)
            keyboard.add_hotkey('g', lambda: self.preset_hotkey_switch(2))
            keyboard.add_hotkey('h', lambda: self.preset_hotkey_switch(3))

            self.exit_hotkey_handler = keyboard.add_hotkey(self.exit_hotkey, self.exit_application)
            formatted_exit_hotkey = self.exit_hotkey.upper()
            self.logger.info(f"Hotkeys active. Press {formatted_exit_hotkey} to exit.")
            print(f"Hotkeys active. Press {formatted_exit_hotkey} to exit.")
        except Exception as e:
            messagebox.showerror("Hotkey Error", f"Failed setting up hotkeys: {e}")
            self.logger.error(f"Hotkey error: {e}")

    def on_f_key(self):
        print("F key pressed")
        self.logger.info("F key pressed globally.")

    def preset_hotkey_switch(self, preset_type):
        try:
            mouse.apply_preset(preset_type)
            self.logger.info(f"Preset switched (hotkey): {preset_type}")
        except Exception as e:
            self.logger.error(f"Preset hotkey switch error: {e}")

    def exit_application(self):
        print(f"Exit hotkey '{self.exit_hotkey.upper()}' triggered. Exiting.")
        self.logger.info(f"Exit hotkey '{self.exit_hotkey.upper()}' triggered. Exiting.")
        self.root.after(0, self.safe_exit)

    def set_new_exit_hotkey(self):
        self.hotkey_window = tk.Toplevel(self.root)
        self.hotkey_window.title("Set New Exit Hotkey")
        self.hotkey_window.geometry("300x150")
        self.hotkey_window.grab_set()

        ttk.Label(self.hotkey_window, text="Press the new exit hotkey combination:", padding=10).pack()
        self.new_hotkey_var = tk.StringVar(value="Press keys...")
        self.new_hotkey_entry = ttk.Entry(
            self.hotkey_window,
            textvariable=self.new_hotkey_var,
            state='readonly',
            justify='center',
            font=("Arial", 12)
        )
        self.new_hotkey_entry.pack(pady=10, padx=20, fill='x')
        self.hotkey_pressed = set()

        def on_key_press(event):
            key = event.keysym.lower()
            if key == 'escape':
                self.hotkey_window.destroy()
                return
            self.hotkey_pressed.add(key)
            hotkey_str = '+'.join(sorted(self.hotkey_pressed))
            self.new_hotkey_var.set(hotkey_str)

        self.hotkey_window.bind('<KeyPress>', on_key_press)
        self.hotkey_window.focus_set()

        def finalize_hotkey():
            if self.hotkey_pressed:
                hotkey_parts = [self.key_to_str(k) for k in self.hotkey_pressed]
                new_hotkey = '+'.join(sorted(hotkey_parts))
                try:
                    keyboard.parse_hotkey(new_hotkey)
                except (keyboard.InvalidHotkeyError, ValueError) as e:
                    messagebox.showerror("Invalid Hotkey", f"Hotkey format invalid: {e}")
                    self.logger.error(f"Invalid hotkey: {new_hotkey} - {e}")
                    return

                if self.exit_hotkey_handler:
                    keyboard.remove_hotkey(self.exit_hotkey_handler)

                self.exit_hotkey_handler = keyboard.add_hotkey(new_hotkey, self.exit_application)
                self.exit_hotkey = new_hotkey
                self.exit_hotkey_var.set(self.exit_hotkey)
                self.logger.info(f"New exit hotkey -> {self.exit_hotkey}")
                messagebox.showinfo("Hotkey Set", f"New exit hotkey -> {self.exit_hotkey}")
            self.hotkey_window.destroy()

        ttk.Button(self.hotkey_window, text="Set Hotkey", command=finalize_hotkey).pack(pady=10)

    def key_to_str(self, key):
        return key.upper()

    def apply_log_level(self):
        log_level_map = {
            "None": 0,
            "Error": 1,
            "Info": 2,
            "Debug": 3
        }
        level = log_level_map.get(self.log_level_var.get(), 2)
        try:
            mouse.set_logging_level(level)
            self.logger.info(f"Mouse log level -> {self.log_level_var.get()}")
            messagebox.showinfo("Logging Level", f"Set to {self.log_level_var.get()}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set logging level: {e}")
            self.logger.error(f"Set logging level error: {e}")

    def view_log(self):
        log_file = 'aimbot_app.log'
        if os.path.exists(log_file):
            try:
                if sys.platform.startswith('darwin'):
                    subprocess.call(['open', log_file])
                elif os.name == 'nt':
                    os.startfile(log_file)
                elif os.name == 'posix':
                    subprocess.call(['xdg-open', log_file])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open log: {e}")
                self.logger.error(f"Open log error: {e}")
        else:
            messagebox.showinfo("Log File", "No log file found.")

    def test_mouse_move(self):
        try:
            mouse.set_cursor_abs(100, 100)
            self.logger.info("Mouse moved to (100,100) for test.")
            messagebox.showinfo("Mouse Test", "Moved to (100,100).")
        except Exception as e:
            messagebox.showerror("Test Error", f"Mouse move failed: {e}")
            self.logger.error(f"Test mouse move error: {e}")

    def safe_exit(self):
        self.running = False
        self.stop_event.set()

        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join()
        if self.aimbot_thread and self.aimbot_thread.is_alive():
            self.aimbot_thread.join()

        try:
            if self.overlay:
                self.overlay.stop()
                self.logger.info("Overlay stopped.")
        except Exception as e:
            self.logger.error(f"Overlay stop error: {e}")

        try:
            mouse.mouse_shutdown()
            self.logger.info("Mouse Anywhere shutdown.")
        except Exception as e:
            self.logger.error(f"Mouse shutdown error: {e}")

        try:
            keyboard.unhook_all_hotkeys()
            self.logger.info("All hotkeys unhooked.")
        except Exception as e:
            self.logger.error(f"Unhook hotkeys error: {e}")

        self.root.quit()
        self.root.destroy()

    def on_close(self):
        if messagebox.askokcancel("Quit", "Really quit?"):
            self.safe_exit()

def process_detections(
    model,
    overlay_dll,
    monitor_width,
    monitor_height,
    id_gen,
    conf_threshold,
    detection_queue,
    logger,
    class_colors,
    class_colors_lock,
    stop_event,
    partial_downscale_func=None,

    fps_var=None
):
    with mss.mss() as sct:
        while not stop_event.is_set():
            start_time = time.time()

            try:
                screenshot_box = {
                    "left": 0,
                    "top": 0,
                    "width": monitor_width,
                    "height": monitor_height
                }

                screenshot = sct.grab(screenshot_box)
                img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

                if partial_downscale_func:
                    img = partial_downscale_func(img)

                results = model(img, verbose=False)

                detection_boxes = []
                current_tick = get_tick_count()

                for det in results[0].boxes:
                    if det.conf < conf_threshold:
                        continue

                    try:
                        x1, y1, x2, y2 = map(int, det.xyxy[0])
                    except Exception as e:
                        logger.error(f"Parsing bbox coords error: {e}")
                        continue

                    width = x2 - x1
                    height = y2 - y1

                    try:
                        cls_id = int(det.cls[0])
                        label = model.names[cls_id] if cls_id < len(model.names) else "Unknown"
                    except Exception as e:
                        logger.error(f"Class name error: {e}")
                        label = "Unknown"

                    cls_lower = label.lower()
                    with class_colors_lock:
                        color_hex = class_colors.get(cls_lower, "#00FF00")
                    color = rgb_to_colorref(
                        int(color_hex[1:3], 16),
                        int(color_hex[3:5], 16),
                        int(color_hex[5:7], 16)
                    )

                    unique_id = next(id_gen)

                    sanitized_x, sanitized_y, sanitized_width, sanitized_height = sanitize_bounding_box(
                        x1, y1, x2, y2, img.width, img.height
                    )

                    detection_box = DetectionBox(
                        id=unique_id,
                        x=sanitized_x,
                        y=sanitized_y,
                        width=sanitized_width,
                        height=sanitized_height,
                        color=color,
                        label=encode_label(label),
                        lastSeen=current_tick,
                        paused=0
                    )
                    detection_boxes.append(detection_box)

                if detection_boxes:
                    DetectionArray = DetectionBox * len(detection_boxes)
                    detections_ctypes = DetectionArray(*detection_boxes)
                    try:
                        overlay_dll.UpdateDetections(detections_ctypes, len(detection_boxes))
                        logger.info(f"Sent {len(detection_boxes)} detections to overlay.")
                    except Exception as e:
                        logger.error(f"Overlay update error: {e}")

                    detection_queue.put(detection_boxes)
                else:

                    try:
                        overlay_dll.UpdateDetections(None, 0)
                    except:
                        pass
                    logger.debug("No detections to send.")

            except Exception as e:
                logger.error(f"Detection thread error: {e}")

            desired_fps = fps_var.get() if fps_var else 5
            if desired_fps <= 0:
                desired_fps = 1
            elapsed_time = time.time() - start_time
            frame_time = 1.0 / desired_fps
            sleep_time = max(0, frame_time - elapsed_time)
            time.sleep(sleep_time)

def get_tick_count():
    GetTickCount = ctypes.windll.kernel32.GetTickCount
    GetTickCount.restype = ctypes.c_uint32
    return GetTickCount()

def sanitize_bounding_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return x1, y1, x2 - x1, y2 - y1

def encode_label(label_text):
    encoded = label_text.encode('utf-8')[:49]
    return encoded + b'\x00' + b'\x00' * (49 - len(encoded))

def rgb_to_colorref(r, g, b):
    return (b << 16) | (g << 8) | r

def main():
    root = tk.Tk()
    app = AimbotApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
