import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import multiprocessing
import time
import os
import sys
import tempfile
from pathlib import Path
import pandas as pd

# Try to import PIL for background image support
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL (Pillow) not available. Background image features will be limited.")

# PSSE imports for PSSE 35.6 with Python 39
sys_paths = [r'C:\Program Files\PTI\PSSE35\35.6\PSSPY39']
env_paths = [r'C:\Program Files\PTI\PSSE35\35.6\PSSBIN', 
             r'C:\Program Files\PTI\PSSE35\35.6\PSSLIB']
for path in sys_paths:
    sys.path.append(path)
for path in env_paths:
    os.environ['PATH'] = os.environ['PATH'] + ';' + path

try:
    import psse35
    import psspy
    import redirect
    import arrbox
    import excelpy
    import pssplot
    import dyntools
    PSSE_AVAILABLE = True
except ImportError as e:
    PSSE_AVAILABLE = False
    print(f"PSSE modules not available: {e}")

from time import perf_counter as tic

class MyLabGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MyLab - Power System Analysis Suite")
        self.root.geometry("1400x900")
        
        # Initialize variables
        self.setup_variables()
        
        # Setup background image
        self.setup_background()
        
        # Create menu bar
        self.create_menu()
        
        # Create GUI components
        self.create_widgets()
        
        # Results storage
        self.accc_thermal_violations = []
        self.accc_voltage_violations = []
        self.dynamic_results = {}
        
    def setup_variables(self):
        """Initialize tkinter variables"""
        # Common file paths
        self.sav_file = tk.StringVar()
        
        # ACCC specific files
        self.sub_file = tk.StringVar()
        self.mon_file = tk.StringVar()
        self.con_file = tk.StringVar()
        self.accc_output_file = tk.StringVar(value="contingency_results.xlsx")
        
        # Dynamic specific files
        self.dyr_file = tk.StringVar()
        self.cnv_file = tk.StringVar()
        self.snp_file = tk.StringVar()
        # Set default output file to current directory with absolute path
        default_output = os.path.join(os.getcwd(), "dynamic_results.outx")
        self.dynamic_output_file = tk.StringVar(value=default_output)
        
        # ACCC Parameters
        self.thermal_threshold = tk.DoubleVar(value=90.0)  # percentage
        self.voltage_high = tk.DoubleVar(value=1.05)
        self.voltage_low = tk.DoubleVar(value=0.92)
        self.tolerance = tk.DoubleVar(value=0.5)
        self.num_threads = tk.IntVar(value=8)
        self.accc_method = tk.StringVar(value="DSP")  # DSP, Parallel, or Multiprocessing
        self.use_multiprocessing = tk.BooleanVar(value=False)
        self.num_processes = tk.IntVar(value=multiprocessing.cpu_count())
        
        # Dynamic Parameters
        self.fault_bus = tk.IntVar(value=101)
        self.fault_start_time = tk.DoubleVar(value=1.0)
        self.fault_duration = tk.DoubleVar(value=0.1)  # seconds
        self.simulation_time = tk.DoubleVar(value=20.0)
        self.fault_impedance = tk.DoubleVar(value=0.0)
        self.enable_plotting = tk.BooleanVar(value=False)  # Enable/disable automatic plotting
        
        # P3_P6 Contingency Parameters
        self.p3p6_input_file = tk.StringVar()  # Input .con file with line/bus information
        self.p3p6_output_file = tk.StringVar(value="output.con")  # Output contingency file
        
        # IDV Generator Parameters
        self.idv_impedance_file = tk.StringVar()
        self.idv_ratings_file = tk.StringVar()
        self.idv_from_bus = tk.StringVar()
        self.idv_to_bus = tk.StringVar()
        self.idv_amps = tk.StringVar()
        self.idv_miles = tk.StringVar()
        self.idv_kv = tk.StringVar()
        self.idv_conductors_per_phase = tk.StringVar(value="1")
        self.idv_geometry = tk.StringVar()
        self.idv_conductor = tk.StringVar()
        
        # IDV data storage
        self.impedance_df = None
        self.ratings_df = None
        
        # Status
        self.is_running = tk.BooleanVar(value=False)
        
    def setup_background(self):
        """Setup background image for the main window"""
        if not PIL_AVAILABLE:
            # If PIL is not available, just set a nice background color
            self.root.configure(bg='#e6f3ff')  # Light blue background
            return
            
        try:
            # Try to load a background image from the same directory
            # You can change this path to point to your desired background image
            background_path = os.path.join(os.path.dirname(__file__), "background.jpg")
            
            # If the default image doesn't exist, create a simple gradient background
            if not os.path.exists(background_path):
                # Create a simple gradient background
                self.create_gradient_background()
            else:
                # Load and set the background image
                self.load_background_image(background_path)
                
        except Exception as e:
            print(f"Could not set background image: {e}")
            # Fallback to a simple color background
            self.root.configure(bg='#e6f3ff')
    
    def create_gradient_background(self):
        """Create a simple gradient background"""
        if not PIL_AVAILABLE:
            self.root.configure(bg='#e6f3ff')  # Light blue fallback
            return
            
        try:
            # Create a gradient image
            width, height = 1400, 900
            image = Image.new('RGB', (width, height), '#f0f0f0')
            
            # Create a simple vertical gradient from light blue to white
            for y in range(height):
                # Calculate color based on position
                ratio = y / height
                r = int(230 + (255 - 230) * ratio)  # 230 to 255 (lighter gradient)
                g = int(245 + (255 - 245) * ratio)  # 245 to 255  
                b = int(255)  # Keep blue at 255
                
                # Draw horizontal line with this color
                for x in range(width):
                    image.putpixel((x, y), (r, g, b))
            
            # Convert to PhotoImage and set as background
            self.bg_image = ImageTk.PhotoImage(image)
            
            # Create a label to hold the background image
            self.bg_label = tk.Label(self.root, image=self.bg_image)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            
        except Exception as e:
            print(f"Could not create gradient background: {e}")
            self.root.configure(bg='#e6f3ff')  # Light blue fallback
    
    def load_background_image(self, image_path):
        """Load and set a background image from file"""
        if not PIL_AVAILABLE:
            print("PIL not available - cannot load background images")
            self.root.configure(bg='#e6f3ff')
            return
            
        try:
            # Open and resize image to fit window
            image = Image.open(image_path)
            image = image.resize((1400, 900), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.bg_image = ImageTk.PhotoImage(image)
            
            # Create a label to hold the background image
            self.bg_label = tk.Label(self.root, image=self.bg_image)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            
        except Exception as e:
            print(f"Could not load background image {image_path}: {e}")
            self.create_gradient_background()
    
    def create_menu(self):
        """Create menu bar with background options"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # View menu for background options
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Change Background Image...", command=self.change_background_image)
        view_menu.add_command(label="Reset to Default Background", command=self.reset_background)
        view_menu.add_separator()
        view_menu.add_command(label="Remove Background", command=self.remove_background)
    
    def change_background_image(self):
        """Allow user to select a new background image"""
        if not PIL_AVAILABLE:
            messagebox.showwarning("Feature Not Available", 
                                 "Background image feature requires PIL (Pillow) library.\n"
                                 "Please install it using: pip install Pillow")
            return
            
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Background Image",
            filetypes=filetypes
        )
        
        if filename:
            self.load_background_image(filename)
    
    def reset_background(self):
        """Reset to default gradient background"""
        try:
            if hasattr(self, 'bg_label'):
                self.bg_label.destroy()
            self.create_gradient_background()
        except Exception as e:
            print(f"Error resetting background: {e}")
    
    def remove_background(self):
        """Remove background image/gradient"""
        try:
            if hasattr(self, 'bg_label'):
                self.bg_label.destroy()
            self.root.configure(bg='#f0f0f0')  # Set to default gray
        except Exception as e:
            print(f"Error removing background: {e}")
        
    def create_widgets(self):
        """Create and layout GUI widgets"""
        # Create main notebook for different analysis types
        self.main_notebook = ttk.Notebook(self.root)
        self.main_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # ACCC Analysis tab
        self.accc_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.accc_frame, text="ACCC Analysis")
        
        # Dynamic Analysis tab
        self.dynamic_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.dynamic_frame, text="Dynamic Analysis")
        
        # P3_P6 Contingency tab
        self.p3p6_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.p3p6_frame, text="P3_P6 Contingency")
        
        # IDV Generator tab
        self.idv_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.idv_frame, text="IDV Generator")
        
        # Results tab
        self.results_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.results_frame, text="Results")
        
        self.create_accc_tab()
        self.create_dynamic_tab()
        self.create_p3p6_tab()
        self.create_idv_tab()
        self.create_results_tab()
        
    def create_accc_tab(self):
        """Create the ACCC analysis tab"""
        # File selection frame
        accc_file_frame = ttk.LabelFrame(self.accc_frame, text="ACCC Input Files", padding=10)
        accc_file_frame.pack(fill="x", padx=10, pady=5)
        
        # SAV file
        ttk.Label(accc_file_frame, text="Case File (.sav):").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(accc_file_frame, textvariable=self.sav_file, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(accc_file_frame, text="Browse", command=lambda: self.browse_file(self.sav_file, "PSSE Case Files", "*.sav")).grid(row=0, column=2, pady=2)
        
        # SUB file
        ttk.Label(accc_file_frame, text="Subsystem File (.sub):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(accc_file_frame, textvariable=self.sub_file, width=60).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(accc_file_frame, text="Browse", command=lambda: self.browse_file(self.sub_file, "Subsystem Files", "*.sub")).grid(row=1, column=2, pady=2)
        
        # MON file
        ttk.Label(accc_file_frame, text="Monitor File (.mon):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(accc_file_frame, textvariable=self.mon_file, width=60).grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(accc_file_frame, text="Browse", command=lambda: self.browse_file(self.mon_file, "Monitor Files", "*.mon")).grid(row=2, column=2, pady=2)
        
        # CON file
        ttk.Label(accc_file_frame, text="Contingency File (.con):").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(accc_file_frame, textvariable=self.con_file, width=60).grid(row=3, column=1, padx=5, pady=2)
        ttk.Button(accc_file_frame, text="Browse", command=lambda: self.browse_file(self.con_file, "Contingency Files", "*.con")).grid(row=3, column=2, pady=2)
        
        # ACCC Parameters frame
        accc_param_frame = ttk.LabelFrame(self.accc_frame, text="ACCC Parameters", padding=10)
        accc_param_frame.pack(fill="x", padx=10, pady=5)
        
        # Left column parameters
        left_accc_param = ttk.Frame(accc_param_frame)
        left_accc_param.pack(side="left", fill="x", expand=True)
        
        ttk.Label(left_accc_param, text="Thermal Loading Threshold (%):").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(left_accc_param, textvariable=self.thermal_threshold, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(left_accc_param, text="Voltage High Limit (p.u.):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(left_accc_param, textvariable=self.voltage_high, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(left_accc_param, text="Voltage Low Limit (p.u.):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(left_accc_param, textvariable=self.voltage_low, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        # Right column parameters
        right_accc_param = ttk.Frame(accc_param_frame)
        right_accc_param.pack(side="right", fill="x", expand=True)
        
        ttk.Label(right_accc_param, text="Tolerance:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(right_accc_param, textvariable=self.tolerance, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(right_accc_param, text="Number of Threads:").grid(row=1, column=0, sticky="w", pady=2)
        self.threads_entry = ttk.Entry(right_accc_param, textvariable=self.num_threads, width=10)
        self.threads_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(right_accc_param, text="ACCC Method:").grid(row=2, column=0, sticky="w", pady=2)
        accc_method_combo = ttk.Combobox(right_accc_param, textvariable=self.accc_method, width=12, state="readonly")
        accc_method_combo['values'] = ('DSP', 'Parallel', 'Multiprocessing')
        accc_method_combo.grid(row=2, column=1, padx=5, pady=2)
        accc_method_combo.bind('<<ComboboxSelected>>', self.on_accc_method_change)
        
        # Add field for number of processes (for multiprocessing)
        ttk.Label(right_accc_param, text="Number of Processes:").grid(row=3, column=0, sticky="w", pady=2)
        self.processes_entry = ttk.Entry(right_accc_param, textvariable=self.num_processes, width=10)
        self.processes_entry.grid(row=3, column=1, padx=5, pady=2)
        
        # Add help text for ACCC methods
        method_help = ttk.Label(right_accc_param, text="DSP: Stable (Recommended) | Parallel: Fast but unstable | Multiprocessing: Best for multiple cases", 
                               font=('TkDefaultFont', 8), foreground='gray')
        method_help.grid(row=2, column=2, columnspan=2, sticky="w", padx=5, pady=2)
        
        ttk.Label(right_accc_param, text="Output Excel File:").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Entry(right_accc_param, textvariable=self.accc_output_file, width=20).grid(row=4, column=1, padx=5, pady=2)
        
        # ACCC Control frame
        accc_control_frame = ttk.LabelFrame(self.accc_frame, text="ACCC Control", padding=10)
        accc_control_frame.pack(fill="x", padx=10, pady=5)
        
        self.accc_run_button = ttk.Button(accc_control_frame, text="Run ACCC Analysis", command=self.run_accc_analysis)
        self.accc_run_button.pack(side="left", padx=5)
        
        ttk.Button(accc_control_frame, text="Export ACCC Results", command=self.export_accc_results).pack(side="left", padx=5)
        ttk.Button(accc_control_frame, text="PSSE Violations Report", command=self.export_psse_violations_report).pack(side="left", padx=5)
        
        # ACCC Progress frame
        accc_progress_frame = ttk.LabelFrame(self.accc_frame, text="ACCC Progress", padding=10)
        accc_progress_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.accc_progress_bar = ttk.Progressbar(accc_progress_frame, mode='indeterminate')
        self.accc_progress_bar.pack(fill="x", pady=5)
        
        self.accc_log_text = scrolledtext.ScrolledText(accc_progress_frame, height=8, wrap=tk.WORD)
        self.accc_log_text.pack(fill="both", expand=True)
        
        # Initialize threads entry state based on default method
        self.on_accc_method_change(None)
        
    def on_accc_method_change(self, event):
        """Handle ACCC method selection change"""
        method = self.accc_method.get()
        
        if method == "DSP":
            # DSP method doesn't use threads or processes - disable both entries
            self.threads_entry.config(state="disabled")
            self.processes_entry.config(state="disabled")
        elif method == "Parallel":
            # Parallel method uses threads - enable threads, disable processes
            self.threads_entry.config(state="normal")
            self.processes_entry.config(state="disabled")
        elif method == "Multiprocessing":
            # Multiprocessing uses processes - disable threads, enable processes
            self.threads_entry.config(state="disabled")
            self.processes_entry.config(state="normal")
    
    def create_dynamic_tab(self):
        """Create the Dynamic analysis tab"""
        # File selection frame
        dyn_file_frame = ttk.LabelFrame(self.dynamic_frame, text="Dynamic Input Files", padding=10)
        dyn_file_frame.pack(fill="x", padx=10, pady=5)
        
        # SAV file (shared)
        ttk.Label(dyn_file_frame, text="Case File (.sav):").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(dyn_file_frame, textvariable=self.sav_file, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(dyn_file_frame, text="Browse", command=lambda: self.browse_file(self.sav_file, "PSSE Case Files", "*.sav")).grid(row=0, column=2, pady=2)
        
        # DYR file
        ttk.Label(dyn_file_frame, text="Dynamics File (.dyr):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(dyn_file_frame, textvariable=self.dyr_file, width=60).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(dyn_file_frame, text="Browse", command=lambda: self.browse_file(self.dyr_file, "Dynamics Files", "*.dyr")).grid(row=1, column=2, pady=2)
        
        # CNV file (Converted case file)
        ttk.Label(dyn_file_frame, text="Converted Case File (.sav):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(dyn_file_frame, textvariable=self.cnv_file, width=60).grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(dyn_file_frame, text="Browse", command=lambda: self.browse_file(self.cnv_file, "Converted Case Files", "*.sav")).grid(row=2, column=2, pady=2)
        
        # SNP file (Snapshot file)
        ttk.Label(dyn_file_frame, text="Snapshot File (.snp):").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(dyn_file_frame, textvariable=self.snp_file, width=60).grid(row=3, column=1, padx=5, pady=2)
        ttk.Button(dyn_file_frame, text="Browse", command=lambda: self.browse_file(self.snp_file, "Snapshot Files", "*.snp")).grid(row=3, column=2, pady=2)
        
        # Dynamic Parameters frame
        dyn_param_frame = ttk.LabelFrame(self.dynamic_frame, text="Dynamic Simulation Parameters", padding=10)
        dyn_param_frame.pack(fill="x", padx=10, pady=5)
        
        # Left column parameters
        left_dyn_param = ttk.Frame(dyn_param_frame)
        left_dyn_param.pack(side="left", fill="x", expand=True)
        
        ttk.Label(left_dyn_param, text="Fault Bus Number:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(left_dyn_param, textvariable=self.fault_bus, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(left_dyn_param, text="Fault Start Time (s):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(left_dyn_param, textvariable=self.fault_start_time, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(left_dyn_param, text="Fault Duration (s):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(left_dyn_param, textvariable=self.fault_duration, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        # Right column parameters
        right_dyn_param = ttk.Frame(dyn_param_frame)
        right_dyn_param.pack(side="right", fill="x", expand=True)
        
        ttk.Label(right_dyn_param, text="Total Simulation Time (s):").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(right_dyn_param, textvariable=self.simulation_time, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(right_dyn_param, text="Fault Impedance (p.u.):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(right_dyn_param, textvariable=self.fault_impedance, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(right_dyn_param, text="Output File (.outx):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(right_dyn_param, textvariable=self.dynamic_output_file, width=15).grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(right_dyn_param, text="Browse", command=self.browse_output_file).grid(row=2, column=2, pady=2)
        
        # Plotting option
        ttk.Checkbutton(right_dyn_param, text="Enable Auto-Plotting", variable=self.enable_plotting).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)
        
        # Dynamic Control frame
        dyn_control_frame = ttk.LabelFrame(self.dynamic_frame, text="Dynamic Control", padding=10)
        dyn_control_frame.pack(fill="x", padx=10, pady=5)
        
        self.dynamic_run_button = ttk.Button(dyn_control_frame, text="Run Dynamic Analysis", command=self.run_dynamic_analysis)
        self.dynamic_run_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(dyn_control_frame, text="Stop Analysis", command=self.stop_analysis, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        
        ttk.Button(dyn_control_frame, text="View Dynamic Results", command=self.view_dynamic_results).pack(side="left", padx=5)
        
        # Dynamic Progress frame
        dyn_progress_frame = ttk.LabelFrame(self.dynamic_frame, text="Dynamic Progress", padding=10)
        dyn_progress_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.progress_bar = ttk.Progressbar(dyn_progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill="x", pady=5)
        
        self.dynamic_log_text = scrolledtext.ScrolledText(dyn_progress_frame, height=8, wrap=tk.WORD)
        self.dynamic_log_text.pack(fill="both", expand=True)
        
    def create_p3p6_tab(self):
        """Create the P3_P6 contingency generation tab"""
        # File selection frame
        p3p6_file_frame = ttk.LabelFrame(self.p3p6_frame, text="P3_P6 Contingency Files", padding=10)
        p3p6_file_frame.pack(fill="x", padx=10, pady=5)
        
        # Input CON file
        ttk.Label(p3p6_file_frame, text="Input File (.con):").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(p3p6_file_frame, textvariable=self.p3p6_input_file, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(p3p6_file_frame, text="Browse", command=lambda: self.browse_file(self.p3p6_input_file, "Contingency Files", "*.con")).grid(row=0, column=2, pady=2)
        
        # Output contingency file
        ttk.Label(p3p6_file_frame, text="Output File (.con):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(p3p6_file_frame, textvariable=self.p3p6_output_file, width=60).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(p3p6_file_frame, text="Browse", command=lambda: self.browse_save_file(self.p3p6_output_file, "Contingency Files", "*.con")).grid(row=1, column=2, pady=2)
        
        # Information frame
        info_frame = ttk.LabelFrame(self.p3p6_frame, text="Information", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        info_text = """P3_P6 Contingency Generator
        
This tool creates single and double contingencies from a .con file containing line/bus information.

Input File Format:
- Each line should contain: FROM_BUS, TO_BUS [, CKT] [, FROM_NAME] [, TO_NAME]
- Supports various delimiters: comma, semicolon, pipe, tab, or whitespace
- Circuit ID defaults to '1' if not specified
- Bus names default to 'BUS <number>' if not specified
- Comments and blank lines are ignored

Example formats:
101, 102, 1, NUC-A, NUC-B
201 202 2 /* STATION-A - STATION-B */
301|302|1|PLANT-X|PLANT-Y

Output:
- Creates single contingencies (P3) for each line
- Creates double contingencies (P6) for all line combinations
- Output file is in PSSE contingency format (.con)"""
        
        info_label = tk.Label(info_frame, text=info_text, justify="left", wraplength=800)
        info_label.pack(anchor="w")
        
        # Control buttons frame
        control_frame = ttk.Frame(self.p3p6_frame)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(control_frame, text="Generate Contingencies", 
                  command=self.generate_p3p6_contingencies,
                  style="Accent.TButton").pack(side="left", padx=5)
        
        # Progress and log frame
        progress_frame = ttk.LabelFrame(self.p3p6_frame, text="Progress & Log", padding=10)
        progress_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.p3p6_progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.p3p6_progress_bar.pack(fill="x", pady=5)
        
        self.p3p6_log_text = scrolledtext.ScrolledText(progress_frame, height=10, wrap=tk.WORD)
        self.p3p6_log_text.pack(fill="both", expand=True)
        
    def create_idv_tab(self):
        """Create the IDV Generator tab"""
        # File selection frame
        idv_file_frame = ttk.LabelFrame(self.idv_frame, text="CSV Files", padding=10)
        idv_file_frame.pack(fill="x", padx=10, pady=5)
        
        # Load impedance file button
        self.idv_impedance_button = ttk.Button(idv_file_frame, text="Browse - Line Impedance Estimator CSV", 
                                             command=self.load_impedance_file, 
                                             style="Accent.TButton")
        self.idv_impedance_button.pack(fill="x", pady=5)
        
        # Load ratings file button (optional)
        self.idv_ratings_button = ttk.Button(idv_file_frame, text="Browse - Ratings CSV (Optional)", 
                                           command=self.load_ratings_file)
        self.idv_ratings_button.pack(fill="x", pady=5)
        
        # Input data frame
        idv_input_frame = ttk.LabelFrame(self.idv_frame, text="Enter Data", padding=10)
        idv_input_frame.pack(fill="x", padx=10, pady=5)
        
        # Left column
        left_input = ttk.Frame(idv_input_frame)
        left_input.pack(side="left", fill="x", expand=True, padx=5)
        
        ttk.Label(left_input, text="From Bus:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(left_input, textvariable=self.idv_from_bus, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(left_input, text="To Bus:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(left_input, textvariable=self.idv_to_bus, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(left_input, text="Min Rating (Amps):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(left_input, textvariable=self.idv_amps, width=15).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(left_input, text="Length (Miles):").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(left_input, textvariable=self.idv_miles, width=15).grid(row=3, column=1, padx=5, pady=2)
        
        # Right column
        right_input = ttk.Frame(idv_input_frame)
        right_input.pack(side="right", fill="x", expand=True, padx=5)
        
        ttk.Label(right_input, text="Voltage Level (kV):").grid(row=0, column=0, sticky="w", pady=2)
        self.idv_kv_combo = ttk.Combobox(right_input, textvariable=self.idv_kv, width=12, state="readonly")
        self.idv_kv_combo.grid(row=0, column=1, padx=5, pady=2)
        self.idv_kv.trace('w', self.update_idv_conductors)
        
        ttk.Label(right_input, text="Conductors Per Phase:").grid(row=1, column=0, sticky="w", pady=2)
        self.idv_cpp_combo = ttk.Combobox(right_input, textvariable=self.idv_conductors_per_phase, 
                                        values=['1', '2'], width=12, state="readonly")
        self.idv_cpp_combo.grid(row=1, column=1, padx=5, pady=2)
        self.idv_conductors_per_phase.trace('w', self.update_idv_conductors)
        
        ttk.Label(right_input, text="Geometry:").grid(row=2, column=0, sticky="w", pady=2)
        self.idv_geometry_combo = ttk.Combobox(right_input, textvariable=self.idv_geometry, width=12, state="readonly")
        self.idv_geometry_combo.grid(row=2, column=1, padx=5, pady=2)
        self.idv_geometry.trace('w', self.update_idv_conductors)
        
        ttk.Label(right_input, text="Conductor Type:").grid(row=3, column=0, sticky="w", pady=2)
        self.idv_conductor_combo = ttk.Combobox(right_input, textvariable=self.idv_conductor, width=12, state="readonly")
        self.idv_conductor_combo.grid(row=3, column=1, padx=5, pady=2)
        
        # Generate button
        generate_frame = ttk.Frame(self.idv_frame)
        generate_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(generate_frame, text="Generate IDV", 
                  command=self.generate_idv,
                  style="Accent.TButton").pack(side="left", padx=5)
        
        # Output frame
        output_frame = ttk.LabelFrame(self.idv_frame, text="Generated IDV", padding=10)
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.idv_output_text = scrolledtext.ScrolledText(output_frame, height=8, wrap=tk.WORD, font=('Courier', 9))
        self.idv_output_text.pack(fill="both", expand=True)
        
    def create_results_tab(self):
        """Create the results display tab"""
        # Create notebook for different result types
        results_notebook = ttk.Notebook(self.results_frame)
        results_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # ACCC Results
        accc_results_frame = ttk.Frame(results_notebook)
        results_notebook.add(accc_results_frame, text="ACCC Results")
        
        # ACCC sub-notebook for thermal and voltage
        accc_sub_notebook = ttk.Notebook(accc_results_frame)
        accc_sub_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Thermal violations tab
        thermal_frame = ttk.Frame(accc_sub_notebook)
        accc_sub_notebook.add(thermal_frame, text="Thermal Violations")
        
        thermal_columns = ("Contingency", "Converged", "Branch", "Rating (MVA)", "Flow (MVA)", "Loading (%)")
        self.thermal_tree = ttk.Treeview(thermal_frame, columns=thermal_columns, show="headings", height=15)
        
        for col in thermal_columns:
            self.thermal_tree.heading(col, text=col)
            self.thermal_tree.column(col, width=120)
        
        thermal_scrollbar = ttk.Scrollbar(thermal_frame, orient="vertical", command=self.thermal_tree.yview)
        self.thermal_tree.configure(yscrollcommand=thermal_scrollbar.set)
        
        self.thermal_tree.pack(side="left", fill="both", expand=True)
        thermal_scrollbar.pack(side="right", fill="y")
        
        # Voltage violations tab
        voltage_frame = ttk.Frame(accc_sub_notebook)
        accc_sub_notebook.add(voltage_frame, text="Voltage Violations")
        
        voltage_columns = ("Contingency", "Converged", "Bus", "Voltage (p.u.)")
        self.voltage_tree = ttk.Treeview(voltage_frame, columns=voltage_columns, show="headings", height=15)
        
        for col in voltage_columns:
            self.voltage_tree.heading(col, text=col)
            self.voltage_tree.column(col, width=150)
        
        voltage_scrollbar = ttk.Scrollbar(voltage_frame, orient="vertical", command=self.voltage_tree.yview)
        self.voltage_tree.configure(yscrollcommand=voltage_scrollbar.set)
        
        self.voltage_tree.pack(side="left", fill="both", expand=True)
        voltage_scrollbar.pack(side="right", fill="y")
        
        # Dynamic Results
        dynamic_results_frame = ttk.Frame(results_notebook)
        results_notebook.add(dynamic_results_frame, text="Dynamic Results")
        
        # Dynamic results display
        self.dynamic_results_text = scrolledtext.ScrolledText(dynamic_results_frame, height=20, wrap=tk.WORD)
        self.dynamic_results_text.pack(fill="both", expand=True, padx=5, pady=5)
        
    def browse_file(self, var, file_types, extension):
        """Open file dialog and set the selected file path"""
        filename = filedialog.askopenfilename(
            title=f"Select {file_types}",
            filetypes=[(file_types, extension), ("All files", "*.*")]
        )
        if filename:
            var.set(filename)
    
    def browse_output_file(self):
        """Browse for output file location"""
        filename = filedialog.asksaveasfilename(
            title="Save Dynamic Results As",
            defaultextension=".outx",
            filetypes=[("PSSE Output Files", "*.outx"), ("All files", "*.*")]
        )
        if filename:
            self.dynamic_output_file.set(filename)
    
    def browse_save_file(self, var, file_types, extension):
        """Open save file dialog and set the selected file path"""
        filename = filedialog.asksaveasfilename(
            title=f"Save {file_types} As",
            defaultextension=extension.replace("*", ""),
            filetypes=[(file_types, extension), ("All files", "*.*")]
        )
        if filename:
            var.set(filename)
    
    def log_message(self, message, log_widget=None):
        """Add a message to the specified log display"""
        if log_widget is None:
            log_widget = self.dynamic_log_text
        
        log_widget.insert(tk.END, f"{message}\n")
        log_widget.see(tk.END)
        self.root.update_idletasks()
    
    def validate_accc_inputs(self):
        """Validate ACCC input files and parameters"""
        if not self.sav_file.get():
            messagebox.showerror("Error", "Please select a case file (.sav)")
            return False
        if not self.sub_file.get():
            messagebox.showerror("Error", "Please select a subsystem file (.sub)")
            return False
        if not self.mon_file.get():
            messagebox.showerror("Error", "Please select a monitor file (.mon)")
            return False
        if not self.con_file.get():
            messagebox.showerror("Error", "Please select a contingency file (.con)")
            return False
        
        # Check if files exist
        for file_path in [self.sav_file.get(), self.sub_file.get(), self.mon_file.get(), self.con_file.get()]:
            if not os.path.exists(file_path):
                messagebox.showerror("Error", f"File not found: {file_path}")
                return False
        
        return True
    
    def validate_dynamic_inputs(self):
        """Validate Dynamic input files and parameters"""
        if not self.sav_file.get():
            messagebox.showerror("Error", "Please select a case file (.sav)")
            return False
        if not self.dyr_file.get():
            messagebox.showerror("Error", "Please select a dynamics file (.dyr)")
            return False
        if not self.cnv_file.get():
            messagebox.showerror("Error", "Please select a converted case file (.sav)")
            return False
        if not self.snp_file.get():
            messagebox.showerror("Error", "Please select a snapshot file (.snp)")
            return False
        
        # Check if files exist
        for file_path in [self.sav_file.get(), self.dyr_file.get(), self.cnv_file.get(), self.snp_file.get()]:
            if not os.path.exists(file_path):
                messagebox.showerror("Error", f"File not found: {file_path}")
                return False
        
        return True
    
    def run_accc_analysis(self):
        """Start the ACCC analysis in a separate thread"""
        if not self.validate_accc_inputs():
            return
        
        if not PSSE_AVAILABLE:
            messagebox.showerror("Error", "PSSE modules are not available. Please check your PSSE installation.")
            return
        
        # Update UI state
        self.is_running.set(True)
        self.accc_run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.accc_progress_bar.start()
        
        # Clear previous results
        self.clear_accc_results()
        
        # Start analysis in separate thread
        self.analysis_thread = threading.Thread(target=self.perform_accc_analysis)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def run_dynamic_analysis(self):
        """Start the Dynamic analysis in a separate thread"""
        if not self.validate_dynamic_inputs():
            return
        
        if not PSSE_AVAILABLE:
            messagebox.showerror("Error", "PSSE modules are not available. Please check your PSSE installation.")
            return
        
        # Update UI state
        self.is_running.set(True)
        self.dynamic_run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.progress_bar.start()
        
        # Clear previous results
        self.clear_dynamic_results()
        
        # Start analysis in separate thread
        self.analysis_thread = threading.Thread(target=self.perform_dynamic_analysis)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def stop_analysis(self):
        """Stop the analysis"""
        self.is_running.set(False)
        self.log_message("Analysis stopped by user")
        self.analysis_complete()
    
    def analysis_complete(self):
        """Called when analysis is finished"""
        self.is_running.set(False)
        self.accc_run_button.config(state="normal")
        self.dynamic_run_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.progress_bar.stop()
        self.accc_progress_bar.stop()
        
    def clear_accc_results(self):
        """Clear ACCC results from the display"""
        # Clear trees
        for item in self.thermal_tree.get_children():
            self.thermal_tree.delete(item)
        for item in self.voltage_tree.get_children():
            self.voltage_tree.delete(item)
        
        # Clear stored results
        self.accc_thermal_violations = []
        self.accc_voltage_violations = []
        
        # Clear log
        self.accc_log_text.delete(1.0, tk.END)
    
    def clear_dynamic_results(self):
        """Clear Dynamic results from the display"""
        self.dynamic_results = {}
        self.dynamic_results_text.delete(1.0, tk.END)
        self.dynamic_log_text.delete(1.0, tk.END)
    
    @staticmethod
    def accc_worker(case_data):
        """Worker function for multiprocessing ACCC analysis (static method)"""
        sav_file, sub_file, mon_file, con_file, tolerance, output_dir = case_data
        
        try:
            # Import PSSE modules in worker process
            import sys
            import os
            sys_paths = [r'C:\Program Files\PTI\PSSE35\35.6\PSSPY39']
            env_paths = [r'C:\Program Files\PTI\PSSE35\35.6\PSSBIN', 
                         r'C:\Program Files\PTI\PSSE35\35.6\PSSLIB']
            for path in sys_paths:
                if path not in sys.path:
                    sys.path.append(path)
            for path in env_paths:
                if path not in os.environ['PATH']:
                    os.environ['PATH'] = os.environ['PATH'] + ';' + path
            
            import psse35
            import psspy
            import redirect
            
            # Initialize PSSE in worker process
            redirect.psse2py()
            ierr = psspy.psseinit()
            
            if ierr != 0:
                return {"error": f"PSSE initialization failed with error: {ierr}"}
            
            # Load case
            psspy.case(sav_file)
            
            # Set solution parameters like working GINR script
            psspy.solution_parameters_4(intgar2=40)
            
            # Create file paths
            dfx_file = str(Path(sav_file).with_suffix('.dfx'))
            acc_file = str(Path(sav_file).with_suffix('.acc'))
            
            # Create DFAX file
            optdfax = [1, 0, 0]  # Same as working GINR script
            ierr_dfax = psspy.dfax_2(optdfax, sub_file, mon_file, con_file, dfx_file)
            
            if ierr_dfax != 0:
                return {"error": f"DFAX creation failed with error: {ierr_dfax}"}
            
            # Run ACCC using DSP method (most reliable for multiprocessing)
            optacc = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  # Same as working GINR script
            empty = ""
            ierr_accc = psspy.accc_with_dsp_3(tolerance, optacc, empty, dfx_file, acc_file, empty, empty, empty)
            
            # Check if files were created
            dfx_exists = os.path.exists(dfx_file)
            acc_exists = os.path.exists(acc_file)
            
            return {
                "case": os.path.basename(sav_file),
                "dfax_error": ierr_dfax,
                "accc_error": ierr_accc,
                "dfx_file": dfx_file,
                "acc_file": acc_file,
                "dfx_exists": dfx_exists,
                "acc_exists": acc_exists,
                "success": ierr_dfax == 0 and ierr_accc == 0 and dfx_exists and acc_exists
            }
            
        except Exception as e:
            return {"error": str(e), "case": os.path.basename(sav_file)}
    
    def perform_accc_analysis(self):
        """Perform the ACCC analysis (runs in separate thread)"""
        try:
            self.log_message("Starting ACCC Analysis...", self.accc_log_text)
            
            # Kill any hanging PSSE processes before starting
            try:
                import subprocess
                result = subprocess.run(['taskkill', '/f', '/im', 'psse35.exe'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    self.log_message("Cleaned up hanging PSSE processes", self.accc_log_text)
            except:
                pass  # Ignore if no processes to kill
            
            self.log_message("Initializing PSSE...", self.accc_log_text)
            
            # Initialize PSSE exactly like working GINR script
            redirect.psse2py()
            ierr = psspy.psseinit()
            
            if ierr != 0:
                self.log_message(f"PSSE initialization failed with error: {ierr}", self.accc_log_text)
                return
            
            # Load case - using exact same approach as working GINR script
            self.log_message(f"Loading case: {self.sav_file.get()}", self.accc_log_text)
            psspy.case(self.sav_file.get())
            
            # Set solution parameters exactly like working GINR script
            # Change max iterations - Has to be called after a case is loaded
            self.log_message("Setting solution parameters...", self.accc_log_text)
            psspy.solution_parameters_4(intgar2=40)
            self.log_message("Solution parameters set successfully", self.accc_log_text)
            
            # Create DFAX file - using same approach as working accc.py
            self.log_message("Creating DFAX file...", self.accc_log_text)
            dfx_file = str(Path(self.sav_file.get()).with_suffix('.dfx'))
            self.log_message(f"DFAX file will be saved as: {dfx_file}", self.accc_log_text)
            
            # Log and verify the input files being used
            self.log_message(f"SUB file: {self.sub_file.get()}", self.accc_log_text)
            self.log_message(f"MON file: {self.mon_file.get()}", self.accc_log_text)
            self.log_message(f"CON file: {self.con_file.get()}", self.accc_log_text)
            
            # Verify all input files exist and are accessible
            for file_type, file_path in [("SUB", self.sub_file.get()), ("MON", self.mon_file.get()), ("CON", self.con_file.get())]:
                if not os.path.exists(file_path):
                    self.log_message(f"ERROR: {file_type} file not found: {file_path}", self.accc_log_text)
                    return
                else:
                    file_size = os.path.getsize(file_path)
                    self.log_message(f"✓ {file_type} file verified ({file_size} bytes)", self.accc_log_text)
                    
                    # Special check for large contingency files
                    if file_type == "CON" and file_size > 100000:  # > 100KB
                        self.log_message(f"Note: Large contingency file detected ({file_size/1024:.1f} KB)", self.accc_log_text)
                        self.log_message("This may take longer to process...", self.accc_log_text)
            
            # Create DFAX file using exact same approach as working GINR script
            self.log_message("Creating DFAX file...", self.accc_log_text)
            
            try:
                # Use exact same DFAX call as working GINR script (simple, no threading)
                self.log_message(f"Calling psspy.dfax_2 with GINR parameters:", self.accc_log_text)
                self.log_message(f"  SUB: {self.sub_file.get()}", self.accc_log_text)
                self.log_message(f"  MON: {self.mon_file.get()}", self.accc_log_text)
                self.log_message(f"  CON: {self.con_file.get()}", self.accc_log_text)
                self.log_message(f"  DFX: {dfx_file}", self.accc_log_text)
                
                # Call DFAX exactly like working GINR script
                self.log_message("Executing DFAX creation (this may take a moment for large contingency files)...", self.accc_log_text)
                self.root.update_idletasks()  # Update GUI to show message
                
                # Ensure output is redirected to prevent terminal input mode
                redirect.psse2py()
                
                # Suppress interactive terminal mode by redirecting to a dummy file
                temp_output = os.path.join(tempfile.gettempdir(), "psse_output.tmp")
                try:
                    psspy.report_output(2, temp_output)  # Redirect output to file
                except:
                    pass
                
                start_time = time.time()
                ierr_dfax = psspy.dfax_2([1,1,0], self.sub_file.get(), self.mon_file.get(), self.con_file.get(), dfx_file)
                end_time = time.time()
                
                # Force output redirection back to Python and clean up
                try:
                    psspy.report_output(6)  # Back to default
                    redirect.psse2py()
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                except:
                    pass
                
                self.log_message(f"psspy.dfax_2() completed in {end_time - start_time:.1f} seconds", self.accc_log_text)
                self.log_message(f"psspy.dfax_2() returned with code: {ierr_dfax}", self.accc_log_text)
                
            except Exception as e:
                self.log_message(f"DFAX creation failed with exception: {str(e)}", self.accc_log_text)
                return
            
            if ierr_dfax != 0:
                self.log_message(f"Warning: DFAX creation returned error code: {ierr_dfax}", self.accc_log_text)
                # Check if file was created despite error
                if os.path.exists(dfx_file):
                    dfx_size = os.path.getsize(dfx_file) / 1024
                    self.log_message(f"DFAX file was created despite error ({dfx_size:.1f} KB)", self.accc_log_text)
                else:
                    self.log_message("DFAX file was NOT created - ACCC cannot proceed", self.accc_log_text)
                    return
            else:
                self.log_message("DFAX file created successfully", self.accc_log_text)
            
            # Verify DFAX file exists and show details
            if not os.path.exists(dfx_file):
                self.log_message(f"ERROR: DFAX file not found: {dfx_file}", self.accc_log_text)
                self.log_message("Cannot proceed with ACCC analysis without DFAX file", self.accc_log_text)
                return
            else:
                dfx_size = os.path.getsize(dfx_file) / 1024
                self.log_message(f"✓ DFAX file verified: {dfx_size:.1f} KB", self.accc_log_text)
                self.log_message(f"✓ DFAX file location: {dfx_file}", self.accc_log_text)
            
            # Set number of threads
            self.log_message(f"Setting number of threads to: {self.num_threads.get()}", self.accc_log_text)
            psspy.number_threads(self.num_threads.get())
            
            # Run contingency analysis
            self.log_message(f"Running parallel contingency analysis with {self.num_threads.get()} threads...", self.accc_log_text)
            acc_file = str(Path(self.sav_file.get()).with_suffix('.acc'))
            self.log_message(f"ACC results file will be saved as: {acc_file}", self.accc_log_text)
            
            # Ensure the output directory exists and is writable
            output_dir = os.path.dirname(acc_file)
            self.log_message(f"Output directory: {output_dir}", self.accc_log_text)
            
            try:
                os.makedirs(output_dir, exist_ok=True)
                # Test write permissions
                test_file = os.path.join(output_dir, "test_write.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                self.log_message("Output directory is writable", self.accc_log_text)
            except Exception as e:
                self.log_message(f"ERROR: Cannot write to output directory: {str(e)}", self.accc_log_text)
                return
            
            self.log_message("Starting ACCC parallel analysis...", self.accc_log_text)
            start_time = tic()
            
            try:
                # Choose ACCC method based on user selection
                if self.accc_method.get() == "DSP":
                    # DSP method - more stable, single-threaded (like working GINR script)
                    self.log_message(f"Calling psspy.accc_with_dsp_3 (DSP method - stable):", self.accc_log_text)
                    self.log_message(f"  Tolerance: {self.tolerance.get()}", self.accc_log_text)
                    self.log_message(f"  DFAX file: {dfx_file}", self.accc_log_text)
                    self.log_message(f"  ACC file: {acc_file}", self.accc_log_text)
                    
                    # Option functions for ACCC - exactly like working GINR script
                    optacc = [
                        0, #Tap adjustments 0=disable 1=enable stepping 2=enable direct
                        0, #Interchange 0=disable 1=enable tie flow only 2=enable tie and loads
                        0, #Phase shift 0=disable 1=enable
                        1, #dc tap 0=disable 1=enable
                        1, #switched shunt 0=disable 1=enable 2=enable continuous
                        1, #solution flag 0=FDNS 1=FNSL 2=optimized FDNS
                        0, #Non-divergent 0=disable 1=enable
                        0, #induction motor 0=stall 1=trip
                        0, #induction failure 0=treat contingency as non-converage 1=treak contingency as solved if it converges
                        0, #dispatch mode 0=disable
                        0, #zip archive 0=do not write ZIP 1=write a ZIP
                    ]
                    
                    empty = ""  # Empty variable for use in functions where no input is needed
                    
                    # Ensure output is redirected before ACCC call  
                    redirect.psse2py()
                    self.root.update_idletasks()
                    
                    # Redirect output to prevent terminal input mode
                    temp_output = os.path.join(tempfile.gettempdir(), "psse_accc_output.tmp")
                    try:
                        psspy.report_output(2, temp_output)  # Redirect output to file
                    except:
                        pass
                    
                    ierr_accc = psspy.accc_with_dsp_3(self.tolerance.get(), optacc, empty, dfx_file, acc_file, empty, empty, empty)
                    
                    # Force output redirection back to Python
                    try:
                        psspy.report_output(6)  # Back to default
                        redirect.psse2py()
                        if os.path.exists(temp_output):
                            os.remove(temp_output)
                    except:
                        pass
                    
                elif self.accc_method.get() == "Parallel":
                    # Parallel method - faster, multi-threaded (like your original accc.py script)
                    # Note: This method can be unstable with complex contingency files
                    self.log_message(f"Attempting Parallel method (multi-threaded):", self.accc_log_text)
                    self.log_message(f"  Tolerance: {self.tolerance.get()}", self.accc_log_text)
                    self.log_message(f"  Threads: {self.num_threads.get()}", self.accc_log_text)
                    self.log_message(f"  DFAX file: {dfx_file}", self.accc_log_text)
                    self.log_message(f"  ACC file: {acc_file}", self.accc_log_text)
                    
                    try:
                        # Set number of threads for parallel processing
                        psspy.number_threads(self.num_threads.get())
                        
                        # Ensure output is redirected before ACCC call
                        redirect.psse2py()
                        self.root.update_idletasks()
                        
                        # Redirect output to prevent terminal input mode
                        temp_output = os.path.join(tempfile.gettempdir(), "psse_accc_output.tmp")
                        try:
                            psspy.report_output(2, temp_output)  # Redirect output to file
                        except:
                            pass
                        
                        self.log_message("Calling psspy.accc_parallel_2...", self.accc_log_text)
                        # Use parallel ACCC method (like your working accc.py script)
                        ierr_accc = psspy.accc_parallel_2(self.tolerance.get(), [0,0,0,1,1,2,0,0,0,1,0], "STUDY", dfx_file, acc_file)
                        
                        # Force output redirection back to Python
                        try:
                            psspy.report_output(6)  # Back to default  
                            redirect.psse2py()
                            if os.path.exists(temp_output):
                                os.remove(temp_output)
                        except:
                            pass
                            
                        self.log_message("Parallel method completed successfully", self.accc_log_text)
                        
                    except Exception as parallel_error:
                        self.log_message(f"Parallel method failed: {str(parallel_error)}", self.accc_log_text)
                        self.log_message("Falling back to DSP method (more stable)...", self.accc_log_text)
                        
                        # Clean up any hanging processes
                        try:
                            psspy.report_output(6)  # Reset output
                            redirect.psse2py()
                        except:
                            pass
                        
                        # Fallback to DSP method
                        self.log_message(f"Fallback: Calling psspy.accc_with_dsp_3 (DSP method):", self.accc_log_text)
                        
                        # Option functions for ACCC - exactly like working GINR script
                        optacc = [
                            0, #Tap adjustments 0=disable 1=enable stepping 2=enable direct
                            0, #Interchange 0=disable 1=enable tie flow only 2=enable tie and loads
                            0, #Phase shift 0=disable 1=enable
                            1, #dc tap 0=disable 1=enable
                            1, #switched shunt 0=disable 1=enable 2=enable continuous
                            1, #solution flag 0=FDNS 1=FNSL 2=optimized FDNS
                            0, #Non-divergent 0=disable 1=enable
                            0, #induction motor 0=stall 1=trip
                            0, #induction failure 0=treat contingency as non-converage 1=treak contingency as solved if it converges
                            0, #dispatch mode 0=disable
                            0, #zip archive 0=do not write ZIP 1=write a ZIP
                        ]
                        
                        empty = ""  # Empty variable for use in functions where no input is needed
                        
                        # Ensure output is redirected before ACCC call  
                        redirect.psse2py()
                        self.root.update_idletasks()
                        
                        # Redirect output to prevent terminal input mode
                        temp_output = os.path.join(tempfile.gettempdir(), "psse_accc_fallback_output.tmp")
                        try:
                            psspy.report_output(2, temp_output)  # Redirect output to file
                        except:
                            pass
                        
                        ierr_accc = psspy.accc_with_dsp_3(self.tolerance.get(), optacc, empty, dfx_file, acc_file, empty, empty, empty)
                        
                        # Force output redirection back to Python
                        try:
                            psspy.report_output(6)  # Back to default
                            redirect.psse2py()
                            if os.path.exists(temp_output):
                                os.remove(temp_output)
                        except:
                            pass
                        
                        self.log_message("Successfully completed using DSP fallback method", self.accc_log_text)
                    
                else:
                    # Multiprocessing method - fastest, uses multiple CPU processes (like FastStudy.py)
                    # Note: Maximum benefit achieved when processing multiple cases simultaneously
                    self.log_message(f"Using Multiprocessing method with {self.num_processes.get()} processes:", self.accc_log_text)
                    self.log_message(f"  Tolerance: {self.tolerance.get()}", self.accc_log_text)
                    self.log_message(f"  Processes: {self.num_processes.get()}", self.accc_log_text)
                    self.log_message("  Note: For single cases, use Parallel method. Multiprocessing best for multiple cases.", self.accc_log_text)
                    
                    # Prepare data for multiprocessing worker
                    case_data = (
                        self.sav_file.get(),
                        self.sub_file.get(), 
                        self.mon_file.get(),
                        self.con_file.get(),
                        self.tolerance.get(),
                        os.path.dirname(self.sav_file.get())
                    )
                    
                    # Use multiprocessing pool
                    self.log_message("Starting multiprocessing pool...", self.accc_log_text)
                    with multiprocessing.Pool(self.num_processes.get()) as pool:
                        # For single case, just use one worker (in real scenarios you'd have multiple cases)
                        result = pool.apply(self.accc_worker, (case_data,))
                    
                    # Process result
                    if "error" in result:
                        ierr_accc = 1  # Set error code
                        self.log_message(f"Multiprocessing ACCC failed: {result['error']}", self.accc_log_text)
                    else:
                        ierr_accc = result['accc_error']
                        self.log_message(f"Multiprocessing ACCC completed for {result['case']}", self.accc_log_text)
                        self.log_message(f"  DFAX error: {result['dfax_error']}", self.accc_log_text)
                        self.log_message(f"  ACCC error: {result['accc_error']}", self.accc_log_text)
                        self.log_message(f"  DFX exists: {result['dfx_exists']}", self.accc_log_text)
                        self.log_message(f"  ACC exists: {result['acc_exists']}", self.accc_log_text)
                end_time = tic()
                
                self.log_message(f"ACCC ({self.accc_method.get()} method) returned with code: {ierr_accc}", self.accc_log_text)
                
                if ierr_accc != 0:
                    self.log_message(f"ACCC completed with warnings/errors (code: {ierr_accc})", self.accc_log_text)
                    # Add specific error code meanings
                    if ierr_accc == 1:
                        self.log_message("Error 1: Invalid options or parameters", self.accc_log_text)
                    elif ierr_accc == 2:
                        self.log_message("Error 2: Problem with DFAX file", self.accc_log_text)
                    elif ierr_accc == 3:
                        self.log_message("Error 3: Problem with output file", self.accc_log_text)
                    else:
                        self.log_message(f"Unspecified ACCC error: {ierr_accc}", self.accc_log_text)
                else:
                    self.log_message("ACCC completed successfully", self.accc_log_text)
                
                self.log_message(f"ACCC Analysis completed in {end_time - start_time:.2f} seconds", self.accc_log_text)
                
            except Exception as e:
                self.log_message(f"Exception during ACCC execution: {str(e)}", self.accc_log_text)
                self.log_message("ACCC analysis failed - check PSSE installation and input files", self.accc_log_text)
                return
            
            # Check if files were created
            if os.path.exists(dfx_file):
                dfx_size = os.path.getsize(dfx_file) / 1024  # KB
                self.log_message(f"✓ DFAX file created: {dfx_file} ({dfx_size:.1f} KB)", self.accc_log_text)
            else:
                self.log_message(f"✗ DFAX file not found: {dfx_file}", self.accc_log_text)
                self.log_message("This will prevent ACCC analysis from working!", self.accc_log_text)
                
            if os.path.exists(acc_file):
                acc_size = os.path.getsize(acc_file) / 1024  # KB
                self.log_message(f"✓ ACC results file created: {acc_file} ({acc_size:.1f} KB)", self.accc_log_text)
                
                # Check if the file has meaningful content
                if acc_size < 1:
                    self.log_message("Warning: ACC file is very small - may be empty", self.accc_log_text)
                else:
                    self.log_message("ACC file appears to have results data", self.accc_log_text)
            else:
                self.log_message(f"✗ ACC results file not found: {acc_file}", self.accc_log_text)
                self.log_message("Check PSSE messages above for ACCC analysis errors", self.accc_log_text)
            
            # Ensure PSSE returns control to Python and doesn't hang in terminal mode
            try:
                # Force redirect again to prevent any hanging
                redirect.psse2py()
                # Close any open PSSE output files that might be keeping the process alive
                psspy.report_output(6)  # Close all output files and redirect to default
            except:
                pass  # These might fail if PSSE is already closed
            
            # Process results
            self.log_message("Processing ACCC results...", self.accc_log_text)
            self.process_accc_results(acc_file)
            
            self.log_message("ACCC Analysis completed successfully!", self.accc_log_text)
            
        except Exception as e:
            self.log_message(f"Error during ACCC analysis: {str(e)}", self.accc_log_text)
        finally:
            self.root.after(0, self.analysis_complete)
    
    def perform_dynamic_analysis(self):
        """Perform the Dynamic analysis (runs in separate thread)"""
        try:
            self.log_message("Starting Dynamic Analysis...")
            self.log_message("Initializing PSSE...")
            
            # Initialize PSSE
            redirect.psse2py()
            ierr = psspy.psseinit(0)
            
            if ierr != 0:
                self.log_message(f"PSSE initialization failed with error: {ierr}")
                return
            
            # Load case - following dynamics.py pattern exactly
            self.log_message(f"Loading case: {self.sav_file.get()}")
            psspy.case(self.sav_file.get())
            
            # Solve power flow with proper convergence checking
            self.log_message("Solving power flow...")
            ierr = psspy.fdns([0,0,0,1,1,0,99,0])
            if ierr != 0:
                self.log_message(f"Power flow solution failed with error: {ierr}")
                self.log_message("Attempting alternative power flow solution...")
                ierr = psspy.fnsl([0,0,0,1,1,0,99,0])
                if ierr != 0:
                    self.log_message(f"Alternative power flow also failed: {ierr}")
                    self.log_message("Continuing with potentially unconverged power flow...")
            else:
                self.log_message("Power flow converged successfully")
            
            # Setup dynamics exactly as in dynamics.py
            self.log_message("Setting up dynamics...")
            psspy.cong(0)
            psspy.conl(0,1,1,[0,0],[100.0,0.0,0.0,100.0])
            psspy.conl(0,1,2,[0,0],[100.0,0.0,0.0,100.0])
            psspy.conl(0,1,3,[0,0],[100.0,0.0,0.0,100.0])
            
            # Create converted case file path
            cnv_file = self.cnv_file.get()
            self.log_message(f"Saving converted case: {cnv_file}")
            psspy.save(cnv_file)
            
            # Setup dynamics data
            self.log_message(f"Loading dynamics data: {self.dyr_file.get()}")
            psspy.fact()
            psspy.tysl(0)
            
            # Use the correct PSSE dyre_new API - exactly as in dynamics.py
            try:
                self.log_message("Loading dynamics file with dyre_new...")
                ierr = psspy.dyre_new([1,1,1,1], self.dyr_file.get(), "", "", "")
                if ierr == 0:
                    self.log_message("Successfully loaded dynamics data using dyre_new")
                else:
                    self.log_message(f"dyre_new returned error code: {ierr}")
                    if ierr == 1:
                        self.log_message("Error 1: Invalid STARTINDX value")
                    elif ierr == 3:
                        self.log_message("Error 3: Error opening dynamics file")
                    elif ierr == 4:
                        self.log_message("Error 4: Prerequisite requirements not met")
                    else:
                        self.log_message(f"Unknown error code: {ierr}")
                    # Don't fail here - continue and see if we can still run
                    self.log_message("Continuing despite dynamics loading issues...")
            except Exception as e:
                self.log_message(f"Exception during dynamics loading: {str(e)}")
                self.log_message("Continuing despite dynamics loading exception...")
            
            # Setup channels - exactly as in dynamics.py
            self.log_message("Setting up dynamic simulation channels...")
            psspy.chsb(0,1,[-1,-1,-1,1,1,0])
            psspy.chsb(0,1,[-1,-1,-1,1,2,0])
            psspy.chsb(0,1,[-1,-1,-1,1,3,0])
            psspy.chsb(0,1,[-1,-1,-1,1,4,0])
            psspy.chsb(0,1,[-1,-1,-1,1,5,0])
            psspy.chsb(0,1,[-1,-1,-1,1,6,0])
            psspy.chsb(0,1,[-1,-1,-1,1,7,0])
            
            # Load the user-specified snapshot file - exactly as in dynamics.py
            snp_file = self.snp_file.get()
            self.log_message(f"Creating snapshot: {snp_file}")
            psspy.snap([177,63,8,0,42], snp_file)
            
            # Setup output file BEFORE starting simulation - this is critical
            output_file_input = self.dynamic_output_file.get().strip()
            self.log_message(f"Raw output file input: '{output_file_input}'")
            
            # If no path specified or just filename, use current working directory
            if not output_file_input or output_file_input == "":
                output_file_input = "dynamic_results.outx"
                self.log_message(f"Using default filename: {output_file_input}")
            
            # Ensure we have an absolute path
            if not os.path.isabs(output_file_input):
                output_path = os.path.join(os.getcwd(), output_file_input)
                self.log_message(f"Converting to absolute path: {output_path}")
            else:
                output_path = output_file_input
            
            # Resolve the path to handle any .. or . components
            output_path = str(Path(output_path).resolve())
            self.log_message(f"Final resolved output file path: {output_path}")
            
            # Make sure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and output_dir != "":
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    self.log_message(f"Ensured output directory exists: {output_dir}")
                except Exception as dir_error:
                    self.log_message(f"Could not create output directory: {str(dir_error)}")
                    # Use current directory as fallback
                    output_path = os.path.join(os.getcwd(), os.path.basename(output_path))
                    self.log_message(f"Using fallback path: {output_path}")
            else:
                self.log_message(f"Using current directory for output: {os.getcwd()}")
            
            # Initialize dynamic simulation - following dynamics.py exactly
            self.log_message("Initializing dynamic simulation...")
            
            # First set reconcile flag to 1 (as in dynamics.py)
            try:
                psspy.set_zsorce_reconcile_flag(1)
                self.log_message("Set zsorce reconcile flag to 1")
            except:
                try:
                    psspy.set_zsorce_reconsile_flag(1)  # Try alternate spelling
                    self.log_message("Set zsorce reconcile flag to 1 (alternate spelling)")
                except:
                    self.log_message("Warning: Could not set zsorce reconcile flag")
            
            psspy.set_relang(1,-1,"")
            
            # Start simulation WITH output file specification
            self.log_message("Starting dynamic simulation with output file...")
            self.log_message(f"Using strt() with output file: {output_path}")
            
            # Use strt() instead of strt_2() to specify the output file directly
            ierr = psspy.strt(0, output_path)
            if ierr != 0:
                self.log_message(f"STRT failed with error: {ierr}")
                if ierr == 4:
                    self.log_message("Error 4: Could not open output file - check path and permissions")
                elif ierr == 1:
                    self.log_message("Error 1: Generators are not converted")
                elif ierr == 3:
                    self.log_message("Error 3: Prior initialization modified loads")
                else:
                    self.log_message("This usually indicates initialization problems")
                
                # Try to get more information about what went wrong
                try:
                    # Force a check of initial conditions and try with strt_2 as fallback
                    self.log_message("Attempting fallback with strt_2...")
                    psspy.tysl(0)  # Re-initialize dynamics
                    ierr = psspy.strt_2([0,0],"")  # Try strt_2 as fallback
                    if ierr != 0:
                        raise Exception(f"Both STRT and STRT_2 failed with error {ierr} - cannot continue")
                    else:
                        self.log_message("STRT_2 succeeded as fallback (but may not create output file)")
                except Exception as strt_error:
                    self.log_message(f"Could not recover from STRT failure: {str(strt_error)}")
                    return
            else:
                self.log_message("Dynamic simulation started successfully with output file")
                self.log_message(f"Channel data will be written to: {output_path}")
            
            # Run to fault time
            fault_time = float(self.fault_start_time.get())
            self.log_message(f"Running simulation to fault time: {fault_time}s")
            ierr = psspy.run(0, fault_time, 1, 1, 0)
            if ierr != 0:
                self.log_message(f"Run to fault time failed: {ierr}")
            
            # Apply fault
            fault_bus = int(self.fault_bus.get())
            fault_impedance = float(self.fault_impedance.get())
            self.log_message(f"Applying 3-phase fault at bus {fault_bus}")
            ierr = psspy.dist_3phase_bus_fault(fault_bus, 0, 1, 21.6, [fault_impedance, -0.2E+10])
            if ierr != 0:
                self.log_message(f"Fault application failed: {ierr}")
            
            # Run during fault
            fault_clear_time = fault_time + float(self.fault_duration.get())
            self.log_message(f"Running during fault until: {fault_clear_time}s")
            ierr = psspy.run(0, fault_clear_time, 1, 1, 0)
            if ierr != 0:
                self.log_message(f"Run during fault failed: {ierr}")
            
            # Clear fault
            self.log_message("Clearing fault")
            ierr = psspy.dist_clear_fault(1)
            if ierr != 0:
                self.log_message(f"Fault clearing failed: {ierr}")
            
            # Run post-fault simulation
            sim_time = float(self.simulation_time.get())
            self.log_message(f"Running post-fault simulation until: {sim_time}s")
            ierr = psspy.run(0, sim_time, 1, 1, 0)  
            if ierr != 0:
                self.log_message(f"Post-fault simulation failed: {ierr}")
            
            self.log_message("Dynamic simulation completed!")
            
            # Check if output file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                self.log_message(f"Output file created: {output_path} ({file_size} bytes)")
            else:
                self.log_message(f"Warning: Output file not found at {output_path}")
                # Try to find any .outx files in the directory
                output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
                outx_files = [f for f in os.listdir(output_dir) if f.endswith('.outx')]
                if outx_files:
                    self.log_message(f"Found other .outx files: {outx_files}")
                    # Use the first one found
                    output_path = os.path.join(output_dir, outx_files[0])
                    self.log_message(f"Using output file: {output_path}")
            
            # Optional plotting
            if self.enable_plotting.get() and os.path.exists(output_path):
                self.log_message("Creating plots...")
                try:
                    self.create_dynamic_plots(output_path, fault_bus)
                except Exception as plot_error:
                    self.log_message(f"Plotting failed: {str(plot_error)}")
            
            # Store results info
            self.dynamic_results = {
                'output_file': output_path,
                'fault_bus': fault_bus,
                'fault_time': fault_time,
                'fault_duration': self.fault_duration.get(),
                'simulation_time': sim_time
            }
            
            # Update results display
            self.root.after(0, self.display_dynamic_results)
            
        except Exception as e:
            self.log_message(f"Error during dynamic analysis: {str(e)}")
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}")
        finally:
            self.root.after(0, self.analysis_complete)
    
    def generate_p3p6_contingencies(self):
        """Generate P3_P6 contingencies using allcon_rev1.py"""
        if not self.validate_p3p6_inputs():
            return
        
        # Start progress indication
        self.p3p6_progress_bar.start()
        self.is_running.set(True)
        
        # Clear previous log
        self.p3p6_log_text.delete(1.0, tk.END)
        
        # Run contingency generation in separate thread
        thread = threading.Thread(target=self.p3p6_worker, daemon=True)
        thread.start()
    
    def validate_p3p6_inputs(self):
        """Validate P3_P6 inputs"""
        input_file = self.p3p6_input_file.get().strip()
        output_file = self.p3p6_output_file.get().strip()
        
        if not input_file:
            messagebox.showerror("Input Error", "Please select an input .con file")
            return False
        
        if not os.path.exists(input_file):
            messagebox.showerror("File Error", f"Input file not found: {input_file}")
            return False
        
        if not output_file:
            messagebox.showerror("Input Error", "Please specify an output .con file")
            return False
        
        # Check if output directory exists, create if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                self.log_message(f"Created output directory: {output_dir}", self.p3p6_log_text)
            except Exception as e:
                messagebox.showerror("Directory Error", f"Cannot create output directory: {str(e)}")
                return False
        
        return True
    
    def p3p6_worker(self):
        """Worker thread for P3_P6 contingency generation"""
        try:
            input_file = self.p3p6_input_file.get().strip()
            output_file = self.p3p6_output_file.get().strip()
            
            self.log_message(f"Starting P3_P6 contingency generation...", self.p3p6_log_text)
            self.log_message(f"Input file: {input_file}", self.p3p6_log_text)
            self.log_message(f"Output file: {output_file}", self.p3p6_log_text)
            
            # Import the contingency generation logic from allcon_rev1.py
            from itertools import combinations
            import re
            
            # Define helper functions (copied from allcon_rev1.py)
            def is_comment_or_blank(line):
                s = line.strip()
                if not s:
                    return True
                if s.startswith(("!", "//", "*", "#")):
                    return True
                if s.startswith("/*") and s.endswith("*/"):
                    return True
                return False
            
            def split_row(raw):
                for delim in (",", ";", "|", "\t"):
                    if delim in raw:
                        return [p.strip() for p in raw.strip().split(delim) if p.strip()]
                return [p for p in raw.strip().split() if p]
            
            def try_int(s):
                try:
                    return int(s)
                except Exception:
                    return None
            
            def parse_row(fields):
                ints = [try_int(f) for f in fields]
                ints = [i for i in ints if i is not None]
                if len(ints) < 2:
                    raise ValueError(f"Row does not contain two bus numbers: {fields}")
                
                from_bus = ints[0]
                to_bus = ints[1]
                
                # Circuit identifier
                ckt = "1"
                if len(ints) >= 3:
                    ckt = str(ints[2])
                else:
                    for tok in fields:
                        if tok.upper() == "CKT":
                            continue
                        if try_int(tok) is None and re.fullmatch(r"[A-Za-z0-9]+", tok):
                            ckt = tok
                            break
                
                # Names from remaining nonnumeric tokens
                nonnum = [f for f in fields if try_int(f) is None and f.upper() != "CKT"]
                from_name = f"BUS {from_bus}"
                to_name = f"BUS {to_bus}"
                if len(nonnum) >= 2:
                    from_name, to_name = nonnum[0], nonnum[1]
                elif len(nonnum) == 1:
                    from_name = nonnum[0]
                
                return from_bus, to_bus, ckt, from_name, to_name
            
            def parse_line(line):
                if is_comment_or_blank(line):
                    return None
                
                # Preserve inline C-style comment for names if present
                inline_comment = None
                m = re.search(r"/\*(.*?)\*/", line)
                if m:
                    inline_comment = m.group(1).strip()
                
                # Remove C-style comments to parse tokens safely
                cleaned = re.sub(r"/\*.*?\*/", " ", line)
                fields = split_row(cleaned)
                
                try:
                    from_bus, to_bus, ckt, from_name, to_name = parse_row(fields)
                except Exception:
                    # Second chance: just harvest numbers
                    nums = [int(n) for n in re.findall(r"\b\d+\b", cleaned)]
                    if len(nums) < 2:
                        return None
                    from_bus, to_bus = nums[0], nums[1]
                    ckt = str(nums[2]) if len(nums) >= 3 else "1"
                    from_name, to_name = f"BUS {from_bus}", f"BUS {to_bus}"
                
                # If inline comment appears like "NAME1 - NAME2", use as names
                if inline_comment:
                    if "-" in inline_comment:
                        left, right = [p.strip() for p in inline_comment.split("-", 1)]
                        if left:
                            from_name = left
                        if right:
                            to_name = right
                    else:
                        from_name = inline_comment
                
                return from_bus, to_bus, ckt, from_name, to_name
            
            # Parse input file
            rows = []
            with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    parsed = parse_line(line)
                    if parsed:
                        rows.append(parsed)
                        self.log_message(f"Parsed line {line_num}: {parsed[3]} - {parsed[4]}", self.p3p6_log_text)
            
            self.log_message(f"Parsed {len(rows)} transmission lines", self.p3p6_log_text)
            
            if not rows:
                self.log_message("No valid transmission lines found in input file", self.p3p6_log_text)
                return
            
            # Generate contingencies
            singles_count = len(rows)
            pairs_count = len(rows) * (len(rows) - 1) // 2
            total_contingencies = singles_count + pairs_count
            
            self.log_message(f"Generating {singles_count} single and {pairs_count} double contingencies", self.p3p6_log_text)
            
            with open(output_file, "w", encoding="utf-8") as out:
                # Singles (P3)
                for idx, (fb, tb, ckt, fn, tn) in enumerate(rows, start=1):
                    label = f"DB_{idx}"
                    out.write(f"CONTINGENCY '{label}'\n")
                    out.write(f"   OPEN BRANCH FROM BUS {fb} TO BUS {tb} CKT {ckt}                   /* {fn} - {tn}\n")
                    out.write("END\n")
                
                if rows:
                    out.write("\n")
                
                # Pairs (P6)
                indexed = list(enumerate(rows, start=1))
                for (i, a), (j, b) in combinations(indexed, 2):
                    (fb1, tb1, ckt1, fn1, tn1) = a
                    (fb2, tb2, ckt2, fn2, tn2) = b
                    label = f"DB_{i}_{j}"
                    out.write(f"CONTINGENCY '{label}'\n")
                    out.write(f"   OPEN BRANCH FROM BUS {fb1} TO BUS {tb1} CKT {ckt1}                   /* {fn1} - {tn1}\n")
                    out.write(f"   OPEN BRANCH FROM BUS {fb2} TO BUS {tb2} CKT {ckt2}                   /* {fn2} - {tn2}\n")
                    out.write("END\n")
                
                # Extra END at the end of file
                out.write("\nEND\n")
            
            self.log_message(f"Successfully generated {total_contingencies} contingencies", self.p3p6_log_text)
            self.log_message(f"Output saved to: {output_file}", self.p3p6_log_text)
            
            # Show completion message
            file_size = os.path.getsize(output_file)
            self.root.after(0, lambda: messagebox.showinfo(
                "Success", 
                f"P3_P6 contingencies generated successfully!\n\n"
                f"Total contingencies: {total_contingencies}\n"
                f"- Single (P3): {singles_count}\n" 
                f"- Double (P6): {pairs_count}\n\n"
                f"Output file: {output_file}\n"
                f"File size: {file_size:,} bytes"
            ))
            
        except Exception as e:
            self.log_message(f"Error generating contingencies: {str(e)}", self.p3p6_log_text)
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}", self.p3p6_log_text)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to generate contingencies:\n{str(e)}"))
        finally:
            self.root.after(0, lambda: self.p3p6_progress_bar.stop())
            self.is_running.set(False)
    
    def process_accc_results(self, acc_file):
        """Process the ACCC results"""
        try:
            accdata = arrbox.CONTINGENCY_PP(acc_file)
            summary = accdata.summary()
            
            branches = summary.melement
            ratings = summary.rating.b
            buses = summary.mvbuslabel
            contingencies = summary.colabel
            
            self.log_message(f"Processing {len(contingencies)} contingencies...", self.accc_log_text)
            self.log_message(f"Found {len(branches)} monitored branches", self.accc_log_text)
            self.log_message(f"Found {len(buses)} monitored buses", self.accc_log_text)
            
            thermal_threshold = self.thermal_threshold.get() / 100.0
            voltage_high = self.voltage_high.get()
            voltage_low = self.voltage_low.get()
            
            self.log_message(f"Thermal threshold: {thermal_threshold*100:.1f}%", self.accc_log_text)
            self.log_message(f"Voltage limits: {voltage_low:.2f} - {voltage_high:.2f} p.u.", self.accc_log_text)
            
            converged_count = 0
            thermal_violation_count = 0
            voltage_violation_count = 0
            
            for i, contingency in enumerate(contingencies):
                if not self.is_running.get():
                    break
                
                solution = accdata.solution(contingency)
                flows = solution.mvaflow
                volts = solution.volts
                cnvflg = solution.cnvflag
                
                # Debug: log first few convergence flags to understand the data
                if i < 5:  # Only log first 5 for debugging
                    self.log_message(f"Debug - Contingency {i+1} '{contingency}': cnvflg = {cnvflg}, type = {type(cnvflg)}", self.accc_log_text)
                
                # Convert convergence flag to readable status
                # PSSE ACCC returns boolean values: True = converged, False = non-converged
                if cnvflg is True or cnvflg == True or str(cnvflg).upper() == 'TRUE':
                    converged_status = "Yes"
                    converged_count += 1
                    is_converged = True
                elif cnvflg is False or cnvflg == False or str(cnvflg).upper() == 'FALSE':
                    converged_status = "BLOWN UP"
                    is_converged = False
                elif cnvflg == 0 or cnvflg == '0' or str(cnvflg).upper().startswith('SOLVED') or str(cnvflg).upper().startswith('CONVERGED'):
                    converged_status = "Yes"
                    converged_count += 1
                    is_converged = True
                elif cnvflg == 1 or cnvflg == '1' or str(cnvflg).upper().startswith('BLOWN') or str(cnvflg).upper().startswith('DIVERG'):
                    converged_status = "BLOWN UP"
                    is_converged = False
                elif cnvflg == 2 or cnvflg == '2' or str(cnvflg).upper().startswith('MAX') or str(cnvflg).upper().startswith('ITER'):
                    converged_status = "MAX ITER"
                    is_converged = False
                else:
                    converged_status = f"Unknown ({cnvflg})"
                    is_converged = False
                
                # If contingency didn't converge, add it to thermal violations list for reporting
                if not is_converged:
                    violation = {
                        'contingency': contingency,
                        'converged': converged_status,
                        'branch': 'N/A - Non-converged',
                        'rating': 0.0,
                        'flow': 0.0,
                        'loading': 0.0
                    }
                    self.accc_thermal_violations.append(violation)
                    continue  # Skip thermal/voltage processing for non-converged cases
                
                # Check thermal violations (only for converged cases)
                try:
                    for j in range(len(branches)):
                        try:
                            branch = branches[j] if j < len(branches) else f"Branch_{j}"
                            rating = ratings[j] if j < len(ratings) else 0
                            flow = flows[j] if j < len(flows) else 0
                            
                            if rating and rating > 0:  # Make sure we have a valid rating
                                loading = abs(flow) / rating
                                
                                # Only report violations above the threshold
                                if loading > thermal_threshold:
                                    violation = {
                                        'contingency': contingency,
                                        'converged': converged_status,
                                        'branch': str(branch),
                                        'rating': float(rating),
                                        'flow': abs(float(flow)),
                                        'loading': float(loading * 100)
                                    }
                                    self.accc_thermal_violations.append(violation)
                                    thermal_violation_count += 1
                        except Exception as branch_error:
                            # Skip this branch if there's an error
                            continue
                except Exception as thermal_error:
                    self.log_message(f"Error processing thermal violations for {contingency}: {str(thermal_error)}", self.accc_log_text)
                
                # Check voltage violations (only for converged cases)
                try:
                    for k in range(len(buses)):
                        try:
                            bus = buses[k] if k < len(buses) else f"Bus_{k}"
                            volt = volts[k] if k < len(volts) else 1.0
                            
                            if volt and (volt > voltage_high or volt < voltage_low):
                                violation = {
                                    'contingency': contingency,
                                    'converged': converged_status,
                                    'bus': str(bus),
                                    'voltage': float(volt)
                                }
                                self.accc_voltage_violations.append(violation)
                                voltage_violation_count += 1
                        except Exception as bus_error:
                            # Skip this bus if there's an error
                            continue
                except Exception as voltage_error:
                    self.log_message(f"Error processing voltage violations for {contingency}: {str(voltage_error)}", self.accc_log_text)
            
            # Debug: Analysis completed - convergence flags already processed above
            
            # Log processing summary
            self.log_message(f"Analysis Summary:", self.accc_log_text)
            self.log_message(f"  Total contingencies: {len(contingencies)}", self.accc_log_text)
            self.log_message(f"  Converged contingencies: {converged_count}", self.accc_log_text)
            self.log_message(f"  Non-converged contingencies: {len(contingencies) - converged_count}", self.accc_log_text)
            self.log_message(f"  Thermal violations found: {thermal_violation_count}", self.accc_log_text)
            self.log_message(f"  Voltage violations found: {voltage_violation_count}", self.accc_log_text)
            
            # Update GUI with results
            self.root.after(0, self.display_accc_results)
            
        except Exception as e:
            self.log_message(f"Error processing ACCC results: {str(e)}", self.accc_log_text)
    
    def display_accc_results(self):
        """Display ACCC results in the GUI trees"""
        # Display thermal violations
        for violation in self.accc_thermal_violations:
            # Handle both dictionary format (new) and tuple format (legacy)
            if isinstance(violation, dict):
                display_data = (
                    violation['contingency'],
                    violation['converged'],
                    violation['branch'],
                    f"{violation['rating']:.1f}" if violation['rating'] > 0 else "N/A",
                    f"{violation['flow']:.1f}" if violation['flow'] > 0 else "N/A",
                    f"{violation['loading']:.1f}" if violation['loading'] > 0 else "N/A"
                )
            else:
                # Legacy tuple format
                display_data = (
                    violation[0],  # Contingency
                    "Yes" if violation[1] == 0 else ("BLOWN UP" if violation[1] == 1 else "MAX ITER"),  # Converged
                    violation[2],  # Branch
                    f"{violation[3]:.1f}",  # Rating
                    f"{violation[4]:.1f}",  # Flow
                    f"{violation[5]:.1f}"   # Loading %
                )
            self.thermal_tree.insert("", tk.END, values=display_data)
        
        # Display voltage violations
        for violation in self.accc_voltage_violations:
            # Handle both dictionary format (new) and tuple format (legacy)
            if isinstance(violation, dict):
                display_data = (
                    violation['contingency'],
                    violation['converged'],
                    violation['bus'],
                    f"{violation['voltage']:.3f}"
                )
            else:
                # Legacy tuple format
                display_data = (
                    violation[0],  # Contingency
                    "Yes" if violation[1] == 0 else ("BLOWN UP" if violation[1] == 1 else "MAX ITER"),  # Converged
                    violation[2],  # Bus
                    f"{violation[3]:.3f}"   # Voltage
                )
            self.voltage_tree.insert("", tk.END, values=display_data)
        
        self.log_message(f"Found {len(self.accc_thermal_violations)} thermal violations", self.accc_log_text)
        self.log_message(f"Found {len(self.accc_voltage_violations)} voltage violations", self.accc_log_text)
        
        # Switch to results tab to show results
        self.main_notebook.select(2)  # Select Results tab
    
    def display_dynamic_results(self):
        """Display comprehensive Dynamic simulation results summary"""
        if self.dynamic_results:
            output_file = self.dynamic_results['output_file']
            fault_bus = self.dynamic_results['fault_bus']
            
            # Basic simulation info
            results_text = f"""Dynamic Simulation Results Summary:
=====================================

Simulation Configuration:
- Output File: {output_file}
- Fault Bus: {fault_bus}
- Fault Start Time: {self.dynamic_results['fault_time']} seconds
- Fault Duration: {self.dynamic_results['fault_duration']} seconds  
- Total Simulation Time: {self.dynamic_results['simulation_time']} seconds

"""
            
            # Try to analyze the output file for additional details
            try:
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file) / 1024  # KB
                    results_text += f"Output File Analysis:\n"
                    results_text += f"- File Size: {file_size:.1f} KB\n"
                    results_text += f"- File Created: {time.ctime(os.path.getctime(output_file))}\n"
                    
                    # Try to use dyntools to get channel information
                    try:
                        import dyntools
                        outobj = dyntools.CHNF(output_file)
                        short_title, chanid, chandata = outobj.get_data()
                        
                        results_text += f"\nChannel Data Analysis:\n"
                        results_text += f"- Total Channels: {len(chanid)}\n"
                        results_text += f"- Time Points: {len(chandata[list(chandata.keys())[0]])}\n"
                        results_text += f"- Time Range: {chandata['time'][0]:.3f} to {chandata['time'][-1]:.3f} seconds\n"
                        
                        # List first few channels
                        results_text += f"\nAvailable Channels (first 10):\n"
                        channel_names = list(chanid.keys())[:10]
                        for i, ch_id in enumerate(channel_names, 1):
                            ch_name = chanid[ch_id]
                            results_text += f"  {i:2d}. Channel {ch_id}: {ch_name}\n"
                        
                        if len(chanid) > 10:
                            results_text += f"  ... and {len(chanid) - 10} more channels\n"
                            
                        # Find key channels related to fault bus
                        fault_related = []
                        for ch_id, ch_name in chanid.items():
                            if str(fault_bus) in ch_name:
                                fault_related.append((ch_id, ch_name))
                        
                        if fault_related:
                            results_text += f"\nFault Bus {fault_bus} Related Channels:\n"
                            for ch_id, ch_name in fault_related[:5]:  # Show first 5
                                results_text += f"  - Channel {ch_id}: {ch_name}\n"
                                
                    except Exception as dyn_error:
                        results_text += f"\nNote: Could not analyze channel data with dyntools: {str(dyn_error)}\n"
                        
                else:
                    results_text += f"Warning: Output file not found at {output_file}\n"
                    
            except Exception as e:
                results_text += f"\nNote: Could not analyze output file: {str(e)}\n"
            
            results_text += f"""
Simulation Status: ✓ COMPLETED SUCCESSFULLY

Analysis Options:
1. View plots using the 'Enable Plotting' option during simulation
2. Open output file ({os.path.basename(output_file)}) in PSSE Plot
3. Use PSSE's dynamic analysis tools for detailed waveform analysis
4. The channel data file contains:
   - Bus voltages and angles
   - Generator variables (speed, power, etc.)
   - System frequency response
   - Fault response characteristics

Plot Information:
- Automatic plots show bus angles and terminal voltages
- Additional machine variables plotted when available
- Plot book name: 'Dynamic Analysis Plot Book'

File Locations:
- Output File: {output_file}
- Working Directory: {os.path.dirname(output_file)}
            """
            
            self.dynamic_results_text.delete(1.0, tk.END)
            self.dynamic_results_text.insert(tk.END, results_text)
            
            # Switch to results tab
            self.main_notebook.select(2)  # Select Results tab
    
    def export_accc_results(self):
        """Export ACCC results to Excel file"""
        if not self.accc_thermal_violations and not self.accc_voltage_violations:
            messagebox.showwarning("Warning", "No ACCC results to export")
            return
        
        try:
            # Try to create Excel workbook with error handling
            try:
                wb = excelpy.workbook()
            except Exception as excel_error:
                # If Excel export fails, create a CSV instead
                self.log_message(f"Excel export failed: {str(excel_error)}", self.accc_log_text)
                self.export_to_csv()
                return
            
            if self.accc_thermal_violations:
                wb.worksheet_rename("Thermal Violations")
                wb.set_range(1,1, ['Contingency', 'Converge Flag', 'Branch', 'Rating (MVA)', 'Flow (MVA)', 'Loading (%)'])
                
                # Convert violations to proper format for Excel export
                thermal_export_data = []
                for violation in self.accc_thermal_violations:
                    if isinstance(violation, dict):
                        thermal_export_data.append([
                            violation['contingency'],
                            violation['converged'],
                            violation['branch'],
                            violation['rating'] if violation['rating'] > 0 else 'N/A',
                            violation['flow'] if violation['flow'] > 0 else 'N/A',
                            violation['loading'] if violation['loading'] > 0 else 'N/A'
                        ])
                    else:
                        # Handle legacy tuple format
                        conv_text = "Yes" if violation[1] == 0 else ("BLOWN UP" if violation[1] == 1 else "MAX ITER")
                        thermal_export_data.append([
                            violation[0], conv_text, violation[2], violation[3], violation[4], violation[5]
                        ])
                
                wb.set_range(2, 1, thermal_export_data)
                wb.autofit_columns((1, 1, 1, 6))
            
            if self.accc_voltage_violations:
                if self.accc_thermal_violations:
                    wb.worksheet_add_after("Voltage Violations")
                    wb.set_active_sheet("Voltage Violations")
                else:
                    wb.worksheet_rename("Voltage Violations")
                wb.set_range(1,1, ['Contingency', 'Converge Flag', 'Bus', 'Voltage'])
                
                # Convert violations to proper format for Excel export
                voltage_export_data = []
                for violation in self.accc_voltage_violations:
                    if isinstance(violation, dict):
                        voltage_export_data.append([
                            violation['contingency'],
                            violation['converged'],
                            violation['bus'],
                            violation['voltage']
                        ])
                    else:
                        # Handle legacy tuple format
                        conv_text = "Yes" if violation[1] == 0 else ("BLOWN UP" if violation[1] == 1 else "MAX ITER")
                        voltage_export_data.append([
                            violation[0], conv_text, violation[2], violation[3]
                        ])
                
                wb.set_range(2, 1, voltage_export_data)
                wb.autofit_columns((1, 1, 1, 4))
            
            wb.save(self.accc_output_file.get())
            wb.close()
            
            self.log_message(f"ACCC Results exported to {self.accc_output_file.get()}", self.accc_log_text)
            messagebox.showinfo("Success", f"ACCC Results exported to {self.accc_output_file.get()}")
            
        except Exception as e:
            self.log_message(f"Error exporting ACCC results: {str(e)}", self.accc_log_text)
            messagebox.showerror("Error", f"Failed to export ACCC results: {str(e)}")
    
    def export_to_csv(self):
        """Export ACCC results to CSV files as fallback"""
        try:
            import csv
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export thermal violations
            if self.accc_thermal_violations:
                csv_filename = f"ACCC_Thermal_Violations_{timestamp}.csv"
                with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Contingency', 'Converged', 'Branch', 'Rating (MVA)', 'Flow (MVA)', 'Loading (%)'])
                    
                    for violation in self.accc_thermal_violations:
                        if isinstance(violation, dict):
                            writer.writerow([
                                violation['contingency'],
                                violation['converged'],
                                violation['branch'],
                                violation['rating'] if violation['rating'] > 0 else 'N/A',
                                violation['flow'] if violation['flow'] > 0 else 'N/A',
                                violation['loading'] if violation['loading'] > 0 else 'N/A'
                            ])
                        else:
                            conv_text = "Yes" if violation[1] == 0 else ("BLOWN UP" if violation[1] == 1 else "MAX ITER")
                            writer.writerow([violation[0], conv_text, violation[2], violation[3], violation[4], violation[5]])
                
                self.log_message(f"Thermal violations exported to: {csv_filename}", self.accc_log_text)
            
            # Export voltage violations
            if self.accc_voltage_violations:
                csv_filename = f"ACCC_Voltage_Violations_{timestamp}.csv"
                with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Contingency', 'Converged', 'Bus', 'Voltage'])
                    
                    for violation in self.accc_voltage_violations:
                        if isinstance(violation, dict):
                            writer.writerow([
                                violation['contingency'],
                                violation['converged'],
                                violation['bus'],
                                violation['voltage']
                            ])
                        else:
                            conv_text = "Yes" if violation[1] == 0 else ("BLOWN UP" if violation[1] == 1 else "MAX ITER")
                            writer.writerow([violation[0], conv_text, violation[2], violation[3]])
                
                self.log_message(f"Voltage violations exported to: {csv_filename}", self.accc_log_text)
            
            messagebox.showinfo("Success", "ACCC results exported to CSV files successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV: {str(e)}")
            self.log_message(f"CSV export error: {str(e)}", self.accc_log_text)
    
    def export_psse_violations_report(self):
        """Export violations using PSSE's built-in violations_report function"""
        try:
            # Check if we have processed results
            acc_file = self.accc_output_file.get()
            if not acc_file or not os.path.exists(acc_file):
                messagebox.showwarning("Warning", "No ACC file found. Please run ACCC analysis first.")
                return
            
            import arrbox
            from datetime import datetime
            
            self.log_message(f"Opening ACC file: {acc_file}", self.accc_log_text)
            
            # Create violations report using PSSE's built-in function
            accobj = arrbox.CONTINGENCY_PP(acc_file)
            
            # Check if ACC object was created successfully
            if hasattr(accobj, 'ierr') and accobj.ierr != 0:
                self.log_message(f"ACC object creation error: {accobj.ierr}", self.accc_log_text)
                messagebox.showerror("Error", f"Failed to open ACC file (error code: {accobj.ierr})")
                return
            
            # Try to get summary first to validate the file
            try:
                summary = accobj.summary()
                if hasattr(summary, 'ierr') and summary.ierr != 0:
                    self.log_message(f"ACC summary error: {summary.ierr}", self.accc_log_text)
                    messagebox.showerror("Error", f"Failed to read ACC file summary (error code: {summary.ierr})")
                    return
                    
                self.log_message(f"ACC file validated successfully", self.accc_log_text)
                
                # Log available ratings if they exist
                if hasattr(summary, 'ratename') and summary.ratename:
                    self.log_message(f"Available ratings: {summary.ratename}", self.accc_log_text)
                    rating_to_use = summary.ratename[0]
                else:
                    self.log_message("No specific ratings found, using defaults", self.accc_log_text)
                    rating_to_use = None
                    
            except Exception as summary_error:
                self.log_message(f"Summary check failed: {str(summary_error)}", self.accc_log_text)
                rating_to_use = None
            
            # Set solution options for violations report
            try:
                if rating_to_use:
                    accobj.solution_options(
                        stype='contingency',
                        busmsm=0.5,           # Bus mismatch tolerance
                        sysmsm=5.0,           # System mismatch tolerance
                        rating=rating_to_use, # Use detected rating
                        flowlimit=90.0,       # 90% loading threshold
                        swdrating=rating_to_use
                    )
                    self.log_message(f"Solution options set with rating: {rating_to_use}", self.accc_log_text)
                else:
                    # Use minimal options without rating specification
                    accobj.solution_options(
                        stype='contingency',
                        flowlimit=90.0        # 90% loading threshold
                    )
                    self.log_message("Solution options set with defaults", self.accc_log_text)
            except Exception as options_error:
                self.log_message(f"Solution options error: {str(options_error)}", self.accc_log_text)
                # Continue without setting options
            
            # Generate violations report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            violations_report_file = f"PSSE_Violations_Report_{timestamp}.txt"
            
            self.log_message(f"Creating PSSE violations report: {violations_report_file}", self.accc_log_text)
            ierr = accobj.violations_report(rptfile=violations_report_file)
            
            if ierr == 0:
                self.log_message(f"PSSE violations report created successfully: {violations_report_file}", self.accc_log_text)
                messagebox.showinfo("Success", f"PSSE violations report created successfully:\n{violations_report_file}")
            else:
                self.log_message(f"PSSE violations report error code: {ierr}", self.accc_log_text)
                messagebox.showerror("Error", f"Failed to create PSSE violations report (error code: {ierr})")
                
        except Exception as e:
            error_msg = str(e)
            self.log_message(f"PSSE violations report error: {error_msg}", self.accc_log_text)
            messagebox.showerror("Error", f"Failed to create PSSE violations report: {error_msg}")
            
            # Provide additional troubleshooting info
            if "acc_summary" in error_msg:
                self.log_message("Suggestion: Try running a fresh ACCC analysis first", self.accc_log_text)
    
    def view_dynamic_results(self):
        """Open dynamic results for viewing and optionally create plots"""
        if not self.dynamic_results or not os.path.exists(self.dynamic_results.get('output_file', '')):
            messagebox.showwarning("Warning", "No dynamic results available or output file not found")
            return
        
        try:
            output_file = self.dynamic_results['output_file']
            fault_bus = self.dynamic_results['fault_bus']
            
            # Ask user what they want to do with the results
            from tkinter import simpledialog
            options = [
                "1. View file location and summary",
                "2. Create/recreate dynamic plots", 
                "3. Open output file in system default program",
                "4. Show detailed channel information"
            ]
            
            choice = simpledialog.askstring(
                "Dynamic Results Viewer",
                f"Dynamic simulation results from fault at bus {fault_bus}:\n{output_file}\n\n" +
                "What would you like to do?\n" + "\n".join(options) + "\n\n" +
                "Enter choice (1-4):"
            )
            
            if choice == "1":
                # Show file information
                file_size = os.path.getsize(output_file) / 1024  # KB
                messagebox.showinfo("Dynamic Results Info", 
                    f"Output File: {output_file}\n"
                    f"File Size: {file_size:.1f} KB\n"
                    f"Fault Bus: {fault_bus}\n"
                    f"Simulation Time: {self.dynamic_results['simulation_time']} seconds\n\n"
                    "Use PSSE Plot or other tools to visualize waveforms.")
                    
            elif choice == "2":
                # Create plots
                if not PSSE_AVAILABLE:
                    messagebox.showerror("Error", "PSSE modules not available for plotting")
                    return
                    
                self.log_message("Creating dynamic plots from results viewer...")
                try:
                    self.create_dynamic_plots(output_file, fault_bus)
                    messagebox.showinfo("Success", 
                        "Dynamic plots created successfully!\n"
                        "Plot book: 'Dynamic Analysis Plot Book'\n"
                        "Contains bus angles and voltage magnitude plots.")
                except Exception as plot_error:
                    messagebox.showerror("Plot Error", f"Failed to create plots: {str(plot_error)}")
                    
            elif choice == "3":
                # Try to open file with system default
                try:
                    import subprocess
                    if os.name == 'nt':  # Windows
                        os.startfile(output_file)
                    else:  # Unix/Linux/Mac
                        subprocess.call(['xdg-open', output_file])
                except Exception as open_error:
                    messagebox.showerror("Error", f"Could not open file: {str(open_error)}")
                    
            elif choice == "4":
                # Show detailed channel information using dyntools
                try:
                    import dyntools
                    outobj = dyntools.CHNF(output_file)
                    short_title, chanid, chandata = outobj.get_data()
                    
                    # Create a detailed channel list
                    channel_info = f"Channel Information for {os.path.basename(output_file)}:\n"
                    channel_info += f"Total Channels: {len(chanid)}\n"
                    channel_info += f"Time Points: {len(chandata['time'])}\n"
                    channel_info += f"Time Range: {chandata['time'][0]:.3f} to {chandata['time'][-1]:.3f} seconds\n\n"
                    
                    channel_info += "Available Channels:\n"
                    for ch_id, ch_name in list(chanid.items())[:20]:  # Show first 20
                        channel_info += f"  {ch_id:3d}: {ch_name}\n"
                    
                    if len(chanid) > 20:
                        channel_info += f"  ... and {len(chanid) - 20} more channels\n"
                    
                    # Show in a scrollable text window
                    info_window = tk.Toplevel(self.root)
                    info_window.title("Channel Information")
                    info_window.geometry("600x400")
                    
                    text_widget = scrolledtext.ScrolledText(info_window, wrap=tk.WORD)
                    text_widget.pack(fill="both", expand=True, padx=10, pady=10)
                    text_widget.insert(tk.END, channel_info)
                    text_widget.config(state="disabled")
                    
                except Exception as dyn_error:
                    messagebox.showerror("Error", f"Could not analyze channels: {str(dyn_error)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process dynamic results: {str(e)}")
    
    def create_dynamic_plots(self, output_file, fault_bus):
        """Create comprehensive dynamic plots similar to dynamics.py"""
        try:
            self.log_message("Opening channel data file for plotting...")
            pssplot.openchandatafile(output_file)
            
            self.log_message("Creating Dynamic Analysis Plot Book...")
            pssplot.plot_book("Dynamic Analysis Plot Book")
            
            # Page 1: Bus Angles
            self.log_message("Creating Page 1: Bus Angles...")
            pssplot.plot_page("Dynamic Analysis Plot Book", 1, "", [2,0,2,2], [1.,1.])
            pssplot.plot_plot("Dynamic Analysis Plot Book", 1, 1, "", [0,0,1,1,1,2,0], [0.0,0.0,100.,100.])
            pssplot.plot_plot_chng("Dynamic Analysis Plot Book", 1, 1, "", [0,0,1,1,1,2,0], [0.0,0.0,0.0,0.0])
            
            # Get bus info from case for better labeling
            try:
                # Try to get actual bus name/voltage from PSSE case
                ierr, bus_name = psspy.notona(fault_bus)
                ierr, base_kv = psspy.busdt1(fault_bus, 'BASE')
                bus_label = f"{fault_bus}[{bus_name.strip():<12} {base_kv:6.3f}]1"
            except:
                # Fallback to generic labeling
                bus_label = f"{fault_bus}[BUS-{fault_bus}       21.600]1"
            
            # Plot first bus angle
            try:
                channel_name = f"1 - ANGL   {bus_label}"
                pssplot.plot_trace_channel("Dynamic Analysis Plot Book", 1, 1, 0, 
                    f"{channel_name} : {output_file}", 1, output_file,
                    channel_name, [-1,-1,-1,16777215])
                self.log_message(f"Plotted angle for bus {fault_bus}")
            except Exception as e:
                self.log_message(f"Could not plot angle for bus {fault_bus}: {str(e)}")
            
            # Set scale for angles plot
            pssplot.plot_plot_chng("Dynamic Analysis Plot Book", 1, 1, "", [0,0,1,1,1,2,0], [-0.02,0.0,50.0,0.0])
            
            # Try to plot second bus angle (adjacent bus)
            try:
                next_bus = fault_bus + 1
                try:
                    # Try to get actual bus name/voltage for second bus
                    ierr, bus_name2 = psspy.notona(next_bus)
                    ierr, base_kv2 = psspy.busdt1(next_bus, 'BASE')
                    bus_label2 = f"{next_bus}[{bus_name2.strip():<12} {base_kv2:6.3f}]1"
                except:
                    bus_label2 = f"{next_bus}[BUS-{next_bus}       21.600]1"
                
                channel_name2 = f"2 - ANGL   {bus_label2}"
                pssplot.plot_trace_channel("Dynamic Analysis Plot Book", 1, 1, 1,
                    f"{channel_name2} : {output_file}", 1, output_file,
                    channel_name2, [-1,-1,-1,16777215])
                self.log_message(f"Plotted angle for bus {next_bus}")
            except Exception as e:
                self.log_message(f"Could not plot angle for bus {fault_bus+1}: {str(e)}")
            
            # Page 2: Terminal Voltages  
            self.log_message("Creating Page 2: Terminal Voltages...")
            pssplot.plot_page("Dynamic Analysis Plot Book", 2, "", [2,0,2,2], [1.,1.])
            pssplot.plot_plot("Dynamic Analysis Plot Book", 2, 1, "", [0,0,1,1,1,2,0], [0.0,0.0,100.,100.])
            pssplot.plot_plot_chng("Dynamic Analysis Plot Book", 2, 1, "", [0,0,1,1,1,2,0], [0.0,0.0,0.0,0.0])
            
            # Plot voltage magnitude traces
            try:
                # First bus voltage magnitude
                voltage_channel1 = f"19 - ETRM   {bus_label}"
                pssplot.plot_trace_channel("Dynamic Analysis Plot Book", 2, 1, 0,
                    f"{voltage_channel1} : {output_file}", 1, output_file,
                    voltage_channel1, [-1,-1,-1,16777215])
                self.log_message(f"Plotted voltage magnitude for bus {fault_bus}")
            except Exception as e:
                self.log_message(f"Could not plot voltage magnitude for bus {fault_bus}: {str(e)}")
            
            # Set scale for voltage plot  
            pssplot.plot_plot_chng("Dynamic Analysis Plot Book", 2, 1, "", [0,0,1,1,1,2,0], [-0.02,0.0,2.5,0.0])
            
            # Second bus voltage magnitude
            try:
                voltage_channel2 = f"20 - ETRM   {bus_label2}"
                pssplot.plot_trace_channel("Dynamic Analysis Plot Book", 2, 1, 1,
                    f"{voltage_channel2} : {output_file}", 1, output_file,
                    voltage_channel2, [-1,-1,-1,16777215])
                self.log_message(f"Plotted voltage magnitude for bus {fault_bus+1}")
            except Exception as e:
                self.log_message(f"Could not plot voltage magnitude for bus {fault_bus+1}: {str(e)}")
            
            # Additional plots - frequency, machine variables if available
            try:
                self.log_message("Creating additional plots for machine variables...")
                
                # Page 3: Generator Speed/Frequency (if available)
                pssplot.plot_page("Dynamic Analysis Plot Book", 3, "Machine Variables", [2,0,2,2], [1.,1.])
                pssplot.plot_plot("Dynamic Analysis Plot Book", 3, 1, "Generator Speed", [0,0,1,1,1,2,0], [0.0,0.0,100.,100.])
                
                # Try to plot generator speed if channels exist
                speed_channels = ["SPEED", "FREQ", "PELEC", "QELEC"]
                plot_idx = 0
                for channel_type in speed_channels:
                    try:
                        # Look for machine variables - these channel numbers may vary
                        for ch_num in range(50, 70):  # Common range for machine variables
                            try:
                                channel_name = f"{ch_num} - {channel_type}"
                                pssplot.plot_trace_channel("Dynamic Analysis Plot Book", 3, 1, plot_idx,
                                    f"{channel_name} : {output_file}", 1, output_file,
                                    channel_name, [-1,-1,-1,16777215])
                                plot_idx += 1
                                if plot_idx >= 4:  # Limit to 4 traces per plot
                                    break
                            except:
                                continue
                        if plot_idx >= 4:
                            break
                    except:
                        continue
                        
                if plot_idx > 0:
                    self.log_message(f"Added {plot_idx} machine variable traces")
                else:
                    self.log_message("No machine variable channels found for plotting")
                    
            except Exception as e:
                self.log_message(f"Could not create machine variable plots: {str(e)}")
            
            self.log_message("Dynamic plotting completed successfully!")
            self.log_message("Plot book 'Dynamic Analysis Plot Book' created with angle and voltage traces")
            
        except Exception as e:
            self.log_message(f"Error creating dynamic plots: {str(e)}")
            self.log_message("Note: Plotting is optional - simulation results are still saved")
    
    # IDV Generator Methods
    def load_impedance_file(self):
        """Load impedance CSV file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Line Impedance Estimator CSV file",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                self.impedance_df = pd.read_csv(file_path)
                print("Impedance CSV columns found:", self.impedance_df.columns.tolist())
                
                # Validate required columns
                if 'kV' not in self.impedance_df.columns:
                    messagebox.showerror("Error", "The impedance CSV must include a 'kV' column.")
                    self.impedance_df = None
                    return
                
                # Update kV dropdown
                kv_values = sorted(self.impedance_df['kV'].unique().astype(str))
                self.idv_kv_combo['values'] = kv_values
                
                # Clear dependent dropdowns
                self.idv_conductor_combo['values'] = []
                self.idv_conductor.set('')
                if 'Geometry' in self.impedance_df.columns:
                    self.idv_geometry_combo['values'] = []
                    self.idv_geometry.set('')
                
                # Update button text
                self.idv_impedance_button.config(text="✓ Impedance File Loaded")
                messagebox.showinfo("Success", f"Impedance file loaded successfully!\nFile: {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading impedance file: {str(e)}")
            self.impedance_df = None
    
    def load_ratings_file(self):
        """Load ratings CSV file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Ratings CSV file",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                self.ratings_df = pd.read_csv(file_path)
                print("Ratings CSV columns found:", self.ratings_df.columns.tolist())
                
                # Update button text
                self.idv_ratings_button.config(text="✓ Ratings File Loaded")
                messagebox.showinfo("Success", f"Ratings file loaded successfully!\nFile: {file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading ratings file: {str(e)}")
            self.ratings_df = None
    
    def update_idv_conductors(self, *args):
        """Update conductor dropdowns based on selected parameters"""
        if self.impedance_df is None:
            print("Debug: Impedance file not loaded yet")
            return
            
        kV = self.idv_kv.get()
        Conductors_Per_Phase = self.idv_conductors_per_phase.get()
        geom = self.idv_geometry.get() if 'Geometry' in self.impedance_df.columns else ''
        
        print(f"Debug: kV = '{kV}', Conductors_Per_Phase = '{Conductors_Per_Phase}', Geometry = '{geom}'")
        
        # If no kV/CPP yet selected, clear dependent menus
        if not kV or not Conductors_Per_Phase:
            if 'Geometry' in self.impedance_df.columns:
                self.idv_geometry_combo['values'] = []
                self.idv_geometry.set('')
            self.idv_conductor_combo['values'] = []
            self.idv_conductor.set('')
            return
            
        try:
            # Convert to int for filtering
            kV_int = int(kV)
            conductors_per_phase_int = int(Conductors_Per_Phase)

            # First level filter: by kV and conductors per phase
            base = self.impedance_df[(self.impedance_df['kV'] == kV_int) & 
                                   (self.impedance_df['Conductors Per Phase'] == conductors_per_phase_int)]
            print(f"Debug: Base filtered rows count: {len(base)}")

            # Populate Geometry options if column exists
            if 'Geometry' in self.impedance_df.columns:
                geoms = sorted(base['Geometry'].dropna().astype(str).unique()) if not base.empty else []
                print(f"Debug: Available geometries: {geoms}")
                self.idv_geometry_combo['values'] = geoms
                # If current geometry not valid, pick first or clear
                if geom not in geoms:
                    if geoms:
                        self.idv_geometry.set(geoms[0])
                        geom = geoms[0]
                        print(f"Debug: Auto-set geometry to: {geom}")
                    else:
                        self.idv_geometry.set('')
                        geom = ''

            # Second level filter: by geometry if present
            if 'Geometry' in self.impedance_df.columns and geom:
                filtered = base[base['Geometry'].astype(str) == geom]
            else:
                filtered = base

            print(f"Debug: Filtered rows count (post-geometry): {len(filtered)}")
            
            # Populate conductor types from filtered set
            if 'Type of Conductor' in filtered.columns:
                types = sorted(filtered['Type of Conductor'].dropna().astype(str).unique())
            else:
                types = []
            print(f"Debug: Available conductor types: {types}")
            
            self.idv_conductor_combo['values'] = types
            if types:
                if self.idv_conductor.get() not in types:
                    self.idv_conductor.set(types[0])
                    print(f"Debug: Set conductor to: {types[0]}")
            else:
                self.idv_conductor.set('')
                self.idv_conductor_combo['values'] = []
                print("Debug: No conductor types found, clearing menu")
                
        except Exception as e:
            print(f"Error in update_idv_conductors: {e}")
            self.idv_conductor_combo['values'] = []
    
    def calculate_mva(self, amps, kV):
        """Calculate MVA from amps and kV"""
        return round((1.732 * kV * amps) / 1000, 2)
    
    def map_conductor_name(self, impedance_conductor, conductors_per_phase):
        """Map conductor name from impedance CSV format to ratings CSV format"""
        # Extract the main conductor info from impedance format
        parts = impedance_conductor.strip().split()
        if len(parts) >= 4:
            # Get conductor size and map to ratings format
            size_raw = parts[0].replace('.0', '')
            
            # Map conductor sizes from impedance CSV to ratings CSV format
            size_mappings = {
                '336.4': '336', '336': '336',
                '477.0': '477', '477': '477',
                '795.0': '795', '795': '795',
                '959.6': '959', '959': '959',
                '1433.6': '1433', '1433': '1433',
                '1590.0': '1590', '1590': '1590',
                '1926.9': '1926', '1926': '1926'
            }
            
            size = size_mappings.get(size_raw, size_raw)
            
            # Get conductor type (ACSR, ACSS/AW, ACSS/TW, etc.)
            conductor_type = ' '.join(parts[2:-1])  # Everything between stranding and name
            
            # Get conductor name (last part)
            name = parts[-1].upper()  # Use uppercase to match ratings format
            
            # Handle special cases for conductor types
            if 'ACSS/AW' in conductor_type:
                conductor_type = 'ACSS/AW'
            elif 'ACSS/ TW' in conductor_type or 'ACSS/TW' in conductor_type:
                conductor_type = 'ACSS/TW'
            elif 'ACSR' in conductor_type:
                conductor_type = 'ACSR'
            
            # Handle special name mappings and ensure proper capitalization
            name_mappings = {
                'LINNET': 'Linnet', 'DRAKE': 'Drake', 'HAWK': 'Hawk',
                'SUWANNEE': 'Suwannee', 'MERRIMACK': 'Merimack',
                'CUMBERLAND': 'Cumberland', 'LAPWING': 'Falcon'
            }
            
            mapped_name = name_mappings.get(name, name.title())
            
            # Build the conductor string for single conductor
            conductor_str = f"{size} {conductor_type} {mapped_name}"
            
            # Add bundled indicator for 2 conductors per phase
            if conductors_per_phase == 2:
                conductor_str = f"{size}x2 {conductor_type} Bundled {mapped_name}"
            
            print(f"Debug: Mapping '{impedance_conductor}' -> '{conductor_str}'")
            return conductor_str
        
        return impedance_conductor  # Return original if parsing fails
    
    def generate_idv(self):
        """Generate IDV string based on input parameters"""
        try:
            # Check if impedance file is loaded
            if self.impedance_df is None:
                messagebox.showerror("Error", "Please load the Line Impedance Estimator CSV file first")
                return
                
            # Validate inputs
            if not all([self.idv_from_bus.get(), self.idv_to_bus.get(), 
                       self.idv_amps.get(), self.idv_miles.get()]):
                messagebox.showerror("Error", "Please fill in all required fields (From Bus, To Bus, Min Rating, Length)")
                return
                
            if not all([self.idv_kv.get(), self.idv_conductor.get()]):
                messagebox.showerror("Error", "Please select voltage level and conductor type")
                return
            
            # If Geometry column exists, require a selection
            if 'Geometry' in self.impedance_df.columns and not self.idv_geometry.get():
                messagebox.showerror("Error", "Please select a Geometry type")
                return
            
            from_bus = int(self.idv_from_bus.get())
            to_bus = int(self.idv_to_bus.get())
            amps = float(self.idv_amps.get())
            miles = float(self.idv_miles.get())
            kV = int(self.idv_kv.get())
            Conductors_Per_Phase = int(self.idv_conductors_per_phase.get())
            ctype = self.idv_conductor.get()
            geom = self.idv_geometry.get() if 'Geometry' in self.impedance_df.columns else None

            # Filter the data and check if any matches exist
            base_filter = (
                (self.impedance_df['kV'] == kV) & 
                (self.impedance_df['Conductors Per Phase'] == Conductors_Per_Phase) & 
                (self.impedance_df['Type of Conductor'] == ctype)
            )
            if 'Geometry' in self.impedance_df.columns and geom:
                base_filter = base_filter & (self.impedance_df['Geometry'].astype(str) == str(geom))
            
            filtered_data = self.impedance_df[base_filter]
            
            if filtered_data.empty:
                msg = (
                    f"No matching data found for:\n"
                    f"kV: {kV}\n"
                    f"Conductors Per Phase: {Conductors_Per_Phase}\n"
                    f"Type of Conductor: {ctype}"
                )
                if 'Geometry' in self.impedance_df.columns:
                    msg += f"\nGeometry: {geom}"
                messagebox.showerror("Error", msg)
                return
                
            row = filtered_data.iloc[0]
            # Allow alternate column names
            try:
                R_per_mile, X_per_mile, B_per_mile = row['R'], row['X'], row['B']
            except KeyError:
                # Try lowercase or other variants
                R_per_mile = row.get('r', None)
                X_per_mile = row.get('x', None)
                B_per_mile = row.get('b', None)
                if None in (R_per_mile, X_per_mile, B_per_mile):
                    raise KeyError("Columns R, X, B not found in the impedance CSV")
            
            # Calculate total impedance by multiplying per-mile values by length
            R = R_per_mile * miles
            X = X_per_mile * miles
            B = B_per_mile * miles
            
            # Calculate MVA from amps and kV
            mva = self.calculate_mva(amps, kV)
            
            # Initialize ratings array with 12 slots (as per PSSE API)
            ratings = [0.0] * 12
            
            # Set first 3 ratings to calculated MVA
            ratings[0] = mva  # RATE1
            ratings[1] = mva  # RATE2  
            ratings[2] = mva  # RATE3
            
            # Get rating from Ratings.csv for slot 4
            if self.ratings_df is not None:
                try:
                    # Map the conductor name to ratings CSV format
                    mapped_conductor = self.map_conductor_name(ctype, Conductors_Per_Phase)
                    print(f"Debug: Mapped conductor from '{ctype}' to '{mapped_conductor}'")
                    
                    # Look for the mapped conductor in the ratings CSV
                    rating_filtered = self.ratings_df[self.ratings_df['Conductor Size'].str.contains(
                        f"^{mapped_conductor}$", case=False, na=False, regex=True)]
                    
                    if rating_filtered.empty:
                        # Try without "Bundled" keyword
                        simple_conductor = mapped_conductor.replace(" Bundled", "")
                        rating_filtered = self.ratings_df[self.ratings_df['Conductor Size'].str.contains(
                            f"^{simple_conductor}$", case=False, na=False, regex=True)]
                        print(f"Debug: Trying without 'Bundled': '{simple_conductor}'")
                    
                    if rating_filtered.empty:
                        # Try partial match with just size and type
                        conductor_parts = mapped_conductor.split()
                        if len(conductor_parts) >= 3:
                            partial_search = f"{conductor_parts[0]}.*{conductor_parts[1]}.*{conductor_parts[-1]}"
                            rating_filtered = self.ratings_df[self.ratings_df['Conductor Size'].str.contains(
                                partial_search, case=False, na=False, regex=True)]
                            print(f"Debug: Trying partial match: '{partial_search}'")
                    
                    if not rating_filtered.empty:
                        print(f"Debug: Found match: {rating_filtered['Conductor Size'].iloc[0]}")
                        
                        # Look for rating columns
                        rating_columns = ['Rating', 'Ampacity', 'Current Rating', 'Amps', 'Normal Rating', 'Emergency Rating']
                        conductor_rating = 0.0
                        
                        for col in rating_columns:
                            if col in rating_filtered.columns:
                                conductor_rating = rating_filtered.iloc[0][col]
                                print(f"Debug: Found rating in column '{col}': {conductor_rating}")
                                break
                        
                        if conductor_rating == 0.0:
                            # If no rating column found, try to get any numeric column
                            numeric_cols = rating_filtered.select_dtypes(include=['number']).columns
                            if len(numeric_cols) > 0:
                                conductor_rating = rating_filtered.iloc[0][numeric_cols[0]]
                                print(f"Debug: Using first numeric column '{numeric_cols[0]}': {conductor_rating}")
                        
                        ratings[3] = conductor_rating  # RATE4
                        print(f"Debug: Set RATE4 to conductor rating: {conductor_rating}")
                    else:
                        print(f"Debug: No matching conductor rating found for '{mapped_conductor}'")
                        
                except Exception as e:
                    print(f"Debug: Error getting conductor rating: {e}, RATE4 set to 0.0")
            
            # Set 5th rating to 9999
            ratings[4] = 9999.0  # RATE5
            
            # Ratings 6-12 remain 0.0 (already initialized)
            print(f"Debug: Final ratings array: {ratings}")

            idv = f"BAT_BRANCH_CHNG_3,{from_bus},{to_bus},'1',,,,,,,{R},{X},{B}," + \
                  "," * 17 + ",".join(map(str, ratings)) + "," * 0 + ";"

            # Display results
            self.idv_output_text.delete("1.0", tk.END)
            self.idv_output_text.insert(tk.END, idv + f"\n\nConverted MVA: {mva}")
            self.idv_output_text.insert(tk.END, f"\nLength: {miles} miles")
            self.idv_output_text.insert(tk.END, f"\nImpedance per mile: R={R_per_mile}, X={X_per_mile}, B={B_per_mile}")
            self.idv_output_text.insert(tk.END, f"\nTotal impedance: R={R}, X={X}, B={B}")
            if geom:
                self.idv_output_text.insert(tk.END, f"\nGeometry: {geom}")
            self.idv_output_text.insert(tk.END, f"\nRatings: RATE1-3={mva} MVA, RATE4={ratings[3]}, RATE5=9999")

        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numeric values for bus numbers and amperage")
        except KeyError as e:
            messagebox.showerror("Error", f"Missing column in CSV file: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = MyLabGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()