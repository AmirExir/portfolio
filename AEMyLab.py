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

# PSSE will be initialized only when needed for PSSE-specific tasks
PSSE_AVAILABLE = False

from time import perf_counter as tic

def initialize_psse():
    """Initialize PSSE modules only when needed for PSSE-specific tasks"""
    global PSSE_AVAILABLE, psspy, redirect, arrbox, excelpy, pssplot, dyntools, psse35
    
    if PSSE_AVAILABLE:
        return True  # Already initialized
    
    # PSSE imports for PSSE 35.6 with Python 39
    sys_paths = [r'C:\Program Files\PTI\PSSE35\35.6\PSSPY39']
    env_paths = [r'C:\Program Files\PTI\PSSE35\35.6\PSSBIN', 
                 r'C:\Program Files\PTI\PSSE35\35.6\PSSLIB']
    
    # Add PSSE paths only when needed
    for path in sys_paths:
        if path not in sys.path:
            sys.path.append(path)
    for path in env_paths:
        if path not in os.environ['PATH']:
            os.environ['PATH'] = os.environ['PATH'] + ';' + path

    try:
        import psse35 as _psse35
        import psspy as _psspy
        import redirect as _redirect
        import arrbox as _arrbox
        import excelpy as _excelpy
        import pssplot as _pssplot
        import dyntools as _dyntools
        
        # Make modules globally available
        globals()['psse35'] = _psse35
        globals()['psspy'] = _psspy
        globals()['redirect'] = _redirect
        globals()['arrbox'] = _arrbox
        globals()['excelpy'] = _excelpy
        globals()['pssplot'] = _pssplot
        globals()['dyntools'] = _dyntools
        
        PSSE_AVAILABLE = True
        print("PSSE modules initialized successfully")
        return True
    except ImportError as e:
        print(f"PSSE modules not available: {e}")
        return False

class AEMyLabGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AEMyLab - Power System Analysis Suite")
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
        # Dark mode state
        self.dark_mode = tk.BooleanVar(value=False)
        
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
        self.idv_shield = tk.StringVar()
        self.idv_conductor = tk.StringVar()
        self.idv_environmental_index = tk.StringVar()
        self.idv_construction_type = tk.StringVar(value="Existing")
        self.idv_filename = tk.StringVar()  # Export filename
        
        # IDV data storage
        self.impedance_df = None
        self.ratings_df = None
        self.current_idv_line = None  # Store the generated IDV line for export
        
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
            # Ensure background is behind all other widgets
            self.bg_label.lower()
            
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
            # Ensure background is behind all other widgets
            self.bg_label.lower()
            
        except Exception as e:
            print(f"Could not load background image {image_path}: {e}")
            self.create_gradient_background()
    
    def create_menu(self):
        """Create menu bar with background options"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # View menu for background and theme options
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="🌙 Dark Mode", variable=self.dark_mode, command=self.toggle_dark_mode)
        view_menu.add_separator()
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
    
    def toggle_dark_mode(self):
        """Toggle between light and dark mode"""
        self.apply_theme()
    
    def apply_theme(self):
        """Apply the current theme (light or dark mode)"""
        style = ttk.Style()
        
        # Check if dark_mode exists and get its value, default to False
        is_dark_mode = getattr(self, 'dark_mode', tk.BooleanVar(value=False)).get()
        
        if is_dark_mode:
            # Dark mode colors
            bg_color = "#2b2b2b"
            fg_color = "#ffffff"
            select_bg = "#404040"
            select_fg = "#ffffff"
            entry_bg = "#404040"
            entry_fg = "#ffffff"
            
            # Configure dark theme
            style.theme_use('clam')
            
            # Configure main window
            self.root.configure(bg=bg_color)
            
            # Configure ttk styles for dark mode
            style.configure('TLabel', background=bg_color, foreground=fg_color)
            style.configure('TFrame', background=bg_color)
            style.configure('TLabelFrame', background=bg_color, foreground=fg_color)
            style.configure('TLabelFrame.Label', background=bg_color, foreground=fg_color)
            style.configure('TNotebook', background=bg_color, foreground=fg_color)
            style.configure('TNotebook.Tab', background=select_bg, foreground=fg_color, padding=[20, 8])
            style.map('TNotebook.Tab', background=[('selected', '#505050'), ('active', '#454545')])
            
            # Entry and Combobox styling
            style.configure('TEntry', fieldbackground=entry_bg, foreground=entry_fg, bordercolor='#555555')
            style.configure('TCombobox', fieldbackground=entry_bg, foreground=entry_fg, bordercolor='#555555')
            style.map('TCombobox', fieldbackground=[('readonly', entry_bg)])
            
            # Scrolled text and tree widgets
            style.configure('Treeview', background=entry_bg, foreground=fg_color, fieldbackground=entry_bg)
            style.configure('Treeview.Heading', background=select_bg, foreground=fg_color)
            
            # Progress bar
            style.configure('TProgressbar', background='#4CAF50', troughcolor=select_bg)
            
            # Checkbutton
            style.configure('TCheckbutton', background=bg_color, foreground=fg_color)
            
            # Update background gradient for dark mode
            if hasattr(self, 'bg_label'):
                self.bg_label.destroy()
            self.create_dark_gradient_background()
            
        else:
            # Light mode colors
            bg_color = "#f0f0f0"
            fg_color = "#000000"
            
            # Configure light theme
            style.theme_use('clam')
            
            # Configure main window
            self.root.configure(bg=bg_color)
            
            # Reset to default light theme
            style.configure('TLabel', background=bg_color, foreground=fg_color)
            style.configure('TFrame', background=bg_color)
            style.configure('TLabelFrame', background=bg_color, foreground=fg_color)
            style.configure('TLabelFrame.Label', background=bg_color, foreground=fg_color)
            style.configure('TNotebook', background=bg_color, foreground=fg_color)
            style.configure('TNotebook.Tab', background='#e1e1e1', foreground=fg_color, padding=[20, 8])
            style.map('TNotebook.Tab', background=[('selected', '#ffffff'), ('active', '#f5f5f5')])
            
            # Entry and Combobox styling
            style.configure('TEntry', fieldbackground='#ffffff', foreground=fg_color, bordercolor='#cccccc')
            style.configure('TCombobox', fieldbackground='#ffffff', foreground=fg_color, bordercolor='#cccccc')
            style.map('TCombobox', fieldbackground=[('readonly', '#ffffff')])
            
            # Scrolled text and tree widgets
            style.configure('Treeview', background='#ffffff', foreground=fg_color, fieldbackground='#ffffff')
            style.configure('Treeview.Heading', background='#e1e1e1', foreground=fg_color)
            
            # Progress bar
            style.configure('TProgressbar', background='#4CAF50', troughcolor='#e0e0e0')
            
            # Checkbutton
            style.configure('TCheckbutton', background=bg_color, foreground=fg_color)
            
            # Update background gradient for light mode
            if hasattr(self, 'bg_label'):
                self.bg_label.destroy()
            self.create_gradient_background()
        
        # Update scrolled text widgets manually (they don't inherit ttk styles)
        self.update_scrolled_text_widgets()
    
    def create_dark_gradient_background(self):
        """Create a dark gradient background"""
        if not PIL_AVAILABLE:
            self.root.configure(bg='#2b2b2b')
            return
            
        try:
            # Create a dark gradient image
            width, height = 1400, 900
            image = Image.new('RGB', (width, height), '#2b2b2b')
            
            # Create a simple vertical gradient from dark gray to darker gray
            for y in range(height):
                ratio = y / height
                r = int(43 + (35 - 43) * ratio)   # 43 to 35 (darker gradient)
                g = int(43 + (35 - 43) * ratio)   # 43 to 35
                b = int(43 + (35 - 43) * ratio)   # 43 to 35
                
                # Draw horizontal line with this color
                for x in range(width):
                    image.putpixel((x, y), (r, g, b))
            
            # Convert to PhotoImage and set as background
            self.bg_image = ImageTk.PhotoImage(image)
            
            # Create a label to hold the background image
            self.bg_label = tk.Label(self.root, image=self.bg_image)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            # Ensure background is behind all other widgets
            self.bg_label.lower()
            
        except Exception as e:
            print(f"Could not create dark gradient background: {e}")
            self.root.configure(bg='#2b2b2b')
    
    def update_scrolled_text_widgets(self):
        """Update scrolled text widgets with current theme colors"""
        try:
            is_dark_mode = getattr(self, 'dark_mode', tk.BooleanVar(value=False)).get()
            if is_dark_mode:
                # Dark mode colors for text widgets
                bg_color = "#3c3c3c"
                fg_color = "#ffffff"
                select_bg = "#555555"
            else:
                # Light mode colors for text widgets
                bg_color = "#ffffff"
                fg_color = "#000000"
                select_bg = "#0078d4"
            
            # Update all scrolled text widgets that exist
            text_widget_names = [
                'accc_log_text',
                'dynamic_log_text', 
                'p3p6_log_text',
                'idv_output_text',
                'dynamic_results_text'
            ]
            
            for widget_name in text_widget_names:
                if hasattr(self, widget_name):
                    widget = getattr(self, widget_name)
                    try:
                        if widget.winfo_exists():
                            widget.configure(
                                bg=bg_color,
                                fg=fg_color,
                                selectbackground=select_bg,
                                selectforeground=fg_color,
                                insertbackground=fg_color
                            )
                    except tk.TclError:
                        pass  # Widget doesn't exist yet
        except Exception as e:
            print(f"Error updating text widgets: {e}")
        
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
        
        # Contingency Generator tab
        self.p3p6_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.p3p6_frame, text="Contingency Generator")
        
        # IDV Generator tab
        self.idv_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.idv_frame, text="IDV Generator")
        
        # TPIT tab
        self.tpit_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.tpit_frame, text="TPIT")
        
        # Results tab
        self.results_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.results_frame, text="Results")
        
        self.create_accc_tab()
        self.create_dynamic_tab()
        self.create_p3p6_tab()
        self.create_idv_tab()
        self.create_tpit_tab()
        self.create_results_tab()
        
        # Apply initial light theme
        self.apply_theme()
        
    def create_accc_tab(self):
        """Create the ACCC analysis tab"""
        # File selection frame
        accc_file_frame = ttk.LabelFrame(self.accc_frame, text="ACCC Input Files", padding=10)
        accc_file_frame.pack(fill="x", padx=10, pady=5)
        
        # SAV file
        ttk.Label(accc_file_frame, text="Case File (.sav):").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(accc_file_frame, textvariable=self.sav_file, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(accc_file_frame, text="📁 Browse", command=lambda: self.browse_file(self.sav_file, "PSSE Case Files", "*.sav"), style="Browse.TButton").grid(row=0, column=2, pady=2)
        
        # SUB file
        ttk.Label(accc_file_frame, text="Subsystem File (.sub):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(accc_file_frame, textvariable=self.sub_file, width=60).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(accc_file_frame, text="📁 Browse", command=lambda: self.browse_file(self.sub_file, "Subsystem Files", "*.sub"), style="Browse.TButton").grid(row=1, column=2, pady=2)
        
        # MON file
        ttk.Label(accc_file_frame, text="Monitor File (.mon):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(accc_file_frame, textvariable=self.mon_file, width=60).grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(accc_file_frame, text="📁 Browse", command=lambda: self.browse_file(self.mon_file, "Monitor Files", "*.mon"), style="Browse.TButton").grid(row=2, column=2, pady=2)
        
        # CON file
        ttk.Label(accc_file_frame, text="Contingency File (.con):").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(accc_file_frame, textvariable=self.con_file, width=60).grid(row=3, column=1, padx=5, pady=2)
        ttk.Button(accc_file_frame, text="📁 Browse", command=lambda: self.browse_file(self.con_file, "Contingency Files", "*.con"), style="Browse.TButton").grid(row=3, column=2, pady=2)
        
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
        
        self.accc_run_button = ttk.Button(accc_control_frame, text="🚀 Run ACCC Analysis", command=self.run_accc_analysis, style="Run.TButton")
        self.accc_run_button.pack(side="left", padx=5)
        
        ttk.Button(accc_control_frame, text="📊 Export ACCC Results", command=self.export_accc_results, style="Export.TButton").pack(side="left", padx=5)
        ttk.Button(accc_control_frame, text="📋 PSSE Violations Report", command=self.export_psse_violations_report, style="Export.TButton").pack(side="left", padx=5)
        
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
        ttk.Button(dyn_file_frame, text="📁 Browse", command=lambda: self.browse_file(self.sav_file, "PSSE Case Files", "*.sav"), style="Browse.TButton").grid(row=0, column=2, pady=2)
        
        # DYR file
        ttk.Label(dyn_file_frame, text="Dynamics File (.dyr):").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(dyn_file_frame, textvariable=self.dyr_file, width=60).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(dyn_file_frame, text="📁 Browse", command=lambda: self.browse_file(self.dyr_file, "Dynamics Files", "*.dyr"), style="Browse.TButton").grid(row=1, column=2, pady=2)
        
        # CNV file (Converted case file)
        ttk.Label(dyn_file_frame, text="Converted Case File (.sav):").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(dyn_file_frame, textvariable=self.cnv_file, width=60).grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(dyn_file_frame, text="📁 Browse", command=lambda: self.browse_file(self.cnv_file, "Converted Case Files", "*.sav"), style="Browse.TButton").grid(row=2, column=2, pady=2)
        
        # SNP file (Snapshot file)
        ttk.Label(dyn_file_frame, text="Snapshot File (.snp):").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(dyn_file_frame, textvariable=self.snp_file, width=60).grid(row=3, column=1, padx=5, pady=2)
        ttk.Button(dyn_file_frame, text="📁 Browse", command=lambda: self.browse_file(self.snp_file, "Snapshot Files", "*.snp"), style="Browse.TButton").grid(row=3, column=2, pady=2)
        
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
        ttk.Button(right_dyn_param, text="📁 Browse", command=self.browse_output_file, style="Browse.TButton").grid(row=2, column=2, pady=2)
        
        # Plotting option
        ttk.Checkbutton(right_dyn_param, text="Enable Auto-Plotting", variable=self.enable_plotting).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)
        
        # Dynamic Control frame
        dyn_control_frame = ttk.LabelFrame(self.dynamic_frame, text="Dynamic Control", padding=10)
        dyn_control_frame.pack(fill="x", padx=10, pady=5)
        
        self.dynamic_run_button = ttk.Button(dyn_control_frame, text="⚡ Run Dynamic Analysis", command=self.run_dynamic_analysis, style="Run.TButton")
        self.dynamic_run_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(dyn_control_frame, text="🛑 Stop Analysis", command=self.stop_analysis, state="disabled", style="Stop.TButton")
        self.stop_button.pack(side="left", padx=5)
        
        ttk.Button(dyn_control_frame, text="👁️ View Dynamic Results", command=self.view_dynamic_results, style="Export.TButton").pack(side="left", padx=5)
        
        # Dynamic Progress frame
        dyn_progress_frame = ttk.LabelFrame(self.dynamic_frame, text="Dynamic Progress", padding=10)
        dyn_progress_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.progress_bar = ttk.Progressbar(dyn_progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill="x", pady=5)
        
        self.dynamic_log_text = scrolledtext.ScrolledText(dyn_progress_frame, height=8, wrap=tk.WORD)
        self.dynamic_log_text.pack(fill="both", expand=True)
        
    def create_p3p6_tab(self):
        """Create the Contingency Generator tab"""
        # Main container
        main_container = ttk.Frame(self.p3p6_frame)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title and description
        title_label = ttk.Label(main_container, text="Contingency Generator", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Information frame
        info_frame = ttk.LabelFrame(main_container, text="Information", padding=10)
        info_frame.pack(fill="x", pady=(0, 10))
        
        info_text = """This tool creates single and double contingencies from branch data, generator data, and auto data.
        
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
- Creates single contingencies for each element
- Creates double contingencies P3 and P6 for all combinations
- Output file is in PSSE contingency format (.con)"""
        
        info_label = tk.Label(info_frame, text=info_text, justify="left", wraplength=800, font=('Arial', 9))
        info_label.pack(anchor="w")
        
        # Input data frame - split into three columns
        input_frame = ttk.LabelFrame(main_container, text="Input Data (Copy & Paste)", padding=10)
        input_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Create three columns for different data types
        columns_frame = ttk.Frame(input_frame)
        columns_frame.pack(fill="both", expand=True)
        
        # Branch Data Column
        branch_frame = ttk.LabelFrame(columns_frame, text="Branch Data", padding=5)
        branch_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        ttk.Label(branch_frame, text="Paste branch data here:", font=('Arial', 9, 'bold')).pack(anchor="w")
        self.branch_data_text = scrolledtext.ScrolledText(branch_frame, height=12, width=30, wrap=tk.WORD, font=('Courier', 8))
        self.branch_data_text.pack(fill="both", expand=True, pady=(5, 0))
        
        # Generator Data Column  
        generator_frame = ttk.LabelFrame(columns_frame, text="Generator Data", padding=5)
        generator_frame.pack(side="left", fill="both", expand=True, padx=2.5)
        
        ttk.Label(generator_frame, text="Paste generator data here:", font=('Arial', 9, 'bold')).pack(anchor="w")
        self.generator_data_text = scrolledtext.ScrolledText(generator_frame, height=12, width=30, wrap=tk.WORD, font=('Courier', 8))
        self.generator_data_text.pack(fill="both", expand=True, pady=(5, 0))
        
        # Auto Data Column
        auto_frame = ttk.LabelFrame(columns_frame, text="Auto Data", padding=5)
        auto_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        ttk.Label(auto_frame, text="Paste auto data here:", font=('Arial', 9, 'bold')).pack(anchor="w")
        self.auto_data_text = scrolledtext.ScrolledText(auto_frame, height=12, width=30, wrap=tk.WORD, font=('Courier', 8))
        self.auto_data_text.pack(fill="both", expand=True, pady=(5, 0))
        
        # Output file selection
        output_frame = ttk.LabelFrame(main_container, text="Output Settings", padding=10)
        output_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(output_frame, text="Output File (.con):").pack(side="left", padx=(0, 10))
        output_entry = ttk.Entry(output_frame, textvariable=self.p3p6_output_file, width=50)
        output_entry.pack(side="left", padx=(0, 5), fill="x", expand=True)
        ttk.Button(output_frame, text="💾 Browse", 
                  command=lambda: self.browse_save_file(self.p3p6_output_file, "Contingency Files", "*.con"), 
                  style="Browse.TButton").pack(side="right")
        
        # Control buttons frame
        control_frame = ttk.Frame(main_container)
        control_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(control_frame, text="🧹 Clear All", 
                  command=self.clear_contingency_data,
                  style="Export.TButton").pack(side="left", padx=5)
        
        ttk.Button(control_frame, text="⚙️ Generate Contingencies", 
                  command=self.generate_contingencies_from_data,
                  style="Special.TButton").pack(side="right", padx=5)
        
        # Progress and log frame
        progress_frame = ttk.LabelFrame(main_container, text="Progress & Log", padding=10)
        progress_frame.pack(fill="both", expand=True)
        
        self.p3p6_progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.p3p6_progress_bar.pack(fill="x", pady=(0, 5))
        
        self.p3p6_log_text = scrolledtext.ScrolledText(progress_frame, height=8, wrap=tk.WORD, font=('Courier', 9))
        self.p3p6_log_text.pack(fill="both", expand=True)
        
    def create_idv_tab(self):
        """Create the IDV Generator tab"""
        # File selection frame
        idv_file_frame = ttk.LabelFrame(self.idv_frame, text="CSV Files", padding=10)
        idv_file_frame.pack(fill="x", padx=10, pady=5)
        
        # Create custom styles for the CSV buttons
        style = ttk.Style()
        
        # Use a theme that supports better color customization
        try:
            style.theme_use('clam')  # Better color support than default
        except:
            pass
        
        # Define colorful styles for the CSV upload buttons with bright text
        style.configure("Impedance.TButton", 
                       foreground="black",
                       background="#32CD32",
                       font=('TkDefaultFont', 9, 'bold'),
                       focuscolor="none",
                       relief="raised",
                       borderwidth=1)
        style.map("Impedance.TButton",
                 foreground=[('active', 'black'), ('!active', 'black')],
                 background=[('active', '#228B22'), ('!active', '#32CD32')])
        
        style.configure("Ratings.TButton", 
                       foreground="white",
                       background="#1E90FF", 
                       font=('TkDefaultFont', 9, 'bold'),
                       focuscolor="none",
                       relief="raised",
                       borderwidth=1)
        style.map("Ratings.TButton",
                 foreground=[('active', 'white'), ('!active', 'white')],
                 background=[('active', '#4169E1'), ('!active', '#1E90FF')])
        
        # Style for Generate IDV button
        style.configure("Generate.TButton", 
                       foreground="white",
                       background="#FF6347", 
                       font=('TkDefaultFont', 10, 'bold'),
                       focuscolor="none",
                       relief="raised",
                       borderwidth=1)
        style.map("Generate.TButton",
                 foreground=[('active', 'white'), ('!active', 'white')],
                 background=[('active', '#FF4500'), ('!active', '#FF6347')])
        
        # Style for Browse/File buttons (Orange theme)
        style.configure("Browse.TButton", 
                       foreground="white",
                       background="#FF8C00", 
                       font=('TkDefaultFont', 9, 'bold'),
                       focuscolor="none",
                       relief="raised",
                       borderwidth=1)
        style.map("Browse.TButton",
                 foreground=[('active', 'white'), ('!active', 'white')],
                 background=[('active', '#FF7F00'), ('!active', '#FF8C00')])
        
        # Style for Run/Action buttons (Purple theme)
        style.configure("Run.TButton", 
                       foreground="white",
                       background="#9370DB", 
                       font=('TkDefaultFont', 10, 'bold'),
                       focuscolor="none",
                       relief="raised",
                       borderwidth=1)
        style.map("Run.TButton",
                 foreground=[('active', 'white'), ('!active', 'white')],
                 background=[('active', '#8A2BE2'), ('!active', '#9370DB')])
        
        # Style for Export/View buttons (Teal theme)
        style.configure("Export.TButton", 
                       foreground="white",
                       background="#20B2AA", 
                       font=('TkDefaultFont', 9, 'bold'),
                       focuscolor="none",
                       relief="raised",
                       borderwidth=1)
        style.map("Export.TButton",
                 foreground=[('active', 'white'), ('!active', 'white')],
                 background=[('active', '#008B8B'), ('!active', '#20B2AA')])
        
        # Style for Stop/Control buttons (Red theme)
        style.configure("Stop.TButton", 
                       foreground="white",
                       background="#DC143C", 
                       font=('TkDefaultFont', 9, 'bold'),
                       focuscolor="none",
                       relief="raised",
                       borderwidth=1)
        style.map("Stop.TButton",
                 foreground=[('active', 'white'), ('!active', 'white')],
                 background=[('active', '#B22222'), ('!active', '#DC143C')])
        
        # Style for Special buttons (Gold theme)
        style.configure("Special.TButton", 
                       foreground="black",
                       background="#FFD700", 
                       font=('TkDefaultFont', 9, 'bold'),
                       focuscolor="none",
                       relief="raised",
                       borderwidth=1)
        style.map("Special.TButton",
                 foreground=[('active', 'black'), ('!active', 'black')],
                 background=[('active', '#FFA500'), ('!active', '#FFD700')])
        
        # Style for TPIT Browse buttons (Vibrant Green theme)
        style.configure("TPITBrowse.TButton", 
                       foreground="white",
                       background="#32CD32", 
                       font=('TkDefaultFont', 9, 'bold'),
                       focuscolor="none",
                       relief="raised",
                       borderwidth=1)
        style.map("TPITBrowse.TButton",
                 foreground=[('active', 'white'), ('!active', 'white')],
                 background=[('active', '#228B22'), ('!active', '#32CD32')])
        
        # Style for TPIT Process button (Vibrant Blue theme)
        style.configure("TPITProcess.TButton", 
                       foreground="white",
                       background="#1E90FF", 
                       font=('TkDefaultFont', 10, 'bold'),
                       focuscolor="none",
                       relief="raised",
                       borderwidth=1)
        style.map("TPITProcess.TButton",
                 foreground=[('active', 'white'), ('!active', 'white')],
                 background=[('active', '#0080FF'), ('!active', '#1E90FF')])
        
        # Load impedance file button with green styling
        self.idv_impedance_button = ttk.Button(idv_file_frame, text="📊 Browse - Line Impedance Estimator CSV", 
                                             command=self.load_impedance_file, 
                                             style="Impedance.TButton")
        self.idv_impedance_button.pack(fill="x", pady=5)
        
        # Load ratings file button with blue styling
        self.idv_ratings_button = ttk.Button(idv_file_frame, text="⚡ Browse - Ratings CSV (Optional)", 
                                           command=self.load_ratings_file,
                                           style="Ratings.TButton")
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
        self.idv_kv_combo = ttk.Combobox(right_input, textvariable=self.idv_kv, width=25, state="readonly")
        self.idv_kv_combo.grid(row=0, column=1, padx=5, pady=2)
        self.idv_kv.trace('w', self.update_idv_conductors)
        
        ttk.Label(right_input, text="Conductors Per Phase:").grid(row=1, column=0, sticky="w", pady=2)
        self.idv_cpp_combo = ttk.Combobox(right_input, textvariable=self.idv_conductors_per_phase, 
                                        values=['1', '2'], width=25, state="readonly")
        self.idv_cpp_combo.grid(row=1, column=1, padx=5, pady=2)
        self.idv_conductors_per_phase.trace('w', self.update_idv_conductors)
        
        ttk.Label(right_input, text="Geometry:").grid(row=2, column=0, sticky="w", pady=2)
        self.idv_geometry_combo = ttk.Combobox(right_input, textvariable=self.idv_geometry, width=25, state="readonly")
        self.idv_geometry_combo.grid(row=2, column=1, padx=5, pady=2)
        self.idv_geometry.trace('w', self.update_idv_conductors)
        
        ttk.Label(right_input, text="Shield:").grid(row=3, column=0, sticky="w", pady=2)
        self.idv_shield_combo = ttk.Combobox(right_input, textvariable=self.idv_shield, width=25, state="readonly")
        self.idv_shield_combo.grid(row=3, column=1, padx=5, pady=2)
        self.idv_shield.trace('w', self.update_idv_conductors)
        
        ttk.Label(right_input, text="Environmental Index:").grid(row=4, column=0, sticky="w", pady=2)
        self.idv_env_index_combo = ttk.Combobox(right_input, textvariable=self.idv_environmental_index, 
                                              values=['1', '4'], width=25, state="readonly")
        self.idv_env_index_combo.grid(row=4, column=1, padx=5, pady=2)
        
        ttk.Label(right_input, text="Construction Type:").grid(row=5, column=0, sticky="w", pady=2)
        self.idv_construction_combo = ttk.Combobox(right_input, textvariable=self.idv_construction_type, 
                                                 values=['Existing', 'New Construction'], width=25, state="readonly")
        self.idv_construction_combo.grid(row=5, column=1, padx=5, pady=2)
        
        ttk.Label(right_input, text="Conductor Type:").grid(row=6, column=0, sticky="w", pady=2)
        self.idv_conductor_combo = ttk.Combobox(right_input, textvariable=self.idv_conductor, width=25, state="readonly")
        self.idv_conductor_combo.grid(row=6, column=1, padx=5, pady=2)
        
        # Generate button
        generate_frame = ttk.Frame(self.idv_frame)
        generate_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(generate_frame, text="🚀 Generate IDV", 
                  command=self.generate_idv,
                  style="Generate.TButton").pack(side="left", padx=5)
        
        ttk.Button(generate_frame, text="💾 Export IDV", 
                  command=self.export_idv,
                  style="Export.TButton").pack(side="left", padx=5)
        
        # Export filename frame
        filename_frame = ttk.LabelFrame(self.idv_frame, text="Export Settings", padding=10)
        filename_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(filename_frame, text="IDV Filename:").pack(side="left", padx=(0, 10))
        filename_entry = ttk.Entry(filename_frame, textvariable=self.idv_filename, width=30)
        filename_entry.pack(side="left", padx=(0, 5), fill="x", expand=True)
        ttk.Label(filename_frame, text=".idv").pack(side="left")
        
        # Output frame
        output_frame = ttk.LabelFrame(self.idv_frame, text="Generated IDV", padding=10)
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.idv_output_text = scrolledtext.ScrolledText(output_frame, height=8, wrap=tk.WORD, font=('Courier', 9))
        self.idv_output_text.pack(fill="both", expand=True)
        
    def create_tpit_tab(self):
        """Create the TPIT (Transmission Project Information Tool) tab"""
        # Main container with padding
        main_container = ttk.Frame(self.tpit_frame)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_container, text="TPIT - Transmission Project Information Tool", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_container, text="Input Files", padding=10)
        file_frame.pack(fill="x", pady=(0, 10))
        
        # Planning Data Export file
        planning_frame = ttk.Frame(file_frame)
        planning_frame.pack(fill="x", pady=5)
        ttk.Label(planning_frame, text="Planning Data Export:").pack(side="left", padx=(0, 10))
        self.planning_file_var = tk.StringVar()
        self.planning_entry = ttk.Entry(planning_frame, textvariable=self.planning_file_var, width=50)
        self.planning_entry.pack(side="left", padx=(0, 5), fill="x", expand=True)
        ttk.Button(planning_frame, text="📂 Browse", 
                  command=lambda: self.browse_file(self.planning_file_var, "Excel Files", "*.xlsx"),
                  style="TPITBrowse.TButton").pack(side="right")
        
        # Existing Report file
        existing_frame = ttk.Frame(file_frame)
        existing_frame.pack(fill="x", pady=5)
        ttk.Label(existing_frame, text="Existing Report:").pack(side="left", padx=(0, 10))
        self.existing_file_var = tk.StringVar()
        self.existing_entry = ttk.Entry(existing_frame, textvariable=self.existing_file_var, width=50)
        self.existing_entry.pack(side="left", padx=(0, 5), fill="x", expand=True)
        ttk.Button(existing_frame, text="📂 Browse", 
                  command=lambda: self.browse_file(self.existing_file_var, "Excel Files", "*.xlsx"),
                  style="TPITBrowse.TButton").pack(side="right")
        
        # MOD Report file
        mod_frame = ttk.Frame(file_frame)
        mod_frame.pack(fill="x", pady=5)
        ttk.Label(mod_frame, text="MOD Report:").pack(side="left", padx=(0, 10))
        self.mod_file_var = tk.StringVar()
        self.mod_entry = ttk.Entry(mod_frame, textvariable=self.mod_file_var, width=50)
        self.mod_entry.pack(side="left", padx=(0, 5), fill="x", expand=True)
        ttk.Button(mod_frame, text="📂 Browse", 
                  command=lambda: self.browse_file(self.mod_file_var, "Excel Files", "*.xlsx"),
                  style="TPITBrowse.TButton").pack(side="right")
        
        # Project Completion Report file
        completion_frame = ttk.Frame(file_frame)
        completion_frame.pack(fill="x", pady=5)
        ttk.Label(completion_frame, text="Project Completion Report:").pack(side="left", padx=(0, 10))
        self.completion_file_var = tk.StringVar()
        self.completion_entry = ttk.Entry(completion_frame, textvariable=self.completion_file_var, width=50)
        self.completion_entry.pack(side="left", padx=(0, 5), fill="x", expand=True)
        ttk.Button(completion_frame, text="📂 Browse", 
                  command=lambda: self.browse_file(self.completion_file_var, "Excel Files", "*.xlsx"),
                  style="TPITBrowse.TButton").pack(side="right")
        
        # Process button
        process_frame = ttk.Frame(main_container)
        process_frame.pack(fill="x", pady=10)
        ttk.Button(process_frame, text="� Process Files", 
                  command=self.process_tpit_files,
                  style="TPITProcess.TButton").pack(side="left", padx=5)
        
        # Output frame
        output_frame = ttk.LabelFrame(main_container, text="Processing Output", padding=10)
        output_frame.pack(fill="both", expand=True, pady=(10, 0))
        
        self.tpit_output_text = scrolledtext.ScrolledText(output_frame, height=15, wrap=tk.WORD, font=('Courier', 9))
        self.tpit_output_text.pack(fill="both", expand=True)
        
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
        print(f"log_message called with: {message}")
        if log_widget is None:
            log_widget = self.dynamic_log_text
            print("Using default log widget (dynamic_log_text)")
        else:
            print(f"Using specified log widget: {log_widget}")
        
        try:
            log_widget.insert(tk.END, f"{message}\n")
            log_widget.see(tk.END)
            self.root.update_idletasks()
            print("Message inserted successfully")
        except Exception as e:
            print(f"Error in log_message: {e}")
    
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
        
        # Initialize PSSE modules when needed
        self.update_status("Initializing PSSE modules for ACCC analysis...")
        if not initialize_psse():
            messagebox.showerror("Error", "PSSE modules could not be initialized. Please check your PSSE installation.")
            self.update_status("Ready")
            return
        self.update_status("PSSE modules initialized. Starting ACCC analysis...")
        
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
        
        # Initialize PSSE modules when needed
        self.update_status("Initializing PSSE modules for Dynamic analysis...")
        if not initialize_psse():
            messagebox.showerror("Error", "PSSE modules could not be initialized. Please check your PSSE installation.")
            self.update_status("Ready")
            return
        self.update_status("PSSE modules initialized. Starting Dynamic analysis...")
        
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
    
    def clear_contingency_data(self):
        """Clear all contingency input data"""
        self.branch_data_text.delete("1.0", tk.END)
        self.generator_data_text.delete("1.0", tk.END)
        self.auto_data_text.delete("1.0", tk.END)
        self.p3p6_log_text.delete("1.0", tk.END)
        self.log_message("All data cleared.", self.p3p6_log_text)
    
    def generate_contingencies_from_data(self):
        """Generate contingencies from the pasted data"""
        print("BUTTON CLICKED! Function is being called.")  # This should show in IDLE console
        try:
            print("About to clear log text...")
            # Clear previous log first
            self.p3p6_log_text.delete(1.0, tk.END)
            print("Log cleared, about to log message...")
            self.log_message("Button clicked - starting validation...", self.p3p6_log_text)
            print("First log message sent...")
            
            # Debug: Check the actual content of text boxes
            branch_data = self.branch_data_text.get("1.0", tk.END).strip()
            generator_data = self.generator_data_text.get("1.0", tk.END).strip()
            auto_data = self.auto_data_text.get("1.0", tk.END).strip()
            output_file = self.p3p6_output_file.get().strip()
            
            self.log_message(f"Branch data length: {len(branch_data)}", self.p3p6_log_text)
            self.log_message(f"Generator data length: {len(generator_data)}", self.p3p6_log_text)
            self.log_message(f"Auto data length: {len(auto_data)}", self.p3p6_log_text)
            self.log_message(f"Output file: '{output_file}'", self.p3p6_log_text)
            
            # Validate inputs
            if not self.validate_contingency_inputs():
                self.log_message("Validation failed - stopping", self.p3p6_log_text)
                return
            
            self.log_message("Validation passed - starting generation...", self.p3p6_log_text)
            
            # Start progress indication
            self.p3p6_progress_bar.start()
            self.is_running.set(True)
            
            self.log_message("Starting contingency generation...", self.p3p6_log_text)
            
            # Run contingency generation in separate thread
            thread = threading.Thread(target=self.contingency_worker, daemon=True)
            thread.start()
            
        except Exception as e:
            self.log_message(f"ERROR in generate_contingencies_from_data: {str(e)}", self.p3p6_log_text)
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}", self.p3p6_log_text)
            messagebox.showerror("Error", f"Function failed: {str(e)}")
    
    def validate_contingency_inputs(self):
        """Validate contingency generation inputs"""
        # Check if at least one data type has content
        branch_data = self.branch_data_text.get("1.0", tk.END).strip()
        generator_data = self.generator_data_text.get("1.0", tk.END).strip()
        auto_data = self.auto_data_text.get("1.0", tk.END).strip()
        
        if not branch_data and not generator_data and not auto_data:
            messagebox.showerror("Input Error", "Please paste at least one type of data (branch, generator, or auto)")
            return False
        
        # Check output file
        output_file = self.p3p6_output_file.get().strip()
        if not output_file:
            messagebox.showerror("Input Error", "Please specify an output .con file")
            return False
        
        # Check if output directory exists, create if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                messagebox.showerror("Directory Error", f"Cannot create output directory: {str(e)}")
                return False
        
        return True
    
    def contingency_worker(self):
        """Worker thread for contingency generation"""
        try:
            output_file = self.p3p6_output_file.get().strip()
            self.log_message(f"Output file: {output_file}", self.p3p6_log_text)
            
            # Get raw text data for debugging
            branch_text = self.branch_data_text.get("1.0", tk.END).strip()
            generator_text = self.generator_data_text.get("1.0", tk.END).strip()
            auto_text = self.auto_data_text.get("1.0", tk.END).strip()
            
            self.log_message(f"Raw branch text length: {len(branch_text)}", self.p3p6_log_text)
            self.log_message(f"Raw generator text length: {len(generator_text)}", self.p3p6_log_text)
            self.log_message(f"Raw auto text length: {len(auto_text)}", self.p3p6_log_text)
            
            if branch_text:
                self.log_message(f"Branch text preview: {branch_text[:100]}...", self.p3p6_log_text)
            if generator_text:
                self.log_message(f"Generator text preview: {generator_text[:100]}...", self.p3p6_log_text)
            if auto_text:
                self.log_message(f"Auto text preview: {auto_text[:100]}...", self.p3p6_log_text)
            
            # Parse the input data
            branch_data = self.parse_data_lines(self.branch_data_text.get("1.0", tk.END), "branch")
            generator_data = self.parse_data_lines(self.generator_data_text.get("1.0", tk.END), "generator")
            auto_data = self.parse_data_lines(self.auto_data_text.get("1.0", tk.END), "auto")
            
            all_elements = branch_data + generator_data + auto_data
            
            if not all_elements:
                self.log_message("Error: No valid data found to process", self.p3p6_log_text)
                return
            
            self.log_message(f"Found {len(all_elements)} elements to process", self.p3p6_log_text)
            self.log_message(f"  - {len(branch_data)} branch elements", self.p3p6_log_text)
            self.log_message(f"  - {len(generator_data)} generator elements", self.p3p6_log_text)
            self.log_message(f"  - {len(auto_data)} auto elements", self.p3p6_log_text)
            
            # Generate contingencies
            with open(output_file, 'w') as f:
                # Write header
                f.write("// Contingency file generated by AEMyLab Contingency Generator\n")
                f.write(f"// Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("//\n")
                
                # Generate single contingencies
                self.log_message("Generating single contingencies...", self.p3p6_log_text)
                single_count = 0
                
                for element in all_elements:
                    element_type, from_bus, to_bus, ckt, from_name, to_name = element
                    
                    if element_type == "branch":
                        contingency_name = f"DB_{single_count + 1}"
                        f.write(f"CONTINGENCY '{contingency_name}'\n")
                        f.write(f"   OPEN BRANCH FROM BUS {from_bus} TO BUS {to_bus} CKT {ckt}                   /* {from_name} - {to_name}\n")
                        f.write("END\n")
                    elif element_type == "generator":
                        contingency_name = f"GEN_{single_count + 1}"
                        f.write(f"CONTINGENCY '{contingency_name}'\n")
                        f.write(f"   DISCONNECT UNIT {from_bus} '{ckt}'                   /* {from_name}\n")
                        f.write("END\n")
                    elif element_type == "auto":
                        contingency_name = f"AUTO_{single_count + 1}"
                        f.write(f"CONTINGENCY '{contingency_name}'\n")
                        f.write(f"   OPEN BRANCH FROM BUS {from_bus} TO BUS {to_bus} CKT {ckt}                   /* {from_name} - {to_name}\n")
                        f.write("END\n")
                    
                    single_count += 1
                
                self.log_message(f"Generated {single_count} single contingencies", self.p3p6_log_text)
                
                # Generate double contingencies P3 and P6 - combinations of all elements
                self.log_message("Generating double contingencies P3 and P6...", self.p3p6_log_text)
                double_count = 0
                
                for i in range(len(all_elements)):
                    for j in range(i + 1, len(all_elements)):
                        elem1 = all_elements[i]
                        elem2 = all_elements[j]
                        
                        type1, from1, to1, ckt1, name1_from, name1_to = elem1
                        type2, from2, to2, ckt2, name2_from, name2_to = elem2
                        
                        contingency_name = f"DBL_{i+1}_{j+1}"
                        f.write(f"CONTINGENCY '{contingency_name}'\n")
                        
                        # First element
                        if type1 == "branch":
                            f.write(f"   OPEN BRANCH FROM BUS {from1} TO BUS {to1} CKT {ckt1}                   /* {name1_from} - {name1_to}\n")
                        elif type1 == "generator":
                            f.write(f"   DISCONNECT UNIT {from1} '{ckt1}'                   /* {name1_from}\n")
                        elif type1 == "auto":
                            f.write(f"   OPEN BRANCH FROM BUS {from1} TO BUS {to1} CKT {ckt1}                   /* {name1_from} - {name1_to}\n")
                        
                        # Second element
                        if type2 == "branch":
                            f.write(f"   OPEN BRANCH FROM BUS {from2} TO BUS {to2} CKT {ckt2}                   /* {name2_from} - {name2_to}\n")
                        elif type2 == "generator":
                            f.write(f"   DISCONNECT UNIT {from2} '{ckt2}'                   /* {name2_from}\n")
                        elif type2 == "auto":
                            f.write(f"   OPEN BRANCH FROM BUS {from2} TO BUS {to2} CKT {ckt2}                   /* {name2_from} - {name2_to}\n")
                        
                        f.write("END\n")
                        double_count += 1
                
                self.log_message(f"Generated {double_count} double contingencies P3 and P6", self.p3p6_log_text)
                self.log_message(f"Total contingencies: {single_count + double_count}", self.p3p6_log_text)
                
                # Add final END statement as in allcon_rev1.py
                f.write("\nEND\n")
            
            self.log_message(f"Contingency file created successfully: {output_file}", self.p3p6_log_text)
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Contingencies generated successfully!\n\nSingle: {single_count}\nDouble: {double_count}\nTotal: {single_count + double_count}\n\nFile: {output_file}"))
            
        except Exception as e:
            self.log_message(f"Error generating contingencies: {str(e)}", self.p3p6_log_text)
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}", self.p3p6_log_text)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to generate contingencies:\n{str(e)}"))
        finally:
            self.root.after(0, lambda: self.p3p6_progress_bar.stop())
            self.is_running.set(False)
    
    def parse_data_lines(self, data_text, data_type):
        """Parse pasted data lines into structured format"""
        if not data_text.strip():
            return []
        
        elements = []
        lines = data_text.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('#'):
                continue
            
            # Skip header lines that contain text like "Bus Number", "From Bus", etc.
            if any(header in line.upper() for header in ['BUS NUMBER', 'FROM BUS', 'TO BUS', 'BUS NAME', 'FROM BUS NAME', 'TO BUS NAME']):
                self.log_message(f"Skipping header line {line_num}: {line[:50]}...", self.p3p6_log_text)
                continue
            
            try:
                # Parse different delimiters
                if ',' in line:
                    fields = [f.strip() for f in line.split(',')]
                elif ';' in line:
                    fields = [f.strip() for f in line.split(';')]
                elif '|' in line:
                    fields = [f.strip() for f in line.split('|')]
                elif '\t' in line:
                    fields = [f.strip() for f in line.split('\t')]
                else:
                    fields = line.split()
                
                # Remove empty fields
                fields = [f for f in fields if f]
                
                if len(fields) < 2:
                    self.log_message(f"Warning: Line {line_num} in {data_type} data has insufficient fields: {line}", self.p3p6_log_text)
                    continue
                
                # Extract bus numbers - handle different data formats
                try:
                    if data_type == 'branch' or data_type == 'auto':
                        # Format: from_bus, from_name, to_bus, to_name, id
                        if len(fields) >= 5:
                            from_bus = int(float(fields[0]))
                            from_name = str(fields[1]).strip()
                            to_bus = int(float(fields[2]))
                            to_name = str(fields[3]).strip()
                            ckt = str(fields[4]).strip()
                        else:
                            # Fallback to old format: from_bus, to_bus, id, from_name, to_name
                            from_bus = int(float(fields[0]))
                            to_bus = int(float(fields[1])) if len(fields) > 1 else from_bus
                            ckt = fields[2] if len(fields) > 2 else "1"
                            from_name = fields[3] if len(fields) > 3 else f"BUS_{from_bus}"
                            to_name = fields[4] if len(fields) > 4 else f"BUS_{to_bus}"
                    else:  # generator data
                        # Format: bus_number, bus_name, id
                        from_bus = int(float(fields[0]))
                        to_bus = from_bus
                        from_name = str(fields[1]).strip() if len(fields) > 1 else f"BUS_{from_bus}"
                        to_name = from_name
                        ckt = str(fields[2]).strip() if len(fields) > 2 else "1"
                        
                except ValueError:
                    self.log_message(f"Warning: Line {line_num} in {data_type} data has invalid bus numbers: {line[:50]}...", self.p3p6_log_text)
                    continue
                
                # Clean names (remove quotes and special characters)
                from_name = from_name.strip('\'"').replace(' ', '_')
                to_name = to_name.strip('\'"').replace(' ', '_')
                
                elements.append((data_type, from_bus, to_bus, ckt, from_name, to_name))
                
            except Exception as e:
                self.log_message(f"Error parsing line {line_num} in {data_type} data: {line} - {str(e)}", self.p3p6_log_text)
                continue
        
        return elements

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
        self.main_notebook.select(5)  # Select Results tab
    
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
            self.main_notebook.select(5)  # Select Results tab
    
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
        shield = self.idv_shield.get() if 'Shield' in self.impedance_df.columns else ''
        
        print(f"Debug: kV = '{kV}', Conductors_Per_Phase = '{Conductors_Per_Phase}', Geometry = '{geom}', Shield = '{shield}'")
        
        # If no kV/CPP yet selected, clear dependent menus
        if not kV or not Conductors_Per_Phase:
            if 'Geometry' in self.impedance_df.columns:
                self.idv_geometry_combo['values'] = []
                self.idv_geometry.set('')
            if 'Shield' in self.impedance_df.columns:
                self.idv_shield_combo['values'] = []
                self.idv_shield.set('')
            self.idv_conductor_combo['values'] = []
            self.idv_conductor.set('')
            # Environmental Index is independent, so don't clear it
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
            
            # Populate Shield options if column exists
            if 'Shield' in self.impedance_df.columns:
                shields = sorted(filtered['Shield'].dropna().astype(str).unique()) if not filtered.empty else []
                print(f"Debug: Available shields: {shields}")
                self.idv_shield_combo['values'] = shields
                # If current shield not valid, pick first or clear
                if shield not in shields:
                    if shields:
                        self.idv_shield.set(shields[0])
                        shield = shields[0]
                        print(f"Debug: Auto-set shield to: {shield}")
                    else:
                        self.idv_shield.set('')
                        shield = ''

            # Third level filter: by shield if present
            if 'Shield' in self.impedance_df.columns and shield:
                filtered = filtered[filtered['Shield'].astype(str) == shield]
            
            print(f"Debug: Filtered rows count (post-shield): {len(filtered)}")
            
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
                'SUWANNEE': 'Suwannee', 'MERRIMACK': 'Merrimack',
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
    
    def get_mva_rating_from_csv(self, conductor_name, environmental_index, kv_level):
        """Get MVA rating from Ratings.csv based on conductor, environmental index, and voltage level"""
        if self.ratings_df is None:
            return None
            
        try:
            # Map conductor name to ratings CSV format
            conductors_per_phase = int(self.idv_conductors_per_phase.get())
            construction_type = self.idv_construction_type.get()
            mapped_conductor = self.map_conductor_name(conductor_name, conductors_per_phase)
            
            # Add "New Construction" suffix if selected
            if construction_type == "New Construction":
                mapped_conductor += " New Construction"
            
            print(f"Debug: Looking for MVA rating - Conductor: '{mapped_conductor}', Env Index: {environmental_index}, kV: {kv_level}")
            
            # Filter by conductor size and environmental index
            rating_match = self.ratings_df[
                (self.ratings_df['Conductor Size'].str.contains(mapped_conductor, case=False, na=False, regex=False)) &
                (self.ratings_df['Environmental Index'] == int(environmental_index))
            ]
            
            if rating_match.empty:
                # Try without exact match - use partial matching
                conductor_parts = mapped_conductor.split()
                if len(conductor_parts) >= 2:
                    # Try matching just the size and main type
                    size_part = conductor_parts[0]
                    type_part = conductor_parts[-1] if len(conductor_parts) > 1 else ""
                    
                    rating_match = self.ratings_df[
                        (self.ratings_df['Conductor Size'].str.contains(size_part, case=False, na=False)) &
                        (self.ratings_df['Conductor Size'].str.contains(type_part, case=False, na=False)) &
                        (self.ratings_df['Environmental Index'] == int(environmental_index))
                    ]
                    print(f"Debug: Trying partial match with size '{size_part}' and type '{type_part}'")
            
            if not rating_match.empty:
                # Determine which MVA column to use based on voltage level
                kv_int = int(kv_level)
                mva_column = None
                
                if kv_int == 69:
                    mva_column = 'MVA Rating 69kV'
                elif kv_int == 138:
                    mva_column = 'MVA Rating 138kV'
                elif kv_int == 345:
                    mva_column = 'MVA Rating 345kV'
                
                if mva_column and mva_column in rating_match.columns:
                    mva_value = rating_match.iloc[0][mva_column]
                    # Check if it's "not used" or similar
                    if isinstance(mva_value, str) and ('not used' in mva_value.lower() or mva_value.strip() == ''):
                        print(f"Debug: MVA rating shows 'not used' for {kv_int}kV")
                        return None
                    else:
                        try:
                            mva_rating = float(mva_value)
                            print(f"Debug: Found MVA rating: {mva_rating} for {kv_int}kV")
                            return mva_rating
                        except (ValueError, TypeError):
                            print(f"Debug: Could not convert MVA value '{mva_value}' to float")
                            return None
                else:
                    print(f"Debug: MVA column '{mva_column}' not found in ratings CSV")
                    return None
            else:
                print(f"Debug: No matching conductor found in ratings CSV")
                return None
                
        except Exception as e:
            print(f"Debug: Error getting MVA rating from CSV: {e}")
            return None
    
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
            
            # If ratings file is loaded, suggest selecting Environmental Index
            if self.ratings_df is not None and not self.idv_environmental_index.get():
                result = messagebox.askyesno("Environmental Index", 
                    "Ratings.csv file is loaded but no Environmental Index is selected.\n\n" +
                    "Do you want to continue with calculated MVA ratings instead of using CSV ratings?")
                if not result:
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
            R = round(R_per_mile * miles, 5)
            X = round(X_per_mile * miles, 5)
            B = round(B_per_mile * miles, 5)
            
            # Calculate MVA from amps and kV
            mva = self.calculate_mva(amps, kV)
            
            # Initialize ratings array with 12 slots (as per PSSE API)
            ratings = [0.0] * 12
            
            # Get Environmental Index for RATE4 lookup
            env_index = self.idv_environmental_index.get()
            
            # RATE1-3 always use calculated MVA from amps/kV (minimum ratings)
            calculated_mva = self.calculate_mva(amps, kV)
            print(f"Debug: Using calculated MVA for RATE1-3: {calculated_mva} (from {amps} Amps at {kV}kV)")
            
            # Set first 3 ratings to calculated MVA
            ratings[0] = calculated_mva  # RATE1
            ratings[1] = calculated_mva  # RATE2  
            ratings[2] = calculated_mva  # RATE3
            
            # Get RATE4 from Ratings.csv based on Environmental Index
            if self.ratings_df is not None and env_index:
                try:
                    # Show available conductors and columns for debugging
                    if 'Conductor Size' in self.ratings_df.columns:
                        available_conductors = self.ratings_df['Conductor Size'].unique()[:10]  # Show first 10
                        print(f"Debug: Available conductors in CSV: {list(available_conductors)}")
                    
                    # Show all column names to debug MVA Rating columns
                    print(f"Debug: All CSV columns: {list(self.ratings_df.columns)}")
                    
                    # Map the conductor name to ratings CSV format
                    mapped_conductor = self.map_conductor_name(ctype, Conductors_Per_Phase)
                    print(f"Debug: Mapped conductor from '{ctype}' to '{mapped_conductor}' for Environmental Index {env_index}")
                    
                    # Filter by conductor and environmental index
                    rating_filtered = self.ratings_df[
                        (self.ratings_df['Conductor Size'].str.contains(mapped_conductor, case=False, na=False, regex=False)) &
                        (self.ratings_df['Environmental Index'] == int(env_index))
                    ]
                    
                    if rating_filtered.empty:
                        # Try without "Bundled" keyword
                        simple_conductor = mapped_conductor.replace(" Bundled", "")
                        rating_filtered = self.ratings_df[
                            (self.ratings_df['Conductor Size'].str.contains(simple_conductor, case=False, na=False, regex=False)) &
                            (self.ratings_df['Environmental Index'] == int(env_index))
                        ]
                        print(f"Debug: Trying without 'Bundled': '{simple_conductor}' with Env Index {env_index}")
                    
                    if rating_filtered.empty:
                        # Try more flexible matching with different formats
                        conductor_parts = mapped_conductor.replace("x2", "").replace(" Bundled", "").split()
                        if len(conductor_parts) >= 2:
                            size_part = conductor_parts[0]
                            name_part = conductor_parts[-1] if len(conductor_parts) > 1 else ""
                            
                            # Try matching by size and name (most specific parts)
                            rating_filtered = self.ratings_df[
                                (self.ratings_df['Conductor Size'].str.contains(size_part, case=False, na=False)) &
                                (self.ratings_df['Conductor Size'].str.contains(name_part, case=False, na=False)) &
                                (self.ratings_df['Environmental Index'] == int(env_index))
                            ]
                            print(f"Debug: Trying flexible match with size '{size_part}' and name '{name_part}' for Env Index {env_index}")
                    
                    if rating_filtered.empty:
                        # Last resort: try just the size and bundling info
                        if "x2" in mapped_conductor or Conductors_Per_Phase == 2:
                            # Look for bundled conductors
                            size_only = mapped_conductor.split()[0].replace("x2", "")
                            rating_filtered = self.ratings_df[
                                (self.ratings_df['Conductor Size'].str.contains(f"{size_only}x2", case=False, na=False)) &
                                (self.ratings_df['Environmental Index'] == int(env_index))
                            ]
                            print(f"Debug: Trying bundled match with size '{size_only}x2' for Env Index {env_index}")
                        else:
                            # Look for single conductors
                            size_only = mapped_conductor.split()[0]
                            rating_filtered = self.ratings_df[
                                (self.ratings_df['Conductor Size'].str.contains(f"^{size_only}\\s", case=False, na=False, regex=True)) &
                                (~self.ratings_df['Conductor Size'].str.contains("x2", case=False, na=False)) &
                                (self.ratings_df['Environmental Index'] == int(env_index))
                            ]
                            print(f"Debug: Trying single conductor match with size '{size_only}' for Env Index {env_index}")
                    
                    if not rating_filtered.empty:
                        print(f"Debug: Found conductor match for Environmental Index {env_index}: {rating_filtered['Conductor Size'].iloc[0]}")
                        
                        # RATE4 should use MVA Rating columns based on voltage level
                        # Environmental Index determines which row to use, voltage determines which MVA column
                        conductor_rating = 0.0
                        
                        # Determine which MVA column to use based on voltage level
                        kv_int = int(kV)
                        mva_column = None
                        
                        # Find the actual MVA column by searching for voltage in column names
                        voltage_patterns = []
                        if kv_int == 69:
                            voltage_patterns = ['69', 'sixty', 'sixtynine']
                        elif kv_int == 138:
                            voltage_patterns = ['138', 'onethirtyeight', 'one thirty eight']
                        elif kv_int == 345:
                            voltage_patterns = ['345', 'threefourty', 'three forty']
                        
                        # Search for MVA Rating column with the voltage
                        for col_name in rating_filtered.columns:
                            col_lower = col_name.lower()
                            if 'mva' in col_lower and 'rating' in col_lower:
                                for pattern in voltage_patterns:
                                    if pattern in col_lower:
                                        mva_column = col_name
                                        break
                                if mva_column:
                                    break
                        
                        print(f"Debug: Looking for MVA column for {kv_int}kV, found: '{mva_column}'")
                        
                        # Try the specific MVA column first
                        if mva_column and mva_column in rating_filtered.columns:
                            try:
                                col_value = rating_filtered.iloc[0][mva_column]
                                if pd.notna(col_value) and str(col_value).strip() != '' and 'not used' not in str(col_value).lower():
                                    conductor_rating = float(col_value)
                                    print(f"Debug: Found RATE4 from '{mva_column}': {conductor_rating} for Environmental Index {env_index}")
                                else:
                                    print(f"Debug: MVA column '{mva_column}' shows 'not used' or empty for {kv_int}kV")
                            except (ValueError, TypeError):
                                print(f"Debug: Could not convert MVA value '{col_value}' to float")
                        
                        # If no MVA rating found, try fallback columns
                        if conductor_rating == 0.0:
                            fallback_columns = ['MVA Rating', 'Rating', 'Conductor Ampacity', 'Ampacity']
                            for col in fallback_columns:
                                if col in rating_filtered.columns:
                                    try:
                                        col_value = rating_filtered.iloc[0][col]
                                        if pd.notna(col_value) and str(col_value).strip() != '' and 'not used' not in str(col_value).lower():
                                            conductor_rating = float(col_value)
                                            print(f"Debug: Found RATE4 from fallback column '{col}': {conductor_rating} for Environmental Index {env_index}")
                                            break
                                    except (ValueError, TypeError):
                                        continue
                        
                        if conductor_rating > 0:
                            ratings[3] = conductor_rating  # RATE4
                            print(f"Debug: Set RATE4 to Environmental Index {env_index} rating: {conductor_rating}")
                        else:
                            # Fall back to same as RATE1-3 if no valid environmental rating found
                            ratings[3] = mva
                            print(f"Debug: No valid Environmental Index {env_index} rating found, using MVA rating: {mva}")
                    else:
                        print(f"Debug: No conductor match found for '{mapped_conductor}' with Environmental Index {env_index}")
                        # Set RATE4 to same as RATE1-3 if no specific conductor rating found
                        ratings[3] = mva
                        print(f"Debug: Set RATE4 to default MVA rating: {mva}")
                        
                except Exception as e:
                    print(f"Debug: Error getting Environmental Index rating: {e}")
                    # Set RATE4 to same as RATE1-3 as fallback
                    ratings[3] = mva  
                    print(f"Debug: Set RATE4 to fallback MVA rating: {mva}")
            else:
                # If no ratings CSV loaded or no Environmental Index selected, set RATE4 to same as RATE1-3
                ratings[3] = mva
                print(f"Debug: No ratings CSV or Environmental Index, set RATE4 to MVA rating: {mva}")
            
            # Set 5th rating to 9999
            ratings[4] = 9999.0  # RATE5
            
            # Ratings 6-12 remain 0.0 (already initialized)
            print(f"Debug: Final ratings array: {ratings}")

            idv = f"BAT_BRANCH_CHNG_3,{from_bus},{to_bus},'1',,,,,,,{R},{X},{B}," + \
                  "," * 17 + ",".join(map(str, ratings)) + "," * 0 + ";"

            # Store the IDV line for export
            self.current_idv_line = idv
            
            # Auto-populate the export filename
            if from_bus and to_bus:
                default_filename = f"line_{from_bus}_{to_bus}"
                self.idv_filename.set(default_filename)

            # Display results
            self.idv_output_text.delete("1.0", tk.END)
            self.idv_output_text.insert(tk.END, idv + f"\n\nMVA Rating: {calculated_mva}")
            
            # Show source of MVA rating (always calculated for RATE1-3)
            construction_type = self.idv_construction_type.get()
            self.idv_output_text.insert(tk.END, f" (calculated from {amps} Amps at {kV}kV)")
            
            self.idv_output_text.insert(tk.END, f"\nLength: {miles} miles")
            self.idv_output_text.insert(tk.END, f"\nImpedance per mile: R={R_per_mile}, X={X_per_mile}, B={B_per_mile}")
            self.idv_output_text.insert(tk.END, f"\nTotal impedance: R={R}, X={X}, B={B}")
            if geom:
                self.idv_output_text.insert(tk.END, f"\nGeometry: {geom}")
            if env_index:
                self.idv_output_text.insert(tk.END, f"\nEnvironmental Index: {env_index}")
            if construction_type:
                self.idv_output_text.insert(tk.END, f"\nConstruction Type: {construction_type}")
            
            # Show ratings explanation
            rate4_source = "calculated" if ratings[3] == calculated_mva else "from Ratings.csv"
            self.idv_output_text.insert(tk.END, f"\nRatings: RATE1-3={calculated_mva} MVA (calculated), RATE4={ratings[3]} ({rate4_source}), RATE5=9999")

        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numeric values for bus numbers and amperage")
        except KeyError as e:
            messagebox.showerror("Error", f"Missing column in CSV file: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def export_idv(self):
        """Export the generated IDV line to a .idv file"""
        try:
            # Check if IDV has been generated
            if not hasattr(self, 'current_idv_line') or not self.current_idv_line:
                messagebox.showwarning("Warning", "Please generate an IDV first before exporting.")
                return
            
            # Get the filename from the input field
            filename = self.idv_filename.get().strip()
            
            # Create default filename if none provided
            if not filename:
                from_bus = self.idv_from_bus.get().strip()
                to_bus = self.idv_to_bus.get().strip()
                if from_bus and to_bus:
                    filename = f"line_{from_bus}_{to_bus}"
                else:
                    filename = "generated_line"
            
            # Add .idv extension if not present
            if not filename.lower().endswith('.idv'):
                filename += '.idv'
            
            # Ask user for save location
            file_path = filedialog.asksaveasfilename(
                title="Save IDV File",
                defaultextension=".idv",
                filetypes=[("IDV Files", "*.idv"), ("All Files", "*.*")]
            )
            
            # If user didn't specify full path, use the filename from input
            if file_path and not os.path.basename(file_path):
                directory = os.path.dirname(file_path) if file_path else os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(directory, filename)
            
            if file_path:
                # Write only the IDV line to the file
                with open(file_path, 'w') as f:
                    f.write(self.current_idv_line + '\n')
                
                messagebox.showinfo("Success", f"IDV file exported successfully to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export IDV file: {str(e)}")

    def perform_vlookup(self, lookup_value, lookup_df, lookup_col, return_col):
        """
        Perform VLOOKUP operation similar to Excel
        
        Args:
            lookup_value: Value to search for
            lookup_df: DataFrame to search in
            lookup_col: Column index to search in (0-based)
            return_col: Column index to return value from (0-based)
        
        Returns:
            Matched value or None if not found
        """
        try:
            if pd.isna(lookup_value) or lookup_value == '':
                return None
            
            # Ensure we have enough columns
            if lookup_col >= len(lookup_df.columns) or return_col >= len(lookup_df.columns):
                return None
            
            # Convert lookup_value to string for comparison
            lookup_str = str(lookup_value).strip()
            
            # Search for exact match in the lookup column
            mask = lookup_df.iloc[:, lookup_col].astype(str).str.strip() == lookup_str
            matches = lookup_df[mask]
            
            if not matches.empty:
                # Return the first match from the return column
                return matches.iloc[0, return_col]
            else:
                return None
                
        except Exception as e:
            print(f"VLOOKUP error for value '{lookup_value}': {str(e)}")
            return None

    def process_tpit_files(self):
        """Process TPIT files and perform VLOOKUP operations"""
        try:
            # Check if all files are selected
            planning_file = self.planning_file_var.get()
            existing_file = self.existing_file_var.get()
            mod_file = self.mod_file_var.get()
            completion_file = self.completion_file_var.get()
            
            if not all([planning_file, existing_file, mod_file, completion_file]):
                messagebox.showerror("Error", "Please select all four input files")
                return
            
            # Clear output text
            self.tpit_output_text.delete("1.0", tk.END)
            self.tpit_output_text.insert(tk.END, "🔄 Starting TPIT processing...\n\n")
            self.tpit_output_text.update()
            
            # Import openpyxl for Excel operations
            from openpyxl import load_workbook
            import pandas as pd
            import os
            
            self.tpit_output_text.insert(tk.END, "📁 Loading input files...\n")
            self.tpit_output_text.update()
            
            # Load Planning Data Export (Excel) - This will be PDE.XLSX
            try:
                planning_df = pd.read_excel(planning_file)
                self.tpit_output_text.insert(tk.END, f"✅ PDE.XLSX loaded: {len(planning_df)} rows\n")
            except Exception as e:
                raise Exception(f"Error loading PDE.XLSX: {str(e)}")
            
            # Load Existing Report (Excel) - This will be used as reference for other file paths
            try:
                existing_df = pd.read_excel(existing_file)
                self.tpit_output_text.insert(tk.END, f"✅ Existing Report loaded: {len(existing_df)} rows\n")
            except Exception as e:
                raise Exception(f"Error loading Existing Report: {str(e)}")
            
            # Load MOD Report (Excel)
            try:
                mod_df = pd.read_excel(mod_file)
                self.tpit_output_text.insert(tk.END, f"✅ MOD Report loaded: {len(mod_df)} rows\n")
            except Exception as e:
                raise Exception(f"Error loading MOD Report: {str(e)}")
            
            # Load Project Completion Report (Excel)
            try:
                completion_df = pd.read_excel(completion_file)
                self.tpit_output_text.insert(tk.END, f"✅ Project Completion Report loaded: {len(completion_df)} rows\n")
            except Exception as e:
                raise Exception(f"Error loading Project Completion Report: {str(e)}")
            
            self.tpit_output_text.insert(tk.END, "\n🔍 Analyzing file structures...\n")
            self.tpit_output_text.update()
            
            # Display column information for verification
            self.tpit_output_text.insert(tk.END, f"\nPlanning Data Export columns: {list(planning_df.columns)}\n")
            self.tpit_output_text.insert(tk.END, f"Existing Report columns: {list(existing_df.columns)}\n") 
            self.tpit_output_text.insert(tk.END, f"MOD Report columns: {list(mod_df.columns)}\n")
            self.tpit_output_text.insert(tk.END, f"Project Completion Report columns: {list(completion_df.columns)}\n")
            self.tpit_output_text.update()
            
            # Implement VLOOKUP logic
            self.tpit_output_text.insert(tk.END, "\n🔄 Implementing VLOOKUP operations...\n")
            self.tpit_output_text.update()
            
            # Load additional Excel files for VLOOKUP operations
            base_dir = os.path.dirname(planning_file)
            
            # Load PPDE.XLSX for first VLOOKUP
            ppde_df = None
            ppde_paths = [
                os.path.join(base_dir, 'PPDE.xlsx'),
                os.path.join(base_dir, 'PPDE.XLSX'),
                existing_file.replace('PDE', 'PPDE') if 'PDE' in existing_file else os.path.join(base_dir, 'PPDE.xlsx')
            ]
            
            for ppde_path in ppde_paths:
                try:
                    if os.path.exists(ppde_path):
                        ppde_df = pd.read_excel(ppde_path, sheet_name='PDE 0')
                        self.tpit_output_text.insert(tk.END, f"✅ PPDE.XLSX loaded from: {ppde_path}\n")
                        break
                except Exception as e:
                    continue
            
            if ppde_df is None:
                self.tpit_output_text.insert(tk.END, "⚠️ PPDE.XLSX not found, skipping first VLOOKUP\n")
            
            # Load MPR.XLSX for second VLOOKUP
            mpr_df = None
            mpr_paths = [
                os.path.join(base_dir, 'MPR.xlsx'),
                os.path.join(base_dir, 'MPR.XLSX'),
                existing_file.replace('PDE', 'MPR') if 'PDE' in existing_file else os.path.join(base_dir, 'MPR.xlsx')
            ]
            
            for mpr_path in mpr_paths:
                try:
                    if os.path.exists(mpr_path):
                        mpr_df = pd.read_excel(mpr_path, sheet_name='Project_Phases')
                        self.tpit_output_text.insert(tk.END, f"✅ MPR.XLSX loaded from: {mpr_path}\n")
                        break
                except Exception as e:
                    continue
            
            if mpr_df is None:
                self.tpit_output_text.insert(tk.END, "⚠️ MPR.XLSX not found, skipping second VLOOKUP\n")
            
            # Load PC.XLSX for third VLOOKUP
            pc_df = None
            pc_paths = [
                os.path.join(base_dir, 'PC.xlsx'),
                os.path.join(base_dir, 'PC.XLSX'),
                existing_file.replace('PDE', 'PC') if 'PDE' in existing_file else os.path.join(base_dir, 'PC.xlsx')
            ]
            
            for pc_path in pc_paths:
                try:
                    if os.path.exists(pc_path):
                        pc_df = pd.read_excel(pc_path, sheet_name='Sheet')
                        self.tpit_output_text.insert(tk.END, f"✅ PC.XLSX loaded from: {pc_path}\n")
                        break
                except Exception as e:
                    continue
            
            if pc_df is None:
                self.tpit_output_text.insert(tk.END, "⚠️ PC.XLSX not found, skipping third VLOOKUP\n")
            
            self.tpit_output_text.update()
            
            # Create the result DataFrame starting with Planning Data Export
            result_df = planning_df.copy()
            
            # Perform VLOOKUP operations
            self.tpit_output_text.insert(tk.END, "\n🔍 Performing VLOOKUP operations...\n")
            self.tpit_output_text.update()
            
            # VLOOKUP 1: VLOOKUP(A2,'[PPDE.XLSX]PDE 0'!$B:$F,5,FALSE)
            if ppde_df is not None:
                self.tpit_output_text.insert(tk.END, "📊 VLOOKUP 1: Looking up values from PPDE.XLSX...\n")
                result_df['VLOOKUP1_Result'] = result_df.iloc[:, 0].apply(
                    lambda x: self.perform_vlookup(x, ppde_df, lookup_col=1, return_col=4)
                )
                self.tpit_output_text.insert(tk.END, f"✅ VLOOKUP 1 completed: {result_df['VLOOKUP1_Result'].notna().sum()} matches found\n")
            
            # VLOOKUP 2: VLOOKUP(G2,[MPR.XLSX]Project_Phases!$B:$G,6,FALSE)
            if mpr_df is not None and len(result_df.columns) > 6:
                self.tpit_output_text.insert(tk.END, "📊 VLOOKUP 2: Looking up values from MPR.XLSX...\n")
                result_df['VLOOKUP2_Result'] = result_df.iloc[:, 6].apply(
                    lambda x: self.perform_vlookup(x, mpr_df, lookup_col=1, return_col=5)
                )
                self.tpit_output_text.insert(tk.END, f"✅ VLOOKUP 2 completed: {result_df['VLOOKUP2_Result'].notna().sum()} matches found\n")
            
            # VLOOKUP 3: VLOOKUP(V2,'[PC.XLSX]Sheet'!$H:$N,7,FALSE)
            if pc_df is not None and len(result_df.columns) > 21:  # V is column 22 (0-indexed = 21)
                self.tpit_output_text.insert(tk.END, "📊 VLOOKUP 3: Looking up values from PC.XLSX...\n")
                result_df['VLOOKUP3_Result'] = result_df.iloc[:, 21].apply(
                    lambda x: self.perform_vlookup(x, pc_df, lookup_col=7, return_col=6)
                )
                self.tpit_output_text.insert(tk.END, f"✅ VLOOKUP 3 completed: {result_df['VLOOKUP3_Result'].notna().sum()} matches found\n")
            
            # Map Project Completion Date from Project Completion Report's Task Finish Date
            self.tpit_output_text.insert(tk.END, "📅 Mapping Project Completion Date from Task Finish Date...\n")
            self.tpit_output_text.update()
            
            # Find Task Finish Date column in completion report
            task_finish_col = None
            for i, col in enumerate(completion_df.columns):
                if 'task' in col.lower() and 'finish' in col.lower() and 'date' in col.lower():
                    task_finish_col = i
                    self.tpit_output_text.insert(tk.END, f"Found Task Finish Date column: '{col}'\n")
                    break
            
            if task_finish_col is not None:
                # Find Project Name column in completion report for matching
                project_name_col = None
                for i, col in enumerate(completion_df.columns):
                    if 'project' in col.lower() and 'name' in col.lower():
                        project_name_col = i
                        self.tpit_output_text.insert(tk.END, f"Found Project Name column: '{col}'\n")
                        break
                
                if project_name_col is not None:
                    # Perform lookup to get Task Finish Date for each project
                    result_df['Project_Completion_Date'] = result_df.iloc[:, 0].apply(  # Assuming project name is in first column
                        lambda x: self.perform_vlookup(x, completion_df, lookup_col=project_name_col, return_col=task_finish_col)
                    )
                    matches = result_df['Project_Completion_Date'].notna().sum()
                    self.tpit_output_text.insert(tk.END, f"✅ Project Completion Date mapping completed: {matches} dates found\n")
                else:
                    self.tpit_output_text.insert(tk.END, "⚠️ Project Name column not found in Project Completion Report\n")
            else:
                self.tpit_output_text.insert(tk.END, "⚠️ Task Finish Date column not found in Project Completion Report\n")
            
            # Save as NPDE.CSV
            output_path = os.path.join(os.path.dirname(planning_file), "NPDE.csv")
            result_df.to_csv(output_path, index=False)
            
            self.tpit_output_text.insert(tk.END, f"\n✅ NPDE.CSV created successfully at: {output_path}\n")
            self.tpit_output_text.insert(tk.END, f"📈 Total rows processed: {len(result_df)}\n")
            self.tpit_output_text.insert(tk.END, f"📊 Total columns in output: {len(result_df.columns)}\n")
            
            self.tpit_output_text.insert(tk.END, "\n🎉 VLOOKUP processing completed successfully!\n")
            
        except Exception as e:
            self.tpit_output_text.insert(tk.END, f"\n❌ Error: {str(e)}\n")
            messagebox.showerror("TPIT Error", str(e))

def main():
    root = tk.Tk()
    app = AEMyLabGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()