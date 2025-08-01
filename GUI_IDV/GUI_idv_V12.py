import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd

# Initialize global variables for dataframes
impedance_df = None
ratings_df = None

# Functions to load CSV files
def load_impedance_file():
    global impedance_df
    file_path = filedialog.askopenfilename(
        title="Select Line Impedance Estimator CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if file_path:
        try:
            impedance_df = pd.read_csv(file_path)
            print("Impedance CSV columns found:", impedance_df.columns.tolist())
            
            # Update the kV dropdown with new data
            kV_menu['values'] = sorted(impedance_df['kV'].unique().astype(str))
            
            # Clear dependent dropdowns
            conductor_menu['values'] = []
            conductor_var.set('')
            
            # Update button text to show file loaded
            btn_load_impedance.config(text="✓ Impedance File Loaded")
            messagebox.showinfo("Success", f"Impedance file loaded successfully!\nFile: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading impedance file: {str(e)}")
    
def load_ratings_file():
    global ratings_df
    file_path = filedialog.askopenfilename(
        title="Select Ratings CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if file_path:
        try:
            ratings_df = pd.read_csv(file_path)
            print("Ratings CSV columns found:", ratings_df.columns.tolist())
            
            # Update button text to show file loaded
            btn_load_ratings.config(text="✓ Ratings File Loaded")
            messagebox.showinfo("Success", f"Ratings file loaded successfully!\nFile: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading ratings file: {str(e)}")
            ratings_df = None

# GUI setup
root = tk.Tk()
root.title("PSSE IDV Generator")

# Callback to update conductor list based on selected kV
def update_conductors(*args):
    global impedance_df
    
    if impedance_df is None:
        print("Debug: Impedance file not loaded yet")
        return
        
    kV = kV_var.get()
    Conductors_Per_Phase = Conductors_Per_Phase_var.get()
    
    print(f"Debug: kV = '{kV}', Conductors_Per_Phase = '{Conductors_Per_Phase}'")  # Debug
    
    if not kV or not Conductors_Per_Phase:
        print("Debug: kV or Conductors_Per_Phase is empty")
        return
        
    try:
        # Debug: Show unique values in the CSV
        print(f"Debug: Unique kV values in CSV: {sorted(impedance_df['kV'].unique())}")
        print(f"Debug: Unique Conductors Per Phase values in CSV: {sorted(impedance_df['Conductors Per Phase'].unique())}")
        
        # Convert kV to int for filtering
        kV_int = int(kV)
        conductors_per_phase_int = int(Conductors_Per_Phase)
        filtered = impedance_df[(impedance_df['kV'] == kV_int) & (impedance_df['Conductors Per Phase'] == conductors_per_phase_int)]
        
        print(f"Debug: Filtered rows count: {len(filtered)}")
        if not filtered.empty:
            print(f"Debug: Sample filtered data:\n{filtered[['kV', 'Conductors Per Phase', 'Type of Conductor']].head()}")
        
        types = sorted(filtered['Type of Conductor'].unique())
        print(f"Debug: Available conductor types: {types}")
        
        conductor_menu['values'] = types
        if types:
            conductor_var.set(types[0])
            print(f"Debug: Set conductor to: {types[0]}")
        else:
            conductor_var.set('')
            conductor_menu['values'] = []
            print("Debug: No conductor types found, clearing menu")
            
    except Exception as e:
        print(f"Error in update_conductors: {e}")
        conductor_menu['values'] = []

def calculate_mva(amps, kV):
    return round((1.732 * kV * amps) / 1000, 2)

def map_conductor_name(impedance_conductor, conductors_per_phase):
    """
    Map conductor name from impedance CSV format to ratings CSV format
    """
    # Extract the main conductor info from impedance format
    # Example: "795.0  26/7  ACSR    DRAKE" -> "795 ACSR Drake"
    
    parts = impedance_conductor.strip().split()
    if len(parts) >= 4:
        # Get conductor size and map to ratings format
        size_raw = parts[0].replace('.0', '')
        
        # Map conductor sizes from impedance CSV to ratings CSV format
        size_mappings = {
            '336.4': '336',
            '336': '336',
            '477.0': '477', 
            '477': '477',
            '795.0': '795',
            '795': '795',
            '959.6': '959',
            '959': '959',
            '1433.6': '1433',
            '1433': '1433',
            '1590.0': '1590',
            '1590': '1590',
            '1926.9': '1926',
            '1926': '1926'
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
            'LINNET': 'Linnet',
            'DRAKE': 'Drake', 
            'HAWK': 'Hawk',
            'SUWANNEE': 'Suwannee',
            'MERRIMACK': 'Merimack',  # Note: different spelling in ratings file
            'CUMBERLAND': 'Cumberland',
            'LAPWING': 'Falcon'  # Lapwing maps to Falcon in ratings
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

def generate_idv():
    global impedance_df, ratings_df
    
    try:
        # Check if impedance file is loaded
        if impedance_df is None:
            messagebox.showerror("Error", "Please load the Line Impedance Estimator CSV file first")
            return
            
        # Validate inputs
        if not entry_from.get() or not entry_to.get() or not entry_amps.get() or not entry_miles.get():
            messagebox.showerror("Error", "Please fill in all required fields (From Bus, To Bus, Min Rating, Length)")
            return
            
        if not kV_var.get() or not conductor_var.get():
            messagebox.showerror("Error", "Please select voltage level and conductor type")
            return
        
        from_bus = int(entry_from.get())
        to_bus = int(entry_to.get())
        amps = float(entry_amps.get())
        miles = float(entry_miles.get())
        kV = int(kV_var.get())
        Conductors_Per_Phase = int(Conductors_Per_Phase_var.get())
        ctype = conductor_var.get()

        # Filter the data and check if any matches exist
        filtered_data = impedance_df[(impedance_df['kV'] == kV) & 
                                   (impedance_df['Conductors Per Phase'] == Conductors_Per_Phase) & 
                                   (impedance_df['Type of Conductor'] == ctype)]
        
        if filtered_data.empty:
            messagebox.showerror("Error", f"No matching data found for:\n"
                               f"kV: {kV}\n"
                               f"Conductors Per Phase: {Conductors_Per_Phase}\n"
                               f"Type of Conductor: {ctype}")
            return
            
        row = filtered_data.iloc[0]
        R_per_mile, X_per_mile, B_per_mile = row['R'], row['X'], row['B']
        
        # Calculate total impedance by multiplying per-mile values by length
        R = R_per_mile * miles
        X = X_per_mile * miles
        B = B_per_mile * miles
        
        # Calculate MVA from amps and kV
        mva = calculate_mva(amps, kV)
        
        # Initialize ratings array with 12 slots (as per PSSE API)
        ratings = [0.0] * 12
        
        # Set first 3 ratings to calculated MVA
        ratings[0] = mva  # RATE1
        ratings[1] = mva  # RATE2  
        ratings[2] = mva  # RATE3
        
        # Get rating from Ratings.csv for slot 4
        if ratings_df is not None:
            try:
                # Map the conductor name to ratings CSV format
                mapped_conductor = map_conductor_name(ctype, Conductors_Per_Phase)
                print(f"Debug: Mapped conductor from '{ctype}' to '{mapped_conductor}'")
                
                # Look for the mapped conductor in the ratings CSV
                # Try exact match first, then partial matches
                rating_filtered = ratings_df[ratings_df['Conductor Size'].str.contains(f"^{mapped_conductor}$", case=False, na=False, regex=True)]
                
                if rating_filtered.empty:
                    # Try without "Bundled" keyword
                    simple_conductor = mapped_conductor.replace(" Bundled", "")
                    rating_filtered = ratings_df[ratings_df['Conductor Size'].str.contains(f"^{simple_conductor}$", case=False, na=False, regex=True)]
                    print(f"Debug: Trying without 'Bundled': '{simple_conductor}'")
                
                if rating_filtered.empty:
                    # Try partial match with just size and type
                    conductor_parts = mapped_conductor.split()
                    if len(conductor_parts) >= 3:
                        partial_search = f"{conductor_parts[0]}.*{conductor_parts[1]}.*{conductor_parts[-1]}"
                        rating_filtered = ratings_df[ratings_df['Conductor Size'].str.contains(partial_search, case=False, na=False, regex=True)]
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
                    print(f"Debug: Available conductor sizes in ratings CSV:")
                    print(ratings_df['Conductor Size'].tolist()[:10])  # Show first 10 for reference
                        
            except Exception as e:
                print(f"Debug: Error getting conductor rating: {e}, RATE4 set to 0.0")
        
        # Set 5th rating to 9999
        ratings[4] = 9999.0  # RATE5
        
        # Ratings 6-12 remain 0.0 (already initialized)
        
        print(f"Debug: Final ratings array: {ratings}")

        idv = f"BAT_BRANCH_CHNG_3,{from_bus},{to_bus},'1',,,,,,,{R},{X},{B}," + \
              "," * 17 + ",".join(map(str, ratings)) + "," * 0 + ";"

        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, idv + f"\n\nConverted MVA: {mva}")
        output_text.insert(tk.END, f"\nLength: {miles} miles")
        output_text.insert(tk.END, f"\nImpedance per mile: R={R_per_mile}, X={X_per_mile}, B={B_per_mile}")
        output_text.insert(tk.END, f"\nTotal impedance: R={R}, X={X}, B={B}")
        output_text.insert(tk.END, f"\nRatings: RATE1-3={mva} MVA, RATE4={ratings[3]}, RATE5=9999")

    except ValueError as e:
        messagebox.showerror("Error", "Please enter valid numeric values for bus numbers and amperage")
    except KeyError as e:
        messagebox.showerror("Error", f"Missing column in CSV file: {str(e)}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# UI Components

# File selection buttons
tk.Label(root, text="1. Load CSV Files", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2, pady=(5,5))

btn_load_impedance = tk.Button(root, text="Browse - Line Impedance Estimator CSV", command=load_impedance_file, bg="lightblue")
btn_load_impedance.grid(row=1, column=0, columnspan=2, pady=5, padx=5, sticky="ew")

btn_load_ratings = tk.Button(root, text="Browse - Ratings CSV (Optional)", command=load_ratings_file, bg="lightgreen")
btn_load_ratings.grid(row=2, column=0, columnspan=2, pady=5, padx=5, sticky="ew")

# Separator
tk.Label(root, text="2. Enter Data", font=("Arial", 10, "bold")).grid(row=3, column=0, columnspan=2, pady=(15,5))

tk.Label(root, text="From Bus").grid(row=4, column=0, sticky="w", padx=5)
entry_from = tk.Entry(root)
entry_from.grid(row=4, column=1, padx=5, pady=2)

tk.Label(root, text="To Bus").grid(row=5, column=0, sticky="w", padx=5)
entry_to = tk.Entry(root)
entry_to.grid(row=5, column=1, padx=5, pady=2)

tk.Label(root, text="Min Rating (Amps)").grid(row=6, column=0, sticky="w", padx=5)
entry_amps = tk.Entry(root)
entry_amps.grid(row=6, column=1, padx=5, pady=2)

tk.Label(root, text="Length (Miles)").grid(row=7, column=0, sticky="w", padx=5)
entry_miles = tk.Entry(root)
entry_miles.grid(row=7, column=1, padx=5, pady=2)

tk.Label(root, text="Voltage Level (kV)").grid(row=8, column=0, sticky="w", padx=5)
kV_var = tk.StringVar()
kV_menu = ttk.Combobox(root, textvariable=kV_var, values=[])
kV_menu.grid(row=8, column=1, padx=5, pady=2)
kV_var.trace('w', update_conductors)

tk.Label(root, text="Conductors Per Phase").grid(row=9, column=0, sticky="w", padx=5)
Conductors_Per_Phase_var = tk.StringVar(value='1')
Conductors_Per_Phase_menu = ttk.Combobox(root, textvariable=Conductors_Per_Phase_var, values=['1', '2'])
Conductors_Per_Phase_menu.grid(row=9, column=1, padx=5, pady=2)
Conductors_Per_Phase_var.trace('w', update_conductors)

tk.Label(root, text="Conductor Type").grid(row=10, column=0, sticky="w", padx=5)
conductor_var = tk.StringVar()
conductor_menu = ttk.Combobox(root, textvariable=conductor_var, values=[])
conductor_menu.grid(row=10, column=1, padx=5, pady=2)

# Generate Button and Output Box
btn_generate = tk.Button(root, text="Generate IDV", command=generate_idv, bg="orange", font=("Arial", 10, "bold"))
btn_generate.grid(row=11, column=0, columnspan=2, pady=15)

output_text = tk.Text(root, height=8, width=120)
output_text.grid(row=12, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()