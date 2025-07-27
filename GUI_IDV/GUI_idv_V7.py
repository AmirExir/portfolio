import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

# Load conductor impedance and rating data from CSV
try:
    impedance_df = pd.read_csv("_Line Impedance Estimator.csv")
    print("CSV columns found:", impedance_df.columns.tolist())  # Debug: show available columns
except FileNotFoundError:
    messagebox.showerror("Error", "CSV file '_Line Impedance Estimator.csv' not found!")
    exit()
except Exception as e:
    messagebox.showerror("Error", f"Error loading CSV: {str(e)}")
    exit()

# GUI setup
root = tk.Tk()
root.title("PSSE IDV Generator")

# Callback to update conductor list based on selected kV
def update_conductors(*args):
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

def generate_idv():
    try:
        # Validate inputs
        if not entry_from.get() or not entry_to.get() or not entry_amps.get():
            messagebox.showerror("Error", "Please fill in all required fields (From Bus, To Bus, Min Rating)")
            return
            
        if not kV_var.get() or not conductor_var.get():
            messagebox.showerror("Error", "Please select voltage level and conductor type")
            return
        
        from_bus = int(entry_from.get())
        to_bus = int(entry_to.get())
        amps = float(entry_amps.get())
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
        R, X, B = row['R'], row['X'], row['B']
        ratings = [row.get(f'Rate{i}', 0.0) for i in range(1, 6)]

        idv = f"BAT_BRANCH_CHNG_3,{from_bus},{to_bus},'1',,,,,,,{R},{X},{B}," + \
              "," * 17 + ",".join(map(str, ratings)) + "," + "," * 20 + ";"

        mva = calculate_mva(amps, kV)
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, idv + f"\n\nConverted MVA: {mva}")

    except ValueError as e:
        messagebox.showerror("Error", "Please enter valid numeric values for bus numbers and amperage")
    except KeyError as e:
        messagebox.showerror("Error", f"Missing column in CSV file: {str(e)}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# UI Components
tk.Label(root, text="From Bus").grid(row=0, column=0)
entry_from = tk.Entry(root)
entry_from.grid(row=0, column=1)

tk.Label(root, text="To Bus").grid(row=1, column=0)
entry_to = tk.Entry(root)
entry_to.grid(row=1, column=1)

tk.Label(root, text="Min Rating (Amps)").grid(row=2, column=0)
entry_amps = tk.Entry(root)
entry_amps.grid(row=2, column=1)

tk.Label(root, text="Voltage Level (kV)").grid(row=3, column=0)
kV_var = tk.StringVar()
kV_menu = ttk.Combobox(root, textvariable=kV_var, values=sorted(impedance_df['kV'].unique().astype(str)))
kV_menu.grid(row=3, column=1)
kV_var.trace('w', update_conductors)

tk.Label(root, text="Conductors Per Phase Type").grid(row=4, column=0)
Conductors_Per_Phase_var = tk.StringVar(value='1')
Conductors_Per_Phase_menu = ttk.Combobox(root, textvariable=Conductors_Per_Phase_var, values=['1', '2'])
Conductors_Per_Phase_menu.grid(row=4, column=1)
Conductors_Per_Phase_var.trace('w', update_conductors)

tk.Label(root, text="Conductor Type").grid(row=5, column=0)
conductor_var = tk.StringVar()
conductor_menu = ttk.Combobox(root, textvariable=conductor_var)
conductor_menu.grid(row=5, column=1)

# Generate Button and Output Box
btn_generate = tk.Button(root, text="Generate IDV", command=generate_idv)
btn_generate.grid(row=6, column=0, columnspan=2, pady=10)

output_text = tk.Text(root, height=6, width=100)
output_text.grid(row=7, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()