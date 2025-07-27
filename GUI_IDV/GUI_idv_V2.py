import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

# Load conductor impedance and rating data from CSV
impedance_df = pd.read_csv("_Line Impedance Estimator.csv")  # CSV must contain 'kV', 'Type', 'Conductors Per Phase', 'R', 'X', 'B', 'Rate1', ...

# GUI setup
root = tk.Tk()
root.title("PSSE IDV Generator")

# Callback to update conductor list based on selected kV
def update_conductors(*args):
    kV = kV_var.get()
    Conductors_Per_Phase = Conductors_Per_Phase_var.get()
    filtered = impedance_df[(impedance_df['kV'] == int(kV)) & (impedance_df['Conductors Per Phase'] == Conductors_Per_Phase)]
    types = sorted(filtered['Type'].unique())
    conductor_menu['values'] = types
    if types:
        conductor_var.set(types[0])

def calculate_mva(amps, kV):
    return round((1.732 * kV * amps) / 1000, 2)

def generate_idv():
    try:
        from_bus = int(entry_from.get())
        to_bus = int(entry_to.get())
        amps = float(entry_amps.get())
        kV = int(kV_var.get())
        Conductors_Per_Phase = Conductors_Per_Phase_var.get()
        ctype = conductor_var.get()

        row = impedance_df[(impedance_df['kV'] == kV) & (impedance_df['Conductors Per Phase'] == Conductors_Per_Phase) & (impedance_df['Type'] == ctype)].iloc[0]
        R, X, B = row['R'], row['X'], row['B']
        ratings = [row.get(f'Rate{i}', 0.0) for i in range(1, 6)]

        idv = f"BAT_BRANCH_CHNG_3,{from_bus},{to_bus},'1',,,,,,,{R},{X},{B}," + \
              "," * 17 + ",".join(map(str, ratings)) + "," + "," * 20 + ";"

        mva = calculate_mva(amps, kV)
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, idv + f"\n\nConverted MVA: {mva}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

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
Conductors_Per_Phase_var = tk.StringVar(value='SINGLE ckt')
Conductors_Per_Phase_menu = ttk.Combobox(root, textvariable=Conductors_Per_Phase_var, values=['SINGLE ckt', 'DBL ckt'])
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