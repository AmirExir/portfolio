# Selective Merge Guide: gnn-dataset-integration → main

## Current Situation
- **Current branch**: `gnn-dataset-integration`
- **Target branch**: `main`
- **Goal**: Merge only GNN-related changes, avoid unwanted files

---

## ✅ Files to INCLUDE in Merge (Production-Ready GNN System)

### Core Training Files:
```
GNN/gnn_clean.py                           # Main training script (338 lines)
GNN/create_graph_dataset.py                # Voltage dataset generator
GNN/create_graph_dataset_thermal.py        # Thermal dataset generator
```

### Notebooks:
```
GNN/GNN_Results_Summary.ipynb              # Complete training + analysis notebook
GNN/PowerGraph_Training.ipynb              # Training experiments
```

### Streamlit App:
```
GNN/app_gnn_streamlit.py                   # Professional UI (emoji-free, 495 lines)
```

### Datasets:
```
GNN/graph_scenarios.pt                     # Voltage graphs (1.8 MB)
GNN/graph_scenarios_thermal.pt             # Thermal graphs (17 MB)
GNN/bus_scenarios.csv                      # Bus data (37,350 rows)
GNN/edge_scenarios.csv                     # Edge data (83,435 rows)
GNN/bus_inputs.csv                         # Input features
GNN/edge_inputs.csv                        # Input features
```

---

## ❌ Files to EXCLUDE (Temporary/Redundant)

### Don't merge these:
```
GNN/__pycache__/                           # Python cache (auto-generated)
GNN/core_gnn copy.py                       # Backup file
GNN/app_streamlit_gnn_powergrid1.py        # Old version
GNN/gnn_seperate_old.py                    # Old version
GNN/models/                                # Model checkpoints (can regenerate)
GNN/powergraph_data.tar.gz                 # Archive (redundant)
```

### Outside GNN directory (be careful):
```
ERCOTAPI/.DS_Store                         # Mac system file
.DS_Store                                  # Mac system file
```

---

## 🚀 Step-by-Step Merge Process

### Step 1: Checkout main and create backup
```bash
cd /Users/amirexir/Documents/GitHub/portfolio

# Create a backup branch (safety first!)
git checkout main
git checkout -b main-backup

# Go back to main
git checkout main
```

### Step 2: Option A - Selective File Merge (Safest)
```bash
# Merge only specific files from gnn-dataset-integration
git checkout gnn-dataset-integration -- GNN/gnn_clean.py
git checkout gnn-dataset-integration -- GNN/create_graph_dataset.py
git checkout gnn-dataset-integration -- GNN/create_graph_dataset_thermal.py
git checkout gnn-dataset-integration -- GNN/GNN_Results_Summary.ipynb
git checkout gnn-dataset-integration -- GNN/PowerGraph_Training.ipynb
git checkout gnn-dataset-integration -- GNN/app_gnn_streamlit.py
git checkout gnn-dataset-integration -- GNN/graph_scenarios.pt
git checkout gnn-dataset-integration -- GNN/graph_scenarios_thermal.pt
git checkout gnn-dataset-integration -- GNN/bus_scenarios.csv
git checkout gnn-dataset-integration -- GNN/edge_scenarios.csv
git checkout gnn-dataset-integration -- GNN/bus_inputs.csv
git checkout gnn-dataset-integration -- GNN/edge_inputs.csv

# Now commit
git add GNN/
git commit -m "Add comprehensive GNN system with 4 architectures and dual classification modes

- Add gnn_clean.py: Unified training script for GCN/GAT/GIN/Transformer
- Add dataset generators for voltage and thermal classification
- Add GNN_Results_Summary.ipynb: Complete training + analysis notebook
- Add app_gnn_streamlit.py: Professional Streamlit interface (emoji-free)
- Add preprocessed datasets (voltage + thermal, 200 scenarios each)
- Transformer achieves 94.53% voltage accuracy and 96.51% thermal accuracy"
```

### Step 3: Option B - Full Merge with Exclusions (Advanced)
```bash
# Merge the branch but don't commit yet
git merge --no-commit --no-ff gnn-dataset-integration

# Remove unwanted files before committing
git reset HEAD GNN/__pycache__/
git reset HEAD GNN/core_gnn\ copy.py
git reset HEAD GNN/app_streamlit_gnn_powergrid1.py
git reset HEAD GNN/gnn_seperate_old.py
git reset HEAD GNN/models/
git reset HEAD GNN/powergraph_data.tar.gz
git reset HEAD ERCOTAPI/.DS_Store
git reset HEAD .DS_Store

# Remove them from working directory too
git checkout HEAD -- GNN/__pycache__/
git checkout HEAD -- "GNN/core_gnn copy.py"
git checkout HEAD -- GNN/app_streamlit_gnn_powergrid1.py
git checkout HEAD -- GNN/gnn_seperate_old.py
git checkout HEAD -- GNN/models/
git checkout HEAD -- GNN/powergraph_data.tar.gz
git checkout HEAD -- ERCOTAPI/.DS_Store
git checkout HEAD -- .DS_Store

# Now commit the merge
git commit -m "Merge gnn-dataset-integration: Add comprehensive GNN system"
```

### Step 4: Verify the merge
```bash
# Check what was actually merged
git log --oneline -5

# See files in GNN directory
ls -la GNN/

# Check for unwanted files
git status
```

### Step 5: Push to remote
```bash
# Push main branch
git push origin main

# Optionally, push backup too
git push origin main-backup
```

---

## 🔍 Quick Verification Checklist

After merge, verify these files exist in main:
- [ ] `GNN/gnn_clean.py` - Main training script
- [ ] `GNN/app_gnn_streamlit.py` - Streamlit UI
- [ ] `GNN/GNN_Results_Summary.ipynb` - Complete notebook
- [ ] `GNN/graph_scenarios.pt` - Voltage dataset
- [ ] `GNN/graph_scenarios_thermal.pt` - Thermal dataset
- [ ] `GNN/create_graph_dataset.py` - Dataset generator
- [ ] `GNN/create_graph_dataset_thermal.py` - Thermal generator

And verify these DON'T exist:
- [ ] No `GNN/__pycache__/` directory
- [ ] No `GNN/core_gnn copy.py`
- [ ] No `.DS_Store` files
- [ ] No `models/` checkpoints (unless you want them)

---

## 🔄 If Something Goes Wrong

### Undo the merge:
```bash
# If you haven't pushed yet
git reset --hard HEAD~1

# Or go back to backup
git checkout main-backup
git branch -D main
git checkout -b main
```

### Start over:
```bash
git checkout main
git reset --hard origin/main
# Then try again following the steps above
```

---

## 📝 Recommended Approach

**I recommend Option A (Selective File Merge)** because:
1. ✅ Complete control over what gets merged
2. ✅ No risk of accidentally including unwanted files
3. ✅ Clean commit history
4. ✅ Easy to understand what changed
5. ✅ Can review each file before committing

Just run the commands in Step 2, Option A!

---

## 🎯 Final Notes

- **Binary files** (`.pt` files) are ~19 MB total - make sure you want to commit them
- Consider adding to `.gitignore`: `__pycache__/`, `*.pyc`, `.DS_Store`, `models/*.pt`
- The merge adds **~200K lines** (mostly CSV data) to your repo
- Transformer model is the star performer: 94.53% voltage, 96.51% thermal! 🏆
