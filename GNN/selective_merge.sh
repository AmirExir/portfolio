#!/bin/bash
# Selective merge script for GNN system
# This merges only the production-ready files from gnn-dataset-integration to main

set -e  # Exit on error

echo "🚀 Starting selective merge of GNN system..."
echo ""

# Ensure we're on main branch
if [ "$(git branch --show-current)" != "main" ]; then
    echo "❌ Error: Must be on main branch"
    echo "Run: git checkout main"
    exit 1
fi

echo "✓ On main branch"
echo ""

# Core training files
echo "📥 Merging core training files..."
git checkout gnn-dataset-integration -- GNN/gnn_clean.py
git checkout gnn-dataset-integration -- GNN/create_graph_dataset.py
git checkout gnn-dataset-integration -- GNN/create_graph_dataset_thermal.py
echo "✓ Core files merged"
echo ""

# Notebooks
echo "📥 Merging notebooks..."
git checkout gnn-dataset-integration -- GNN/GNN_Results_Summary.ipynb
git checkout gnn-dataset-integration -- GNN/PowerGraph_Training.ipynb
echo "✓ Notebooks merged"
echo ""

# Streamlit app
echo "📥 Merging Streamlit app..."
git checkout gnn-dataset-integration -- GNN/app_gnn_streamlit.py
echo "✓ Streamlit app merged"
echo ""

# Datasets
echo "📥 Merging datasets..."
git checkout gnn-dataset-integration -- GNN/graph_scenarios.pt
git checkout gnn-dataset-integration -- GNN/graph_scenarios_thermal.pt
git checkout gnn-dataset-integration -- GNN/bus_scenarios.csv
git checkout gnn-dataset-integration -- GNN/edge_scenarios.csv
git checkout gnn-dataset-integration -- GNN/bus_inputs.csv
git checkout gnn-dataset-integration -- GNN/edge_inputs.csv
echo "✓ Datasets merged"
echo ""

# Show what will be committed
echo "📋 Files staged for commit:"
git status --short
echo ""

# Ask for confirmation
read -p "🤔 Commit these changes? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    git add GNN/
    git commit -m "Add comprehensive GNN system with 4 architectures and dual classification modes

Features:
- Add gnn_clean.py: Unified training script for GCN/GAT/GIN/Transformer
- Add dataset generators for voltage and thermal classification
- Add GNN_Results_Summary.ipynb: Complete training + analysis notebook
- Add PowerGraph_Training.ipynb: Training experiments
- Add app_gnn_streamlit.py: Professional Streamlit interface (emoji-free)
- Add preprocessed datasets (voltage + thermal, 200 scenarios each)

Performance:
- Voltage Classification: Transformer achieves 94.53% accuracy
- Thermal Classification: Transformer achieves 96.51% accuracy
- Best model on both tasks with high macro F1 scores

Architecture:
- 4 GNN models: GCN (baseline), GAT (attention), GIN (isomorphism), Transformer
- 2 classification modes: Bus-level voltage, Line-level thermal
- Complete training pipeline with DataLoader and class weights
- Professional Streamlit interface for demonstrations"
    
    echo ""
    echo "✅ Merge committed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Review: git log --oneline -3"
    echo "2. Push: git push origin main"
    echo "3. If needed, restore backup: git checkout main-backup"
else
    echo ""
    echo "❌ Merge cancelled"
    echo "To undo staged changes: git reset HEAD GNN/"
fi
