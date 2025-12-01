"""
Create graph_scenarios_thermal.pt for thermal classification.
Each graph represents one contingency scenario with transmission lines as nodes.
"""
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def create_thermal_graphs():
    print("Loading edge_scenarios.csv...")
    edge_df = pd.read_csv("edge_scenarios.csv")
    
    # Get unique scenarios
    scenarios = sorted(edge_df['scenario'].unique())
    print(f"Found {len(scenarios)} scenarios")
    
    # Prepare features: x_pu, length_km, loading_percent
    feature_cols = ['x_pu', 'length_km', 'loading_percent']
    
    # Check which features exist
    available_features = [c for c in feature_cols if c in edge_df.columns]
    if len(available_features) == 0:
        raise ValueError(f"No thermal features found in edge_scenarios.csv. Expected: {feature_cols}")
    
    print(f"Using features: {available_features}")
    
    # Fit global scaler on all data
    X_all = edge_df[available_features].fillna(0).to_numpy(dtype=float)
    scaler = StandardScaler()
    scaler.fit(X_all)
    
    # Check thermal_class distribution
    if 'thermal_class' in edge_df.columns:
        print(f"\nThermal class distribution:")
        print(edge_df['thermal_class'].value_counts().sort_index())
    else:
        raise ValueError("thermal_class column not found in edge_scenarios.csv")
    
    graph_list = []
    
    for scen in scenarios:
        scen_df = edge_df[edge_df['scenario'] == scen].reset_index(drop=True)
        
        # Features: each edge (line) is a node
        X = scen_df[available_features].fillna(0).to_numpy(dtype=float)
        X_scaled = scaler.transform(X)
        
        # Labels: thermal_class
        y = scen_df['thermal_class'].to_numpy(dtype=int)
        
        # Build line-graph: connect lines that share a bus
        # Each row index is a node (representing a transmission line)
        num_lines = len(scen_df)
        
        # Create mapping: bus -> list of line indices
        bus_to_lines = {}
        for idx, row in scen_df.iterrows():
            from_bus = int(row['from_bus'])
            to_bus = int(row['to_bus'])
            
            if from_bus not in bus_to_lines:
                bus_to_lines[from_bus] = []
            if to_bus not in bus_to_lines:
                bus_to_lines[to_bus] = []
            
            bus_to_lines[from_bus].append(idx)
            bus_to_lines[to_bus].append(idx)
        
        # Create edges: connect lines that share a bus
        edge_list = []
        for bus, lines in bus_to_lines.items():
            if len(lines) > 1:
                # Connect all pairs of lines at this bus
                for i, line_a in enumerate(lines):
                    for line_b in lines[i+1:]:
                        edge_list.append([line_a, line_b])
                        edge_list.append([line_b, line_a])  # Undirected
        
        if len(edge_list) == 0:
            # No edges - add self-loops to avoid issues
            edge_index = torch.tensor([[i for i in range(num_lines)],
                                      [i for i in range(num_lines)]], dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        # Create PyG Data object
        data = Data(
            x=torch.tensor(X_scaled, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.long)
        )
        
        graph_list.append(data)
    
    print(f"\nCreated {len(graph_list)} thermal graphs")
    print(f"  Example graph: {graph_list[0].num_nodes} nodes (lines), {graph_list[0].num_features} features")
    
    # Check overall class distribution
    all_labels = torch.cat([g.y for g in graph_list])
    unique, counts = torch.unique(all_labels, return_counts=True)
    print(f"  Overall class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    
    # Save
    output_path = "graph_scenarios_thermal.pt"
    torch.save(graph_list, output_path)
    print(f"\nSaved to {output_path}")
    
    return graph_list

if __name__ == "__main__":
    create_thermal_graphs()
