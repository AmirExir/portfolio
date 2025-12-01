"""
Create separate graph objects for each contingency scenario.
This script reads bus_scenarios.csv and edge_scenarios.csv and creates
200 separate PyG Data objects (one per scenario) saved as graph_scenarios.pt
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def voltage_to_class(v):
    """Convert voltage to 5-class classification"""
    if v < 0.95:
        return 0  # low
    elif 0.95 <= v < 0.98:
        return 1  # slightly low
    elif 0.98 <= v < 1.00:
        return 2  # near nominal
    elif 1.00 <= v < 1.02:
        return 3  # slightly high
    else:
        return 4  # high

def create_graph_for_scenario(bus_sub, edge_sub, scaler=None):
    """
    Create a PyG Data object for a single scenario.
    
    Args:
        bus_sub: DataFrame with bus data for one scenario
        edge_sub: DataFrame with edge data for one scenario
        scaler: Optional pre-fitted StandardScaler
    
    Returns:
        PyG Data object
    """
    # Map bus names to indices (0 to N-1)
    bus_to_idx = {bus: idx for idx, bus in enumerate(bus_sub['bus'].unique())}
    
    # Build edge index
    edge_list = []
    for _, row in edge_sub.iterrows():
        from_bus = row['from_bus']
        to_bus = row['to_bus']
        if from_bus in bus_to_idx and to_bus in bus_to_idx:
            edge_list.append([bus_to_idx[from_bus], bus_to_idx[to_bus]])
    
    if len(edge_list) == 0:
        # If no valid edges, create self-loops for all nodes
        num_nodes = len(bus_to_idx)
        edge_index = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long).t()
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    # Compute neighbor count for each bus
    neighbor_counts = {}
    for bus in bus_sub['bus']:
        from_count = len(edge_sub[edge_sub['from_bus'] == bus])
        to_count = len(edge_sub[edge_sub['to_bus'] == bus])
        neighbor_counts[bus] = from_count + to_count
    
    # Build feature matrix (4 features: voltage, load_MW, p_inj_mw, neighbor_count)
    features = []
    labels = []
    
    for bus in bus_sub['bus'].unique():
        bus_data = bus_sub[bus_sub['bus'] == bus].iloc[0]
        
        # Extract features
        voltage = bus_data.get('voltage', 1.0)
        load_MW = bus_data.get('load_MW', 0.0)
        p_inj_mw = bus_data.get('p_inj_mw', 0.0)
        neighbor_count = neighbor_counts.get(bus, 0)
        
        features.append([voltage, load_MW, p_inj_mw, neighbor_count])
        
        # Create label (voltage classification)
        labels.append(voltage_to_class(voltage))
    
    X = np.array(features, dtype=float)
    y = np.array(labels, dtype=int)
    
    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)
    
    # Normalize features
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    # Create PyG Data object
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data, scaler

def main():
    print("Loading CSV files...")
    bus_df = pd.read_csv('bus_scenarios.csv')
    edge_df = pd.read_csv('edge_scenarios.csv')
    
    print(f"Loaded {len(bus_df)} bus rows, {len(edge_df)} edge rows")
    
    # Get unique scenarios
    scenarios = sorted(bus_df['scenario'].unique())
    print(f"Found {len(scenarios)} unique scenarios")
    
    # Fit global scaler on all data
    print("Fitting global scaler...")
    all_features = []
    for scenario in scenarios:
        bus_sub = bus_df[bus_df['scenario'] == scenario]
        edge_sub = edge_df[edge_df['scenario'] == scenario]
        
        neighbor_counts = {}
        for bus in bus_sub['bus']:
            from_count = len(edge_sub[edge_sub['from_bus'] == bus])
            to_count = len(edge_sub[edge_sub['to_bus'] == bus])
            neighbor_counts[bus] = from_count + to_count
        
        for bus in bus_sub['bus'].unique():
            bus_data = bus_sub[bus_sub['bus'] == bus].iloc[0]
            voltage = bus_data.get('voltage', 1.0)
            load_MW = bus_data.get('load_MW', 0.0)
            p_inj_mw = bus_data.get('p_inj_mw', 0.0)
            neighbor_count = neighbor_counts.get(bus, 0)
            all_features.append([voltage, load_MW, p_inj_mw, neighbor_count])
    
    X_all = np.array(all_features, dtype=float)
    X_all = np.nan_to_num(X_all, nan=0.0)
    global_scaler = StandardScaler().fit(X_all)
    
    print("Creating graph objects for each scenario...")
    graph_list = []
    
    for i, scenario in enumerate(scenarios):
        bus_sub = bus_df[bus_df['scenario'] == scenario]
        edge_sub = edge_df[edge_df['scenario'] == scenario]
        
        data, _ = create_graph_for_scenario(bus_sub, edge_sub, scaler=global_scaler)
        graph_list.append(data)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(scenarios)} scenarios")
    
    print(f"\nCreated {len(graph_list)} graph objects")
    print(f"  Each graph: {graph_list[0].num_nodes} nodes, {graph_list[0].num_features} features")
    
    # Check class distribution
    all_labels = torch.cat([g.y for g in graph_list])
    unique, counts = torch.unique(all_labels, return_counts=True)
    print(f"  Overall class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    
    # Save to file
    print("\nSaving to graph_scenarios.pt...")
    torch.save(graph_list, 'graph_scenarios.pt')
    print("Done!")

if __name__ == "__main__":
    main()
