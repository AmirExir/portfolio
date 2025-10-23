import argparse
import os
import pandas as pd
import torch
from powergrid import PowerGrid, get_dataloader
from gengraph import build_graphs

def main():
    parser = argparse.ArgumentParser(description="Generate PowerGrid dataset and process graphs.")
    parser.add_argument("--root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--type", type=str, choices=["binary", "multiclass", "regression"], default="multiclass", help="Type of prediction task")
    args = parser.parse_args()

    # Initialize PowerGrid dataset
    dataset = PowerGrid(root=args.root, name=args.name, task_type=args.type)
    print(f"Loaded dataset '{args.name}' with {len(dataset)} graphs.")

    # Build PyG Data objects from dataset
    data_list = build_graphs(dataset)
    print(f"Built {len(data_list)} graph objects.")

    # Save PyG dataset
    save_path = os.path.join(args.root, "graph_scenarios.pt")
    torch.save(data_list, save_path)
    print(f"Saved PyG dataset to {save_path}")

    # Extract and save node and edge features to CSV for inspection
    all_nodes = []
    all_edges = []
    for i, data in enumerate(data_list):
        # Nodes
        node_df = pd.DataFrame(data.x.numpy())
        node_df["graph_id"] = i
        all_nodes.append(node_df)

        # Edges
        edge_index = data.edge_index.numpy()
        edge_attr = data.edge_attr.numpy() if data.edge_attr is not None else None
        edge_df = pd.DataFrame({
            "from_node": edge_index[0],
            "to_node": edge_index[1]
        })
        if edge_attr is not None:
            for j in range(edge_attr.shape[1]):
                edge_df[f"edge_attr_{j}"] = edge_attr[:, j]
        edge_df["graph_id"] = i
        all_edges.append(edge_df)

    nodes_df = pd.concat(all_nodes, ignore_index=True)
    edges_df = pd.concat(all_edges, ignore_index=True)

    nodes_csv_path = os.path.join(args.root, "nodes_summary.csv")
    edges_csv_path = os.path.join(args.root, "edges_summary.csv")
    nodes_df.to_csv(nodes_csv_path, index=False)
    edges_df.to_csv(edges_csv_path, index=False)
    print(f"Saved node features summary to {nodes_csv_path}")
    print(f"Saved edge features summary to {edges_csv_path}")

    # Print feature dimensionality and class balance info
    feat_dim = data_list[0].x.shape[1] if len(data_list) > 0 else 0
    print(f"Node feature dimension: {feat_dim}")

    # Aggregate target labels
    ys = torch.cat([data.y for data in data_list], dim=0) if len(data_list) > 0 else torch.tensor([])
    if ys.numel() > 0:
        unique_classes, counts = torch.unique(ys, return_counts=True)
        print(f"Number of graphs: {len(data_list)}")
        print("Class distribution:")
        for cls, cnt in zip(unique_classes.tolist(), counts.tolist()):
            print(f"  Class {cls}: {cnt} samples")
    else:
        print("No target labels found in dataset.")

if __name__ == "__main__":
    main()