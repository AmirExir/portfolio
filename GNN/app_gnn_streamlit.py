"""Power Grid GNN: exploratory source-class training and topology inspection."""
import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path
import sys
import json

# Add parent directory to path to import from gnn_clean
sys.path.append(str(Path(__file__).parent))
from gnn_clean import train_gnn_multi_graph
from data_pipeline import load_csv_graphs, SOURCE_NOTE

# Page configuration
st.set_page_config(
    page_title="Power Grid GNN",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Power Grid GNN")
st.write("Train graph models, explore grid scenarios, and compare results on held-out scenarios.")
st.caption("By Amir Exir · Research demo using supplied class labels; physical units and label thresholds are unverified.")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Mode selection
    mode = st.selectbox(
        "Classification Mode",
        ["voltage", "thermal"],
        help="Voltage: bus-level classification, Thermal: line-level classification"
    )
    
    # Model selection
    model_type = st.selectbox(
        "GNN Architecture",
        ["gcn", "gat", "gin", "transformer"],
        format_func=lambda x: x.upper(),
        help="Choose the graph neural network architecture"
    )
    
    st.divider()
    
    # Training parameters
    st.subheader("Training Parameters")
    epochs = st.slider("Epochs", 10, 200, 50, 10)
    lr = st.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3, format_func=lambda x: f"{x:.0e}")
    weight_decay = st.select_slider("Weight Decay", options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3], value=5e-4, format_func=lambda x: f"{x:.0e}")
    batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=32)
    use_relu = st.checkbox("Use ReLU Activation", value=True)
    seed = st.number_input("Random Seed", value=42, min_value=0)
    
    st.divider()
    
    # Visualization parameters
    st.subheader("Visualization Settings")
    show_edge_labels = st.checkbox("Show Edge Labels", value=False)
    node_size = st.slider("Node Size", 5, 30, 15)

# Load data
@st.cache_data
def load_dataset(mode, source_revision):
    """Cache inspectable CSV graphs by source file modification/size signature."""
    try:
        return load_csv_graphs(mode), None
    except (OSError, ValueError, KeyError) as exc:
        return None, f"Cannot load scenario CSVs: {exc}"

# Train model
def train_model(data, model_type, epochs, lr, weight_decay, seed, use_relu, batch_size):
    """Train the GNN model with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Training in progress...")
    
    model, hist_df = train_gnn_multi_graph(
        data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        use_relu=use_relu,
        batch_size=batch_size,
        model_type=model_type,
        progress_callback=lambda epoch, total: (progress_bar.progress(epoch / total), status_text.text(f"Epoch {epoch}/{total}"))
    )
    
    progress_bar.progress(100)
    status_text.text("Training complete!")
    
    return model, hist_df

# Visualize graph
def visualize_graph(graph_data, scenario_id, mode, show_edge_labels=False, node_size=15):
    """Create interactive graph visualization using Plotly"""
    
    # Convert PyG graph to NetworkX
    edge_index = graph_data.edge_index.cpu().numpy()
    node_features = graph_data.x.cpu().numpy()
    node_labels = graph_data.y.cpu().numpy()
    
    G = nx.Graph()
    G.add_nodes_from(range(graph_data.num_nodes))
    edges = [(int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1])]
    G.add_edges_from(edges)
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    
    # Edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    # Color nodes by their class
    node_colors = node_labels
    
    if mode == "voltage":
        class_names = [f"Source class {i}" for i in range(5)]
        colorscale = 'RdYlGn_r'  # Red for high, green for low
    else:
        class_names = [f"Source class {i}" for i in range(4)]
        colorscale = 'Reds'
    
    node_text = []
    for i, node in enumerate(G.nodes()):
        features = "<br>".join(f"{name}: {value:.3f} (source value)" for name, value in zip(graph_data.feature_names, node_features[i]))
        text = f"{'Bus' if mode == 'voltage' else 'Line'} {graph_data.entity_ids[i]}<br>{features}<br>Class: {int(node_labels[i])}"
        node_text.append(text)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale=colorscale,
            size=node_size,
            color=node_colors,
            cmin=0, cmax=len(class_names) - 1,
            colorbar=dict(
                thickness=15,
                title=dict(text="Class", side='right'),
                xanchor='left',
                tickmode='array',
                tickvals=list(range(len(class_names))),
                ticktext=class_names
            ),
            line=dict(width=1, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=dict(
                           text=f"Scenario {scenario_id} - {mode.title()} Graph ({graph_data.num_nodes} nodes)",
                           font=dict(size=16)
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       annotations=[dict(
                           text=f"Graph topology for contingency scenario {scenario_id}",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=600
                   ))
    
    if show_edge_labels:
        for start, end in G.edges():
            fig.add_annotation(x=(pos[start][0] + pos[end][0]) / 2,
                               y=(pos[start][1] + pos[end][1]) / 2,
                               text=f"{start}–{end}", showarrow=False, font=dict(size=9))
    return fig

# Load dataset
source_dir = Path(__file__).resolve().parent
source_paths = [source_dir / "edge_scenarios.csv"]
if mode == "voltage":
    source_paths.append(source_dir / "bus_scenarios.csv")
source_revision = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in source_paths if path.exists())
data, error = load_dataset(mode, source_revision)

if error:
    st.error(f"{error}")
    st.info("Check the source CSV schema and required class/status columns. See GNN/README.md.")
    st.stop()

scenario_ids = [graph.scenario_id for graph in data]
scenario_id = st.sidebar.selectbox("Scenario ID", scenario_ids)
scenario_to_view = scenario_ids.index(scenario_id)
configuration = (mode, model_type, epochs, lr, weight_decay, seed, use_relu, batch_size, source_revision)
if st.session_state.get("training_configuration") != configuration:
    for state_key in ("model", "hist_df", "trained"):
        st.session_state.pop(state_key, None)

if mode == "thermal":
    st.warning("Thermal demo: scenario dispatch and load are absent from the predictors, limiting what the model can learn.")

# Dataset info
with st.expander("Data and research assumptions", expanded=False):
    st.write(SOURCE_NOTE)
    st.caption("Supplied class IDs are preserved. They are not verified voltage or thermal compliance criteria.")
    if mode == "thermal":
        st.write("Thermal predictors contain line properties and topology only. Identical predictors can have different thermal classes, so this is an exploratory baseline.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Graphs", len(data))
    with col2:
        st.metric("Nodes per Graph", data[0].num_nodes)
    with col3:
        st.metric("Features per Node", data[0].num_features)
    with col4:
        all_labels = torch.cat([g.y for g in data])
        unique_classes = len(torch.unique(all_labels))
        st.metric("Classes", unique_classes)
    
    st.caption(f"Open branch rows excluded from active topology: {sum(g.excluded_open_branches for g in data)}. Bidirectional message edges are used.")
    # Class distribution
    all_labels = torch.cat([g.y for g in data])
    unique, counts = torch.unique(all_labels, return_counts=True)
    class_dist = pd.DataFrame({
        'Class': unique.tolist(),
        'Count': counts.tolist()
    })
    st.dataframe(class_dist, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Train & compare", "Explore topology", "Evaluation"])

# Tab 1: Training
with tab1:
    st.header("Train a graph model")
    st.write("Choose a model and settings in the sidebar, then start training. Validation selects the checkpoint; separate test scenarios measure its performance.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"{model_type.upper()} · {mode.title()} classes")
        config_data = {
            "Mode": mode.upper(),
            "Model": model_type.upper(),
            "Epochs": epochs,
            "Learning Rate": f"{lr:.0e}",
            "Weight Decay": f"{weight_decay:.0e}",
            "Batch Size": batch_size,
            "Activation": "ReLU" if use_relu else "None",
            "Random Seed": seed
        }
        st.caption(f"{len(data)} source scenarios · {epochs} epochs · Batch size {batch_size} · Seed {seed}")
        with st.expander("Full configuration"):
            st.dataframe(pd.DataFrame(config_data.items(), columns=["Setting", "Value"]).astype(str),
                         hide_index=True, use_container_width=True)
    
    with col2:
        if st.button("Start Training", type="primary", use_container_width=True):
            # A new attempt supersedes the previous run, including when it fails.
            for state_key in ("model", "hist_df", "trained", "training_configuration"):
                st.session_state.pop(state_key, None)
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    model, hist_df = train_model(
                        data, model_type, epochs, lr, weight_decay, seed, use_relu, batch_size
                    )
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.hist_df = hist_df
                    st.session_state.trained = True
                    st.session_state.training_configuration = configuration
                    
                    st.success("Training completed successfully!")
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

        if 'hist_df' in st.session_state:
            final = st.session_state.hist_df.attrs['evaluation']['test']
            st.caption('Held-out test results from the checkpoint selected by validation loss')
            metric_columns = st.columns(4)
            for column, (label, metric) in zip(metric_columns, (
                ("Accuracy", "accuracy"), ("Precision", "precision_weighted"),
                ("Weighted F1", "f1_weighted"), ("Macro F1", "f1_macro"),
            )):
                column.metric(label, f"{final[metric]:.2%}")
        else:
            st.caption("Results will include a majority-class baseline, per-class scores, and a downloadable evaluation report.")
    
    # Display training history if available
    if 'hist_df' in st.session_state:
        st.divider()
        st.subheader("Training History")
        
        hist_df = st.session_state.hist_df
        
        # Plot metrics
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['train_loss'], 
                                        mode='lines', name='Train Loss', line=dict(color='blue')))
        fig_metrics.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['val_loss'], 
                                        mode='lines', name='Val Loss', line=dict(color='red')))
        fig_metrics.update_layout(title='Loss Over Epochs', xaxis_title='Epoch', yaxis_title='Loss', height=400)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['val_acc'], 
                                        mode='lines', name='Accuracy', line=dict(color='green')))
            fig_acc.update_layout(title='Validation Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy', height=300)
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['val_f1'], 
                                       mode='lines', name='F1 Score', line=dict(color='purple')))
            fig_f1.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['val_f1_macro'], 
                                       mode='lines', name='Macro F1', line=dict(color='orange')))
            fig_f1.update_layout(title='F1 Scores', xaxis_title='Epoch', yaxis_title='F1 Score', height=300)
            st.plotly_chart(fig_f1, use_container_width=True)
        
        # Show data table
        with st.expander("View Training Data"):
            st.dataframe(hist_df, use_container_width=True)

# Tab 2: Graph Visualization
with tab2:
    st.header("Explore a scenario")
    st.caption("Select a scenario in the sidebar. Hover over a bus or line to inspect its source features and class.")
    
    if scenario_to_view >= len(data):
        st.error(f"Scenario {scenario_to_view} does not exist. Valid range: 0-{len(data)-1}")
    else:
        graph_data = data[scenario_to_view]
        
        # Display graph info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", graph_data.num_nodes)
        with col2:
            st.metric("Edges", graph_data.edge_index.shape[1])
        with col3:
            st.metric("Features", graph_data.num_features)
        with col4:
            unique_labels = len(torch.unique(graph_data.y))
            st.metric("Unique Classes", unique_labels)
        
        # Visualize graph
        fig = visualize_graph(graph_data, scenario_id, mode, show_edge_labels, node_size)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show node statistics
        with st.expander("Node Statistics"):
            node_features = graph_data.x.cpu().numpy()
            node_labels = graph_data.y.cpu().numpy()
            
            feature_names = graph_data.feature_names
            st.caption("Source feature values: original physical units cannot be recovered from the bundled preprocessing.")

            stats_data = []
            for i, name in enumerate(feature_names[:node_features.shape[1]]):
                stats_data.append({
                    'Feature': name,
                    'Mean': f"{node_features[:, i].mean():.3f}",
                    'Std': f"{node_features[:, i].std():.3f}",
                    'Min': f"{node_features[:, i].min():.3f}",
                    'Max': f"{node_features[:, i].max():.3f}"
                })
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
            
            # Class distribution for this scenario
            unique, counts = torch.unique(graph_data.y, return_counts=True)
            class_dist = pd.DataFrame({
                'Class': unique.tolist(),
                'Count': counts.tolist()
            })
            st.subheader("Class Distribution")
            st.bar_chart(class_dist.set_index('Class'))

# Tab 3: Performance Analysis
with tab3:
    st.header("Held-out Scenario Evaluation")
    if 'hist_df' not in st.session_state:
        st.info("Start a run in Train & compare to view held-out results and the majority-class baseline.")
    else:
        hist_df = st.session_state.hist_df
        report = hist_df.attrs["evaluation"]
        test = report["test"]
        baseline = report["majority_baseline"]
        st.caption(f"Selected epoch {report['best_epoch']} by validation loss. Test scenarios were excluded from training, scaling, and checkpoint selection.")
        st.dataframe(pd.DataFrame([
            {"Model": "Selected GNN", "Test accuracy": test["accuracy"], "Test macro F1": test["f1_macro"]},
            {"Model": f"Training majority class ({baseline['class']})", "Test accuracy": baseline["accuracy"], "Test macro F1": baseline["f1_macro"]},
        ]), hide_index=True, use_container_width=True)
        st.write({name: len(ids) for name, ids in report["splits"].items()})
        if report["missing_training_classes"]:
            st.warning(f"Classes absent from training: {report['missing_training_classes']}")
        st.subheader("Per-class Test Metrics")
        rows = {label: values for label, values in test["per_class"].items() if label.isdigit()}
        st.dataframe(pd.DataFrame(rows).T, use_container_width=True)
        st.subheader("Test Confusion Matrix")
        matrix = test["confusion_matrix"]
        st.dataframe(pd.DataFrame(matrix, index=[f"Actual {i}" for i in range(len(matrix))],
                                   columns=[f"Predicted {i}" for i in range(len(matrix))]), use_container_width=True)
        st.subheader("Validation Metrics Over Epochs")
        metrics_to_plot = st.multiselect("Select metrics to plot",
            ['val_acc', 'val_prec', 'val_rec', 'val_f1', 'val_f1_macro'], default=['val_acc', 'val_f1_macro'])
        if metrics_to_plot:
            fig = go.Figure()
            for metric in metrics_to_plot:
                fig.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df[metric], mode='lines', name=metric))
            fig.update_layout(xaxis_title='Epoch', yaxis_title='Score', height=400)
            st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download evaluation report", json.dumps(report, indent=2),
                           file_name=f"gnn_{mode}_evaluation.json", mime="application/json")
        st.caption("Reusing this test set for architecture or parameter selection turns it into validation data; an independent external dataset is needed for a final generalization claim.")

# Footer
st.divider()
st.caption("Power Grid GNN · Amir Exir · GCN, GAT, GIN, and Transformer models")
