"""
Streamlit App for Power Grid GNN Training and Visualization
Supports voltage and thermal violation detection with graph visualization
"""
import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path
import sys

# Add parent directory to path to import from gnn_clean
sys.path.append(str(Path(__file__).parent))
from gnn_clean import GCN, GAT, GIN, GraphTransformer, train_gnn_multi_graph, set_seed

# Page configuration
st.set_page_config(
    page_title="Power Grid GNN Analyzer",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">Power Grid Violation Detection with GNNs</p>', unsafe_allow_html=True)

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
    scenario_to_view = st.number_input("Scenario ID", min_value=0, max_value=199, value=0)
    show_edge_labels = st.checkbox("Show Edge Labels", value=False)
    node_size = st.slider("Node Size", 5, 30, 15)

# Load data
@st.cache_data
def load_dataset(mode):
    """Load the preprocessed graph dataset"""
    script_dir = Path(__file__).parent
    
    if mode == "thermal":
        pt_path = script_dir / "graph_scenarios_thermal.pt"
        generator_script = script_dir / "create_graph_dataset_thermal.py"
    else:
        pt_path = script_dir / "graph_scenarios.pt"
        generator_script = script_dir / "create_graph_dataset.py"
    
    # Auto-generate dataset if missing
    if not pt_path.exists():
        st.warning(f"Dataset '{pt_path}' not found. Generating now... This may take 1-2 minutes.")
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, str(generator_script)],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(script_dir)
            )
            if result.returncode != 0:
                return None, f"Failed to generate dataset: {result.stderr}"
            st.success(f"✅ Dataset generated successfully!")
        except subprocess.TimeoutExpired:
            return None, "Dataset generation timed out (>5 minutes)"
        except Exception as e:
            return None, f"Error generating dataset: {str(e)}"
    
    try:
        data = torch.load(str(pt_path), weights_only=False)
        if not isinstance(data, list):
            return None, f"Expected list of graphs, got {type(data)}"
        return data, None
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"

# Train model
def train_model(data, model_type, epochs, lr, weight_decay, seed, use_relu, batch_size):
    """Train the GNN model with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Custom callback to update progress (we'll simulate since we can't modify train function easily)
    status_text.text("Training in progress...")
    
    model, hist_df = train_gnn_multi_graph(
        data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        use_relu=use_relu,
        batch_size=batch_size,
        model_type=model_type
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
        class_names = ["Low (<0.95)", "Slightly Low [0.95-0.98)", "Nominal [0.98-1.00)", 
                      "Slightly High [1.00-1.02)", "High (≥1.02)"]
        colorscale = 'RdYlGn_r'  # Red for high, green for low
    else:
        class_names = ["Normal", "Warning", "Overload", "Critical"]
        colorscale = 'Reds'
    
    node_text = []
    for i, node in enumerate(G.nodes()):
        if mode == "voltage":
            text = f"Bus {node}<br>Voltage: {node_features[i, 0]:.3f} pu<br>Class: {class_names[node_labels[i]]}"
        else:
            text = f"Line {node}<br>Loading: {node_features[i, 2]:.1f}%<br>Class: {class_names[node_labels[i]]}"
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
    
    return fig

# Main content
tab1, tab2, tab3 = st.tabs(["Training", "Graph Visualization", "Performance Analysis"])

# Load dataset
data, error = load_dataset(mode)

if error:
    st.error(f"{error}")
    st.info("Please generate the dataset first using `create_graph_dataset.py` or `create_graph_dataset_thermal.py`")
    st.stop()

# Dataset info
with st.expander("Dataset Information", expanded=False):
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
    
    # Class distribution
    all_labels = torch.cat([g.y for g in data])
    unique, counts = torch.unique(all_labels, return_counts=True)
    class_dist = pd.DataFrame({
        'Class': unique.tolist(),
        'Count': counts.tolist()
    })
    st.dataframe(class_dist, use_container_width=True)

# Tab 1: Training
with tab1:
    st.header("Model Training")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration Summary")
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
        for key, value in config_data.items():
            st.text(f"{key}: {value}")
    
    with col2:
        if st.button("Start Training", type="primary", use_container_width=True):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    model, hist_df = train_model(
                        data, model_type, epochs, lr, weight_decay, seed, use_relu, batch_size
                    )
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.hist_df = hist_df
                    st.session_state.trained = True
                    
                    st.success("Training completed successfully!")
                    
                    # Display final metrics
                    final = hist_df.iloc[-1]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{final['val_acc']:.2%}")
                    with col2:
                        st.metric("Precision", f"{final['val_prec']:.2%}")
                    with col3:
                        st.metric("F1 Score", f"{final['val_f1']:.2%}")
                    with col4:
                        st.metric("Macro F1", f"{final['val_f1_macro']:.2%}")
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    
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
    st.header("Scenario Graph Visualization")
    
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
        fig = visualize_graph(graph_data, scenario_to_view, mode, show_edge_labels, node_size)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show node statistics
        with st.expander("Node Statistics"):
            node_features = graph_data.x.cpu().numpy()
            node_labels = graph_data.y.cpu().numpy()
            
            if mode == "voltage":
                feature_names = ["Voltage (pu)", "Load (MW)", "P Injection (MW)", "Neighbor Count"]
            else:
                feature_names = ["Reactance (pu)", "Length (km)", "Loading (%)"]
            
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
    st.header("Model Performance Analysis")
    
    if 'hist_df' not in st.session_state:
        st.info("Please train a model first in the Training tab")
    else:
        hist_df = st.session_state.hist_df
        final = hist_df.iloc[-1]
        best_epoch = hist_df.loc[hist_df['val_loss'].idxmin()]
        
        st.subheader("Best Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Final Metrics")
            st.metric("Accuracy", f"{final['val_acc']:.2%}")
            st.metric("Precision", f"{final['val_prec']:.2%}")
            st.metric("Recall", f"{final['val_rec']:.2%}")
            st.metric("F1 Score", f"{final['val_f1']:.2%}")
            st.metric("Macro F1", f"{final['val_f1_macro']:.2%}")
        
        with col2:
            st.markdown("### Best Checkpoint")
            st.metric("Best Epoch", int(best_epoch['epoch']))
            st.metric("Best Val Loss", f"{best_epoch['val_loss']:.4f}")
            st.metric("Accuracy @ Best", f"{best_epoch['val_acc']:.2%}")
        
        with col3:
            st.markdown("### Training Stats")
            st.metric("Total Epochs", len(hist_df))
            st.metric("Final Train Loss", f"{final['train_loss']:.4f}")
            st.metric("Final Val Loss", f"{final['val_loss']:.4f}")
            improvement = hist_df.iloc[0]['val_acc'] - final['val_acc']
            st.metric("Accuracy Improvement", f"{improvement:.2%}")
        
        # Detailed metrics over time
        st.divider()
        st.subheader("Metrics Evolution")
        
        metrics_to_plot = st.multiselect(
            "Select metrics to plot",
            ['val_acc', 'val_prec', 'val_rec', 'val_f1', 'val_f1_macro'],
            default=['val_acc', 'val_f1']
        )
        
        if metrics_to_plot:
            fig = go.Figure()
            for metric in metrics_to_plot:
                fig.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df[metric], 
                                        mode='lines', name=metric.replace('val_', '').upper()))
            fig.update_layout(title='Validation Metrics Over Time', 
                            xaxis_title='Epoch', yaxis_title='Score', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison
        st.divider()
        st.subheader("Model Comparison Guide")
        
        comparison_data = {
            "Model": ["GCN", "GAT", "GIN", "Transformer"],
            "Typical Accuracy": ["85-86%", "84-85%", "87-88%", "94-95%"],
            "Speed": ["Fast", "Medium", "Fast", "Slow"],
            "Best For": [
                "Baseline, fast training",
                "When node importance varies",
                "Structure-aware learning",
                "Maximum accuracy"
            ]
        }
        st.table(pd.DataFrame(comparison_data))

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Power Grid GNN Analyzer | Built with Streamlit & PyTorch Geometric</p>
    <p>Supports GCN, GAT, GIN, and Transformer architectures for voltage and thermal violation detection</p>
</div>
""", unsafe_allow_html=True)
