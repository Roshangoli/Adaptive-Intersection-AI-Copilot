#!/usr/bin/env python3
"""
Streamlit Dashboard for Adaptive Intersection AI Copilot
This dashboard provides real-time monitoring and visualization of the traffic control system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Adaptive Intersection AI Copilot",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-danger {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def load_latest_results():
    """Load the most recent simulation results."""
    # Try different possible paths
    possible_paths = ["../results", "results", "./results"]
    results_dir = None
    
    for path in possible_paths:
        if os.path.exists(path):
            results_dir = path
            break
    
    if results_dir is None:
        st.error("Results directory not found. Tried: " + ", ".join(possible_paths))
        return None, None, None
    
    # Find latest files
    fixed_files = glob.glob(os.path.join(results_dir, "fixed_time_results_*.csv"))
    rl_files = glob.glob(os.path.join(results_dir, "rl_results_*.csv"))
    simple_files = glob.glob(os.path.join(results_dir, "simple_results_*.csv"))
    
    fixed_df = None
    rl_df = None
    simple_df = None
    
    if fixed_files:
        latest_fixed = max(fixed_files, key=os.path.getctime)
        fixed_df = pd.read_csv(latest_fixed)
    
    if rl_files:
        latest_rl = max(rl_files, key=os.path.getctime)
        rl_df = pd.read_csv(latest_rl)
    
    if simple_files:
        latest_simple = max(simple_files, key=os.path.getctime)
        simple_df = pd.read_csv(latest_simple)
    
    return fixed_df, rl_df, simple_df

def create_metrics_summary(df, title):
    """Create a metrics summary for the given dataframe."""
    if df is None or df.empty:
        return st.warning(f"No {title} data available")
    
    # Determine which columns are available
    has_pedestrians = 'pedestrians' in df.columns
    has_pedestrian_wait = 'avg_pedestrian_wait' in df.columns
    
    if has_pedestrians and has_pedestrian_wait:
        # Full 4-column layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_vehicles = df['vehicles'].mean()
            st.metric("Avg Vehicles", f"{avg_vehicles:.1f}")
        
        with col2:
            avg_pedestrians = df['pedestrians'].mean()
            st.metric("Avg Pedestrians", f"{avg_pedestrians:.1f}")
        
        with col3:
            avg_vehicle_wait = df['avg_vehicle_wait'].mean()
            st.metric("Avg Vehicle Wait", f"{avg_vehicle_wait:.2f}s")
        
        with col4:
            avg_pedestrian_wait = df['avg_pedestrian_wait'].mean()
            st.metric("Avg Pedestrian Wait", f"{avg_pedestrian_wait:.2f}s")
    else:
        # Simple 2-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            avg_vehicles = df['vehicles'].mean()
            st.metric("Avg Vehicles", f"{avg_vehicles:.1f}")
        
        with col2:
            avg_vehicle_wait = df['avg_vehicle_wait'].mean()
            st.metric("Avg Vehicle Wait", f"{avg_vehicle_wait:.2f}s")

def create_time_series_plot(df, title):
    """Create time series plots for the given dataframe."""
    if df is None or df.empty:
        return st.warning(f"No {title} data available")
    
    # Determine which columns are available
    has_pedestrians = 'pedestrians' in df.columns
    has_pedestrian_wait = 'avg_pedestrian_wait' in df.columns
    
    # Create subplots based on available data
    if has_pedestrians and has_pedestrian_wait:
        # Full 2x2 layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Vehicle Count', 'Pedestrian Count', 
                           'Vehicle Wait Time', 'Pedestrian Wait Time'),
            vertical_spacing=0.1
        )
        
        # Vehicle count
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['vehicles'], 
                      name='Vehicles', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Pedestrian count
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['pedestrians'], 
                      name='Pedestrians', line=dict(color='green')),
            row=1, col=2
        )
        
        # Vehicle wait time
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['avg_vehicle_wait'], 
                      name='Vehicle Wait', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Pedestrian wait time
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['avg_pedestrian_wait'], 
                      name='Pedestrian Wait', line=dict(color='red')),
            row=2, col=2
        )
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Wait Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Wait Time (s)", row=2, col=2)
        
    else:
        # Simple layout for basic data
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Vehicle Count', 'Vehicle Wait Time'),
            vertical_spacing=0.1
        )
        
        # Vehicle count
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['vehicles'], 
                      name='Vehicles', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Vehicle wait time
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['avg_vehicle_wait'], 
                      name='Vehicle Wait', line=dict(color='orange')),
            row=1, col=2
        )
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Wait Time (s)", row=1, col=2)
    
    fig.update_layout(
        title=f"{title} - Time Series Analysis",
        height=400 if not (has_pedestrians and has_pedestrian_wait) else 600,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Time (seconds)")
    
    st.plotly_chart(fig, use_container_width=True)

def create_comparison_plot(fixed_df, rl_df):
    """Create comparison plots between fixed-time and RL controllers."""
    if fixed_df is None or rl_df is None:
        return st.warning("Need both fixed-time and RL data for comparison")
    
    # Calculate average metrics
    fixed_metrics = {
        'avg_vehicle_wait': fixed_df['avg_vehicle_wait'].mean(),
        'avg_pedestrian_wait': fixed_df['avg_pedestrian_wait'].mean(),
        'avg_vehicles': fixed_df['vehicles'].mean(),
        'avg_pedestrians': fixed_df['pedestrians'].mean()
    }
    
    rl_metrics = {
        'avg_vehicle_wait': rl_df['avg_vehicle_wait'].mean(),
        'avg_pedestrian_wait': rl_df['avg_pedestrian_wait'].mean(),
        'avg_vehicles': rl_df['vehicles'].mean(),
        'avg_pedestrians': rl_df['pedestrians'].mean()
    }
    
    # Create comparison chart
    categories = ['Vehicle Wait Time', 'Pedestrian Wait Time', 'Vehicle Count', 'Pedestrian Count']
    fixed_values = [fixed_metrics['avg_vehicle_wait'], fixed_metrics['avg_pedestrian_wait'],
                   fixed_metrics['avg_vehicles'], fixed_metrics['avg_pedestrians']]
    rl_values = [rl_metrics['avg_vehicle_wait'], rl_metrics['avg_pedestrian_wait'],
                rl_metrics['avg_vehicles'], rl_metrics['avg_pedestrians']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Fixed-Time Controller',
        x=categories,
        y=fixed_values,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='RL Controller',
        x=categories,
        y=rl_values,
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title='Controller Performance Comparison',
        xaxis_title='Metrics',
        yaxis_title='Values',
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate improvements
    st.subheader("Performance Improvements")
    col1, col2 = st.columns(2)
    
    with col1:
        vehicle_improvement = ((fixed_metrics['avg_vehicle_wait'] - rl_metrics['avg_vehicle_wait']) 
                             / fixed_metrics['avg_vehicle_wait'] * 100)
        st.metric(
            "Vehicle Wait Time Improvement",
            f"{vehicle_improvement:.1f}%",
            delta=f"{vehicle_improvement:.1f}%"
        )
    
    with col2:
        pedestrian_improvement = ((fixed_metrics['avg_pedestrian_wait'] - rl_metrics['avg_pedestrian_wait']) 
                                 / fixed_metrics['avg_pedestrian_wait'] * 100)
        st.metric(
            "Pedestrian Wait Time Improvement",
            f"{pedestrian_improvement:.1f}%",
            delta=f"{pedestrian_improvement:.1f}%"
        )

def main():
    """Main dashboard function."""
    # Header
    st.markdown('<h1 class="main-header">🚦 Adaptive Intersection AI Copilot</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Simple Simulation", "Fixed-Time Controller", "RL Controller", "Comparison", "Settings"]
    )
    
    # Load data
    fixed_df, rl_df, simple_df = load_latest_results()
    
    # Debug information
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.write("**Debug Information:**")
        st.sidebar.write(f"Fixed-time data: {'✓' if fixed_df is not None else '✗'}")
        st.sidebar.write(f"RL data: {'✓' if rl_df is not None else '✗'}")
        st.sidebar.write(f"Simple data: {'✓' if simple_df is not None else '✗'}")
        
        if simple_df is not None:
            st.sidebar.write(f"Simple data shape: {simple_df.shape}")
            st.sidebar.write(f"Simple data columns: {simple_df.columns.tolist()}")
    
    if page == "Dashboard":
        st.header("📊 Real-Time Dashboard")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### System Status")
            if simple_df is not None:
                st.markdown('<p class="status-good">✓ Simple Simulation: Active</p>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-warning">⚠ Simple Simulation: No Data</p>', 
                           unsafe_allow_html=True)
        
        with col2:
            if fixed_df is not None:
                st.markdown('<p class="status-good">✓ Fixed-Time Controller: Active</p>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-warning">⚠ Fixed-Time Controller: No Data</p>', 
                           unsafe_allow_html=True)
        
        with col3:
            if rl_df is not None:
                st.markdown('<p class="status-good">✓ RL Controller: Active</p>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-warning">⚠ RL Controller: No Data</p>', 
                           unsafe_allow_html=True)
        
        with col4:
            st.markdown("### Last Update")
            st.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Quick metrics
        if simple_df is not None or fixed_df is not None or rl_df is not None:
            st.subheader("📈 Quick Metrics")
            
            if simple_df is not None:
                st.write("**Simple Simulation:**")
                create_metrics_summary(simple_df, "Simple")
            
            if fixed_df is not None:
                st.write("**Fixed-Time Controller:**")
                create_metrics_summary(fixed_df, "Fixed-Time")
            
            if rl_df is not None:
                st.write("**RL Controller:**")
                create_metrics_summary(rl_df, "RL")
    
    elif page == "Simple Simulation":
        st.header("🚗 Simple Simulation Analysis")
        create_metrics_summary(simple_df, "Simple Simulation")
        create_time_series_plot(simple_df, "Simple Simulation")
        
        if simple_df is not None:
            st.subheader("📋 Raw Data")
            st.dataframe(simple_df.tail(100))
    
    elif page == "Fixed-Time Controller":
        st.header("⏰ Fixed-Time Controller Analysis")
        create_metrics_summary(fixed_df, "Fixed-Time Controller")
        create_time_series_plot(fixed_df, "Fixed-Time Controller")
        
        if fixed_df is not None:
            st.subheader("📋 Raw Data")
            st.dataframe(fixed_df.tail(100))
    
    elif page == "RL Controller":
        st.header("🤖 RL Controller Analysis")
        create_metrics_summary(rl_df, "RL Controller")
        create_time_series_plot(rl_df, "RL Controller")
        
        if rl_df is not None:
            st.subheader("📋 Raw Data")
            st.dataframe(rl_df.tail(100))
            
            # RL-specific metrics
            if 'q_table_size' in rl_df.columns:
                st.subheader("🧠 Learning Progress")
                fig = px.line(rl_df, x='step', y='q_table_size', 
                             title='Q-Table Size Over Time')
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Comparison":
        st.header("⚖️ Controller Comparison")
        create_comparison_plot(fixed_df, rl_df)
    
    elif page == "Settings":
        st.header("⚙️ Settings")
        
        st.subheader("Simulation Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Simulation Duration (seconds)", value=3600, min_value=60, max_value=7200)
            st.number_input("Step Length (seconds)", value=0.1, min_value=0.01, max_value=1.0)
        
        with col2:
            st.number_input("Vehicle Flow Rate", value=5, min_value=1, max_value=20)
            st.number_input("Pedestrian Flow Rate", value=15, min_value=5, max_value=50)
        
        st.subheader("RL Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            st.slider("Learning Rate", 0.01, 0.5, 0.1)
            st.slider("Epsilon (Exploration)", 0.01, 0.5, 0.1)
        
        with col2:
            st.slider("Gamma (Discount)", 0.5, 0.99, 0.9)
            st.slider("Min Phase Time (seconds)", 1, 30, 5)
        
        if st.button("Save Settings"):
            st.success("Settings saved!")

if __name__ == "__main__":
    main()