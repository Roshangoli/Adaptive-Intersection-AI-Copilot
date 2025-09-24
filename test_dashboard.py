#!/usr/bin/env python3
"""
Simple test dashboard to debug visualization issues.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob

st.set_page_config(
    page_title="Test Dashboard",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Test Dashboard")

# Load data
def load_data():
    results_dir = "results"
    if not os.path.exists(results_dir):
        st.error("Results directory not found")
        return None
    
    simple_files = glob.glob(os.path.join(results_dir, "simple_results_*.csv"))
    if not simple_files:
        st.error("No simple results files found")
        return None
    
    latest_simple = max(simple_files, key=os.path.getctime)
    st.info(f"Loading data from: {latest_simple}")
    
    try:
        df = pd.read_csv(latest_simple)
        st.success(f"Data loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load and display data
df = load_data()

if df is not None:
    st.subheader("Data Preview")
    st.dataframe(df.head(10))
    
    st.subheader("Data Info")
    st.write(f"Columns: {df.columns.tolist()}")
    st.write(f"Shape: {df.shape}")
    
    st.subheader("Basic Statistics")
    st.write(df.describe())
    
    st.subheader("Simple Plot")
    try:
        fig = px.line(df, x='step', y='vehicles', title='Vehicle Count Over Time')
        st.plotly_chart(fig, use_container_width=True)
        st.success("Plot created successfully!")
    except Exception as e:
        st.error(f"Error creating plot: {e}")
    
    st.subheader("Multi-plot Test")
    try:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Vehicle Count', 'Vehicle Wait Time')
        )
        
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['vehicles'], 
                      name='Vehicles', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['avg_vehicle_wait'], 
                      name='Vehicle Wait', line=dict(color='orange')),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(title_text="Time (seconds)")
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Wait Time (s)", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        st.success("Multi-plot created successfully!")
    except Exception as e:
        st.error(f"Error creating multi-plot: {e}")