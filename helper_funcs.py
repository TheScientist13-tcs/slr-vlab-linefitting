import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from sklearn.preprocessing import RobustScaler


@st.cache_data
def load_data(dataset_address):
    dataset = pd.read_csv(dataset_address)
    return dataset


@st.cache_data
def plotly_scatter(x, y):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color="#508C9B", size=14, opacity=0.8),
            name="Data",
        )
    )
    fig.update_layout(paper_bgcolor=None, height=505, plot_bgcolor="#0C0404")
    return fig


def calculate_rss(y, y_pred):
    return np.round(np.sum((y - y_pred) ** 2), 3)


def filter_df(df, selected_cols, normalize=True):
    filtered_dataset = df[selected_cols]
    filtered_dataset = filtered_dataset.dropna(axis=0)
    filtered_dataset = filtered_dataset.drop_duplicates()
    filtered_dataset = filtered_dataset.reset_index(drop=True)
    if normalize:
        scaler = RobustScaler()
        filtered_dataset = scaler.fit_transform(filtered_dataset)
        filtered_dataset = pd.DataFrame(filtered_dataset, columns=selected_cols)
    return filtered_dataset


@st.cache_data
def styled_output_box(
    text, background_color="#f0f0f0", text_color="black", padding=10, border_radius=5
):
    """
    Create a styled output box using HTML and Markdown.

    Parameters:
    - text (str): The text to display inside the box.
    - background_color (str): Background color of the box (default: '#f0f0f0').
    - text_color (str): Color of the text inside the box (default: 'black').
    - padding (int): Padding around the text inside the box (default: 10).
    - border_radius (int): Border radius to round the corners of the box (default: 5).
    """
    return f"""
        <div 
            style="
                background-color: {background_color};
                color: {text_color};
                padding: {padding}px;
                border-radius: {border_radius}px;
                border: 1px solid #ccc;
                box-shadow: 2px 2px 2px rgba(0, 0, 0, 0.2);
                "
        >
        {text}
        </div>
    """
