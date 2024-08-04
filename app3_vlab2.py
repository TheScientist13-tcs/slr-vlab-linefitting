### By: Dharyll Prince M. Abellana

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")
from sklearn.preprocessing import RobustScaler
import streamlit as st
import plotly.graph_objects as go
import streamlit as st
from helper_funcs import (
    load_data,
    plotly_scatter,
    calculate_rss,
    filter_df,
    styled_output_box,
)


def main():
    try:
        st.set_page_config(
            page_title="Simple Linear Regression", layout="wide", page_icon="📉"
        )
        st.title(":gray-background[Virtual Lab: Fitting a Line to the Data]")
        st.text("By: Dharyll Prince M. Abellana | Assistant Professor of Computer Science | University of the Philippines Cebu")

        # Load the data

        file_name = "income.csv"
        dataset = load_data(file_name)
        dataset.columns = map(str.capitalize, dataset.columns)

        st.markdown("###")
        st.markdown(
            "### Experiment on the values of the 'Intercept' and the 'Slope' that would give the lowest 'RSS' value"
        )
        col1, col_, col2 = st.columns([0.2, 0.05, 0.7])

        # Model Building
        ## Simple Linear Regression
        with col1:

            st.markdown("##### Parameters")
            st.markdown("###### Enter any real number of your choice below:")
            beta_0_exp = st.number_input("INTERCEPT", value=0.0, step=0.01)
            beta_0_exp = np.round(float(beta_0_exp), 3)
            beta_1_exp = st.number_input("SLOPE", value=0.2, step=0.01)
            beta_1_exp = np.round(float(beta_1_exp), 3)

        with col2:
            filtered_dataset = filter_df(dataset, dataset.columns)
            x_feature = "Experience"
            y_feature = "Income"
            x = filtered_dataset[x_feature].values
            y = filtered_dataset[y_feature].values
            fig = plotly_scatter(x, y)

        with col1:
            y_exp = beta_0_exp + beta_1_exp * x
            rss = calculate_rss(y, y_exp)
            st.markdown("##### RSS")
            st.markdown(styled_output_box(str(rss)), unsafe_allow_html=True)
            st.markdown("###### ***Try to find the lowest possible value for RSS.")

        with col2:

            fig.add_trace(
                go.Line(
                    x=x,
                    y=y_exp,
                    mode="lines",
                    line=dict(color="#134B70", width=4),
                    name=f"y = {beta_0_exp:.3f} + {beta_1_exp:.3f}x",
                )
            )

            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="#46494c",
                zerolinecolor="#46494c",
                tickfont=dict(size=15),
                title_text=x_feature.capitalize(),
                titlefont=dict(size=17),
            )

            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="#46494c",
                zerolinecolor="#46494c",
                tickfont=dict(size=15),
                title_text=y_feature.capitalize(),
                titlefont=dict(size=17),
            )

            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
