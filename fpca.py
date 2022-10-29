#!/usr/bin/env python3
# Source: https://towardsdatascience.com/beyond-classic-pca-functional-principal-components-analysis-fpca-applied-to-time-series-with-python-914c058f47a0
# Or https://gist.github.com/pierrelouisbescond/9c0b8b925741e4376baf4883306a9022
import pandas as pd
import numpy as np
from fdasrsf import fPCA, time_warping, fdawarp, fdahpca


# Import the CSV file with only useful columns
# source: https://www.data.gouv.fr/fr/datasets/temperature-quotidienne-departementale-depuis-janvier-2018/
df = pd.read_csv("temperature-quotidienne-departementale.csv", sep=";", usecols=[0,1,4])

# Rename columns to simplify syntax
df = df.rename(columns={"Code INSEE département": "Region", "TMax (°C)": "Temp"})

# Select 2019 records only
df = df[(df["Date"]>="2019-01-01") & (df["Date"]<="2019-12-31")]

# Pivot table to get "Date" as index and regions as columns 
df = df.pivot(index='Date', columns='Region', values='Temp')

# Select a set of regions across France
df = df[["06","25","59","62","83","85","75"]]

display(df)

# # Convert the Pandas dataframe to a Numpy array with time-series only
# f = df.to_numpy().astype(float)

# # Create a float vector between 0 and 1 for time index
# time = np.linspace(0,1,len(f))

# # Functional Alignment
# # Align time-series
# warp_f = time_warping.fdawarp(f, time)
# warp_f.srsf_align()

# warp_f.plot()

# # Functional Principal Components Analysis

# # Define the FPCA as a vertical analysis
# fPCA_analysis = fPCA.fdavpca(warp_f)

# # Run the FPCA on a 3 components basis 
# fPCA_analysis.calc_fpca(no=3)
# fPCA_analysis.plot()

# import plotly.graph_objects as go

# # Plot of the 3 functions
# fig = go.Figure()

# # Add traces
# fig.add_trace(go.Scatter(y=fPCA_analysis.f_pca[:,0,0], mode='lines', name="PC1"))
# fig.add_trace(go.Scatter(y=fPCA_analysis.f_pca[:,0,1], mode='lines', name="PC2"))
# fig.add_trace(go.Scatter(y=fPCA_analysis.f_pca[:,0,2], mode='lines', name="PC3"))

# fig.update_layout(
#     title_text='<b>Principal Components Analysis Functions</b>', title_x=0.5,
# )

# fig.show()

# # Coefficients of PCs against regions
# fPCA_coef = fPCA_analysis.coef

# # Plot of PCs against regions
# fig = go.Figure(data=go.Scatter(x=fPCA_coef[:,0], y=fPCA_coef[:,1], mode='markers+text', text=df.columns))

# fig.update_traces(textposition='top center')

# fig.update_layout(
#     autosize=False,
#     width=800,
#     height=700,
#     title_text='<b>Function Principal Components Analysis on 2018 French Temperatures</b>', title_x=0.5,
#     xaxis_title="PC1",
#     yaxis_title="PC2",
# )
# fig.show()
