import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit import title

# Page configuration
st.set_page_config(layout="wide")
st.title('Iris Dataset Analysis')


# Load and prepare data
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris.feature_names


df, feature_names = load_data()
X = df[feature_names].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar controls
st.sidebar.header('Analysis Controls')
clusters = st.sidebar.slider('Select Number of Clusters:', 1, 6, 3)

# 1. Feature Distribution Analysis
st.header('1. Feature Distributions by Species')

# Colors for species
colors = {'setosa': '#FF4B4B', 'versicolor': '#4B4BFF', 'virginica': '#4BFF4B'}

# Feature selection for box plot
selected_feature = st.selectbox('Select Feature for Box Plot:', feature_names)
fig_box = px.box(
    df,
    x='Species',
    y=selected_feature,
    color='Species',
    color_discrete_map=colors
)

fig_box.update_layout(
    title=f"Distribution of {selected_feature} by Species",
    xaxis_title="Species",
    yaxis_title=selected_feature
)
st.plotly_chart(fig_box)


# 2. Feature Relationships
st.header('2. Feature Relationships')

fig_scatter_matrix = px.scatter_matrix(df, dimensions=feature_names, color='Species', color_discrete_map=colors)
fig_scatter_matrix.update_layout(
    title="Feature Relationships by Species",
    height=800,  # Increase height to add more space
)
st.plotly_chart(fig_scatter_matrix)

# 3. Feature Correlations
st.header('3. Feature Correlations')
correlation = df[feature_names].corr()
# Revert the order of rows in the correlation matrix and round values to 2 decimal places
correlation_reverted_rows = correlation.iloc[::-1, :].round(2)

# Create correlation heatmap
fig_heatmap = px.imshow(
    correlation_reverted_rows,
    color_continuous_scale='RdBu',
    text_auto=True,
    zmin=-1,
    zmax=1,
)

fig_heatmap.update_layout(
    title="Feature Correlation Heatmap",
    height=600,  # Increase height
    coloraxis_colorbar=dict(
        thickness=30,
        y=0.5,
    ),

)

st.plotly_chart(fig_heatmap)

# 4. Elbow Analysis
st.header('4. Elbow Analysis')


@st.cache_data
def perform_elbow_analysis(X, max_clusters=10):
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    return inertias


inertias = perform_elbow_analysis(X_scaled, clusters)

fig_elbow = px.line(x=range(1, len(inertias) + 1), y=inertias, markers=True)
fig_elbow.update_layout(
    title="Elbow Method Analysis",
    xaxis_title='Number of Clusters',
    yaxis_title='Inertia'
)
st.plotly_chart(fig_elbow)

# 5. Clustering Analysis
st.header('5. Clustering Analysis')

# Perform clustering
kmeans = KMeans(n_clusters=clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = cluster_labels.astype(str)
df_sorted = df.sort_values(by='Cluster', ascending=True)

# Create comparison plots
col1, col2 = st.columns(2)
with col1:
    st.subheader('Clustering Result')

    fig_clusters = px.scatter(df_sorted, x=feature_names[2], y=feature_names[3], color='Cluster')
    fig_clusters.update_layout(
        title="KMeans Clustering Result"
    )
    st.plotly_chart(fig_clusters)

with col2:
    st.subheader('Actual Species')
    fig_species = px.scatter(df, x=feature_names[2], y=feature_names[3], color='Species', color_discrete_map=colors)
    fig_species.update_layout(
        title="Actual Species Distribution",
    )
    st.plotly_chart(fig_species)

# 6. Clustering Performance Analysis
st.header('6. Clustering Performance')
confusion_df = pd.crosstab(df['Species'], df['Cluster'], margins=True)
st.write("Confusion Matrix (Species vs Clusters):")
st.write(confusion_df)

# 7. Additional Statistics
st.header('7. Feature Statistics')
col3, col4 = st.columns(2)

with col3:
    st.subheader('Statistics by Species')
    species_stats = df.groupby('Species')[feature_names].agg(['mean', 'std']).round(2)
    st.write(species_stats)

with col4:
    st.subheader('Statistics by Cluster')
    cluster_stats = df.groupby('Cluster')[feature_names].agg(['mean', 'std']).round(2)
    st.write(cluster_stats)
