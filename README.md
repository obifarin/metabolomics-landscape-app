# Metabolomics Landscape App
Embeddings Explorer for The Landscape of Metabolomics Research

## Overview
This Streamlit app, Landscape of Metabolomics Research, is designed to help users visualize and explore trends in metabolomics research using data-driven techniques. The app provides interactive visualizations of research embeddings generated by a transformer-based model (PubMedBERT), which converts scientific abstracts into numerical representations. These embeddings are then visualized using t-SNE plots. The app also allows users to search for specific keywords and authors within the dataset, and analyze keyword trends over time.

## Features
1. **Embeddings Explorer**:
   - Visualizes t-SNE embeddings colored by publication year and the presence of specific keywords within abstracts or titles.
   - Users can select specific research clusters and keywords to narrow down their search.
   - Option to show only papers with keyword matches to focus on relevant research.
   - Customizable search parameters include selecting specific research clusters, keywords, and searching by abstract or title.

2. **Keyword Trend Analysis**:
   - Tracks the frequency of specific keywords in the metabolomics literature over time.
   - Displays percentage of papers mentioning selected keywords for each year.
   - Helps identify emerging research trends and track the rise or decline of specific topics.
   - Compare multiple keywords in the same visualization to see relative popularity.

3. **Author Search**:
   - Users can search for publications by specific authors and visualize their work in the embeddings plot.
   - Option to display other papers in the background for context.
   - Features quick-access buttons for notable researchers in the field (Jeremy Nicholson, Oliver Fiehn, Alisdair Fernie).
   - Hover information reveals paper titles, journals, and publication years.

4. **Information Sharing**:
   - Users can share their findings via email for collaborative research.

## Usage
1. **Home Page**:
   - Learn about the study, its goals, and the methods used.
   - Watch demo videos including an app demonstration and a 10-minute study summary.
  
2. **Embeddings Explorer**:
   - Navigate to the Embeddings Explorer via the sidebar.
   - Enter the cluster name, keywords, and the location (abstract or title) to search for relevant studies.
   - Optionally filter to show only papers with keyword matches.
   - Click "Generate Plot" to display a t-SNE plot with research publications colored by keyword presence and publication year.
   - Hover over points to see details about the corresponding papers.

3. **Keyword Trend Analysis**:
   - Enter one or more keywords separated by commas.
   - Select whether to search in paper abstracts or titles.
   - Click "Generate Trend Analysis" to create a visualization showing how keyword frequency changes over time.
   - Compare multiple keywords to identify research trends and patterns.

4. **Author Search**:
   - Enter an author's name (first and last name) or use one of the predefined author buttons.
   - Optionally display other papers in the background for context.
   - View a visualization highlighting papers by the selected author as blue dots.
   - Hover over points to see paper titles, journals, and publication years.

## Installation
To run the app, ensure that the required dependencies are installed:

```bash
pip install streamlit pandas plotly openpyxl
```

To run the app locally, use the command:

```bash
streamlit run app.py
```

## Dependencies
The app relies on the following key libraries:

- Streamlit: For the web interface.
- Pandas: For handling the dataset.
- Plotly: For creating interactive visualizations.
- Openpyxl: To read Excel files.
- Regular Expressions (re): For keyword searching.

## Data
The dataset used is an Excel file (metabolomics_landscape_app_01MAR2024.xlsx) containing research papers with columns including:

- tsne_2D_x, tsne_2D_y: Embedding coordinates for plotting.
- predicted_category: The research cluster assigned to each paper.
- abstract, title, authors: Metadata for each paper.
- pub_year: Publication year.
- journal_title: The journal where the paper was published.
