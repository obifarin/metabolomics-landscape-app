import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")

try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import re
    #import smtplib
    #from email.mime.text import MIMEText
    #from email.mime.multipart import MIMEMultipart
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Load your dataframe
@st.cache_data
def load_data():
    df = pd.read_excel('metabolomics_landscape_app_01MAR2024.xlsx')
    return df

df = load_data()

def clusterByKeywords2(cluster_name, keywords, location, only_matches=False):
    if cluster_name != 'All embeddings':
        cluster_df = df[df['predicted_category'] == cluster_name]
    else:
        cluster_df = df

    cluster_df['keyword_presence'] = 'No Keyword Match'

    # Apply keyword filtering with OR functionality
    for keyword_entry in keywords:
        keyword_entry = keyword_entry.strip()
        if not keyword_entry:
            continue
            
        # Check if this is an OR group (contains | character)
        if '|' in keyword_entry:
            # Split into individual keywords for the OR relationship
            or_keywords = [k.strip() for k in keyword_entry.split('|') if k.strip()]
            display_name = f"{' | '.join(or_keywords)}"  # Use a readable name
            
            # Check if any of the OR keywords match
            def matches_any_keyword(text):
                if pd.isna(text):
                    return False
                # Check each keyword in the OR group
                return any(bool(re.search(k, str(text), re.IGNORECASE)) for k in or_keywords)
                
            # Apply the OR matching function
            matches_mask = cluster_df[location].apply(matches_any_keyword)
            cluster_df.loc[matches_mask, 'keyword_presence'] = display_name
        else:
            # Original single keyword logic
            cluster_df.loc[cluster_df[location].apply(lambda x: bool(re.search(keyword_entry, x, re.IGNORECASE))), 
                          'keyword_presence'] = keyword_entry

    if only_matches:
        cluster_df = cluster_df[cluster_df['keyword_presence'] != 'No Keyword Match']

    # Recalculate the min and max years from the filtered dataset
    min_year = cluster_df['pub_year'].min()
    max_year = cluster_df['pub_year'].max()

    color_scale_time = [
        (0, 'rgb(0, 0, 255)'),  # Blue for oldest
        (0.5, 'rgb(255, 255, 0)'),  # Yellow for middle
        (1, 'rgb(255, 0, 0)')   # Red for newest
    ]

    # Define a faint, transparent gray color for 'No Keyword Match'
    no_match_color = 'rgba(200, 200, 200, 0.1)'  # Reduced opacity

    # Modify the color assignment for 'None' keyword
    cluster_df.loc[cluster_df['keyword_presence'] == 'None', 'keyword_presence'] = 'No Keyword Match'

    # Create separate DataFrames for matched and unmatched entries
    matched_df = cluster_df[cluster_df['keyword_presence'] != 'No Keyword Match']
    unmatched_df = cluster_df[cluster_df['keyword_presence'] == 'No Keyword Match']

    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=("Colored by Year", "Colored by Keyword Presence"),
        vertical_spacing=0.1
    )

    # Plot for Colored by Year
    fig_time_matched = px.scatter(matched_df, x='tsne_2D_x', y='tsne_2D_y', color='pub_year',
                                  color_continuous_scale=color_scale_time, opacity=1,
                                  hover_data={'tsne_2D_x': False, 'tsne_2D_y': False, 'title': True, 'pub_year': True, 'authors': True},
                                  range_color=[min_year, max_year])

    fig_time_unmatched = px.scatter(unmatched_df, x='tsne_2D_x', y='tsne_2D_y',
                                    color_discrete_sequence=[no_match_color],
                                    opacity=0.3,
                                    hover_data={'tsne_2D_x': False, 'tsne_2D_y': False, 'title': True})

    # Customize hover text for matched entries in the time plot
    fig_time_matched.update_traces(
        hovertemplate="Title: %{customdata[0]}<br>Authors: %{customdata[2]}<br>Publication Year: %{customdata[1]}<extra></extra>"
    )

    # Customize hover text for unmatched entries in the time plot
    fig_time_unmatched.update_traces(
        hovertemplate="Title: %{customdata[0]}<extra></extra>"
    )

    for trace in fig_time_matched['data']:
        trace.marker.size = 5  # Slightly larger default size=6
        fig.add_trace(trace, row=1, col=1)

    for trace in fig_time_unmatched['data']:
        trace.marker.size = 3  # Smaller
        trace.name = 'No Keyword Match'
        fig.add_trace(trace, row=1, col=1)

    # Plot for Colored by Keyword Presence
    fig_keywords_matched = px.scatter(matched_df, 
                                      x='tsne_2D_x', 
                                      y='tsne_2D_y', 
                                      color='keyword_presence',
                                      color_discrete_sequence=px.colors.qualitative.Alphabet,
                                      opacity=1,
                                      hover_data={'tsne_2D_x': False, 'tsne_2D_y': False, 'title': True, 'keyword_presence': True, 'authors': True})

    fig_keywords_unmatched = px.scatter(unmatched_df,
                                        x='tsne_2D_x',
                                        y='tsne_2D_y',
                                        color_discrete_sequence=[no_match_color],
                                        opacity=0.3,
                                        hover_data={'tsne_2D_x': False, 'tsne_2D_y': False, 'title': True})

    # Customize hover text for matched entries in the keyword plot
    fig_keywords_matched.update_traces(
        hovertemplate="Title: %{customdata[0]}<br>Authors: %{customdata[2]}<br>Keyword: %{customdata[1]}<extra></extra>"
    )

    # Customize hover text for unmatched entries in the keyword plot
    fig_keywords_unmatched.update_traces(
        hovertemplate="Title: %{customdata[0]}<extra></extra>"
    )

    for trace in fig_keywords_matched['data']:
        trace.marker.size = 5  # Slightly larger default size=6
        fig.add_trace(trace, row=2, col=1)

    for trace in fig_keywords_unmatched['data']:
        trace.marker.size = 3  # Smaller
        trace.name = 'No Keyword Match'
        fig.add_trace(trace, row=2, col=1)

    fig.update_layout(
        title="Embeddings Explorer",
        plot_bgcolor='white',
        height=700, width=1000,
        title_font=dict(size=24, family='Arial, sans-serif', color='#333333'),
        font=dict(size=14, family='Arial, sans-serif', color='#333333'),
        margin=dict(l=50, r=50, t=80, b=50),
        coloraxis=dict(colorscale=color_scale_time, 
                       colorbar=dict(title="Year", y=0.85, thickness=15, len=0.3),
                       cmin=min_year, cmax=max_year),  # Set dynamic range for filtered subset
        coloraxis2=dict(colorbar=dict(title="Keyword Presence", y=0.35, thickness=15, len=0.3)),
    )

    fig.update_xaxes(title='', showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(title='', showticklabels=False, showgrid=False, zeroline=False)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgray', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgray', mirror=True)

    fig.update_annotations(font_size=18)

    fig.update_layout(
        legend=dict(
            title=dict(text='Keywords'),
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1
        )
    )

    return fig


def highlightAuthor(author_name, show_other):
    # Split the author_name into parts based on whitespace
    author_parts = author_name.split()
    
    # Determine last_name and first_initial based on the number of parts
    if len(author_parts) > 1:
        last_name = author_parts[-1].lower()  # Last part is the last name
        first_initial = author_parts[0][0].lower()  # First character of first part is initial
    else:
        last_name = author_parts[0].lower()  # Only one part, assume it's the full name
        first_initial = ''  # No first initial if only one part
    
    # Function to check if any author's last name and initial match the given criteria
    def author_in_list(authors):
        if not isinstance(authors, str):
            return False
        authors = authors.lower()
        # Split authors into individual names and check each one
        for author in authors.split(','):
            author = author.strip()  # Remove any leading or trailing whitespace
            author_parts = author.split()  # Split author name into parts
            if len(author_parts) > 1:
                author_last_name = author_parts[0].lower()  # First part is the last name
                author_first_initial = author_parts[1][0].lower()  # First character of second part is initial
                # Check if last_name and first_initial match
                if last_name == author_last_name and (not first_initial or first_initial == author_first_initial):
                    return True
        return False

    # Apply author_in_list function to create a new column 'highlight' in the DataFrame
    df['highlight'] = df['authors'].apply(author_in_list)
    
    # Assign 'color' based on 'highlight' column (highlighted or not)
    df['color'] = df['highlight'].apply(lambda x: author_name if x else 'other')

    # Create a new figure
    fig = go.Figure()

    # If show_other is True, add scatter plot for non-highlighted points
    if show_other:
        other_df = df[~df['highlight']]  # DataFrame of non-highlighted points
        fig.add_trace(go.Scattergl(  # Use Scattergl for better performance
            x=other_df['tsne_2D_x'], 
            y=other_df['tsne_2D_y'],
            mode='markers',
            marker=dict(color='rgba(160, 160, 160, 0.5)', size=3),  # Smaller size, no border
            name='Other',
            hoverinfo='none'  # No hover information for other points
        ))

    # Add scatter plot for highlighted points
    highlight_df = df[df['highlight']]  # DataFrame of highlighted points
    fig.add_trace(go.Scattergl(  # Use Scattergl for better performance
        x=highlight_df['tsne_2D_x'], 
        y=highlight_df['tsne_2D_y'],
        mode='markers',
        marker=dict(
            color='rgb(0, 0, 225)',  # Blue
            size=10,  # Keep original size
            line=dict(width=2, color='rgb(0, 0, 100)')  # Keep original border
        ),
        name=author_name,
        # Create hover text with title and journal information
        text = highlight_df.apply(lambda row: f"Title: {row['title']}<br>Journal: {row['journal_title']}<br>Date of Publication: {row['pub_year']}", axis=1),
        hoverinfo='text'
    ))

    # Update layout settings for the plot
    fig.update_layout(
        title=f"Papers by {author_name}",
        plot_bgcolor='white',  # White background
        height=600,  # Set plot height
        width=1000,  # Set plot width
        title_font=dict(size=24, family='Arial, sans-serif', color='#333333'),  # Title font settings
        font=dict(size=14, family='Arial, sans-serif', color='#333333'),  # General font settings
        margin=dict(l=50, r=50, t=80, b=50),  # Margins
        showlegend=True,  # Show legend
    )

    # Remove axis labels, tick marks, and grid
    fig.update_xaxes(title='', showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(title='', showticklabels=False, showgrid=False, zeroline=False)

    # Add light gray border lines
    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgray', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgray', mirror=True)

    return fig


def analyze_keyword_trends(keywords, location):
    """
    Analyze and visualize the frequency of keywords over time.
    
    Parameters:
    - keywords: List of keywords to track (can include OR relationships using | character)
    - location: Where to search for keywords ('abstract' or 'title')
    
    Returns:
    - Plotly figure showing keyword trends over time
    """
    # Filter out empty keywords
    keywords = [k.strip() for k in keywords if k.strip()]
    
    if not keywords:
        return None
    
    # Get all unique years in the dataset, excluding 2024
    all_years = sorted([year for year in df['pub_year'].unique() if year != 2024])
    
    # Calculate total papers per year (excluding 2024)
    year_counts = df[df['pub_year'] != 2024]['pub_year'].value_counts().reindex(all_years, fill_value=0)
    
    # Create figure
    fig = go.Figure()
    
    # Process each keyword or keyword group
    for keyword_entry in keywords:
        # Check if this is an OR group (contains | character)
        if '|' in keyword_entry:
            # Split into individual keywords for the OR relationship
            or_keywords = [k.strip() for k in keyword_entry.split('|') if k.strip()]
            display_name = f"{' | '.join(or_keywords)}"  # Use a readable name for the legend
            
            # Create regex patterns for each keyword in the OR group
            patterns = [re.compile(k, re.IGNORECASE) for k in or_keywords]
        else:
            # Single keyword case (existing functionality)
            display_name = keyword_entry
            patterns = [re.compile(keyword_entry, re.IGNORECASE)]
        
        # Initialize a counts dictionary with zeros for all years
        keyword_counts = {year: 0 for year in all_years}
        
        # Count papers containing the keyword(s) for each year
        for year in all_years:
            year_df = df[df['pub_year'] == year]
            if not year_df.empty:
                # Count papers matching ANY of the patterns in the OR group
                def matches_any_pattern(text):
                    if pd.isna(text):
                        return False
                    # Return True if any pattern matches
                    return any(p.search(str(text)) for p in patterns)
                
                # Count matching papers
                matches = year_df[location].apply(matches_any_pattern).sum()
                
                # Calculate percentage if there are papers this year
                if year_counts[year] > 0:
                    keyword_counts[year] = (matches / year_counts[year]) * 100
        
        # Create x and y lists ensuring all years are included
        x_values = all_years
        y_values = [keyword_counts[year] for year in all_years]
        
        # Add trace with explicit line connection
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            name=display_name,
            line=dict(
                shape='linear',  # Linear interpolation between points
                dash='solid',    # Solid line
                width=2          # Line width
            ),
            marker=dict(
                size=8,          # Larger markers to highlight data points
                symbol='circle'  # Circle markers
            ),
            connectgaps=True,    # Important: Connect gaps across zero/null values
            hovertemplate='Year: %{x}<br>Percentage: %{y:.2f}%<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title="Keyword Trends Over Time",
        xaxis_title="Year",
        yaxis_title="Percentage of Papers (%)",
        plot_bgcolor='white',
        height=500,  # Increase height to accommodate legend
        width=900,
        title_font=dict(size=24, family='Arial, sans-serif', color='#333333'),
        font=dict(size=14, family='Arial, sans-serif', color='#333333'),
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            title=dict(text='Keywords'),
            orientation="h",     # Horizontal orientation
            yanchor="bottom",   # Anchor point at bottom
            y=-0.5,            # Move legend down further (more negative value = lower position)
            xanchor="center",   # Center horizontally
            x=0.5,             # Center position
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1
        )
    )
    
    # Show fewer x-axis labels and reduce grid line thickness
    selected_years = all_years[::4]  # For example, every 5th year
    fig.update_xaxes(
        type='category',
        tickmode='array',
        tickvals=selected_years,
        ticktext=[str(year) for year in selected_years],
        showgrid=True,
        gridwidth=0.1,  # Thinner grid lines
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.1,  # Thinner grid lines
        gridcolor='lightgray'
    )
    
    return fig

st.title("A _Map_ of :blue[_Metabolomics_] Research 📄")
st.write("**_Aditya Simhadri_** and [**_Olatomiwa O. Bifarin_**](https://www.linkedin.com/in/obifarin/), [Fernández Lab](https://sites.gatech.edu/fernandez/), Georgia Tech")

# Create a navigation menu with more options
page = st.sidebar.selectbox(
    "Choose a page", 
    ["Home", "Embeddings Explorer", "Keyword Trend Analysis", "Author Search"]
)

if page == "Home":
    st.subheader("About the Study")
    st.write("""
    This app is designed to help users visualize and explore trends in metabolomics 
             research using data-driven techniques. The app provides interactive 
             visualizations of research embeddings generated by a transformer-based 
             model (PubMedBERT), which converts scientific abstracts into numerical 
             representations. These embeddings are then visualized using t-SNE plots. 
             The app also allows users to search for specific keywords and authors 
             within the dataset, and analyze keyword trends over time. Data was collected
             from PubMed, data range: 1998-early 2024.
    """)
    st.write("**Link to** [**Preprint**](https://www.biorxiv.org/content/10.1101/2025.03.18.643696v1)")

    st.subheader(""" Demo of App :video_camera:""")
    st.video("https://www.youtube.com/watch?v=ZwG4-E1wtwA")

    #st.subheader(""" 10-minute Study Summary :movie_camera:""")
    #st.video("https://www.youtube.com/watch?v=eHrCx2LhdCk")

elif page == "Embeddings Explorer":
    st.header("Embeddings :blue[_Explorer_] 🌐")
    with st.expander("How to use"):
        st.write("""
        This tool allows you to explore research papers in the metabolomics field using t-SNE visualizations.
        
        **How to use:**
        1. Select a research cluster from the dropdown menu in the sidebar (or choose "All embeddings")
        2. Enter keywords of interest, separated by commas
        3. Use the pipe symbol (|) between terms for OR logic: "metabolite|metabolites" will match papers with either term
        4. Choose whether to search in paper abstracts or titles
        5. Optionally check "Show only papers with keyword matches" to filter the visualization
        6. Click "Generate Plot" to create the visualization
        
        **Examples:**
        - Single keyword: "TOCSY"
        - Multiple keywords: "breast cancer, colon cancer"
        - OR relationship: "Deep learning|neural networks, Faecalibacterium"
        
        The visualization displays two plots:
        - **Top plot**: Papers colored by publication year (blue → yellow → red from oldest to newest)
        - **Bottom plot**: Papers colored by keyword matches
        
        **Note**: If a paper matches multiple keywords, it will be colored according to the last matching keyword in your list.
        
        Hover over any point to see details about the corresponding paper.
        """)
    
    # Parameters section - moved from sidebar to main page
    st.subheader("Parameters")
    
    # Create two columns for the parameters
    col1, col2 = st.columns(2)
    
    with col1:
        cluster_name = st.selectbox("Select Research Cluster", 
                                   options=["All embeddings"] + list(df['predicted_category'].unique()))
        location = st.selectbox("Select Location to Search Keywords", 
                               options=["abstract", "title"])
    
    with col2:
        keywords = st.text_input("Enter Keywords (comma-separated)").split(',')
        only_matches = st.checkbox("Show only papers with keyword matches", value=False)
    
    # Generate plot button
    if st.button("Generate Plot", type="primary"):
        if keywords and location:
            fig = clusterByKeywords2(cluster_name, keywords, location, only_matches)
            st.plotly_chart(fig)
        else:
            st.error("Please provide all inputs")

elif page == "Keyword Trend Analysis":
    # Keyword Trend Analysis page
    st.header("Keyword :blue[_Trend Analysis_] 📈")
    with st.expander("How to use"):
        st.write("""
        This feature tracks how frequently specific keywords appear in metabolomics literature over time.
        
        **How to use:**
        1. Enter one or more keywords of interest, separated by commas
        2. Use the pipe symbol (|) between keywords for OR logic: "metabolite|metabolites" will match papers with either term
        3. Select whether to search in paper abstracts or titles
        4. Click "Generate Trend Analysis" to create the visualization
        
        **Examples:**
        - Single keyword: "lipidomics"
        - Multiple keywords: "breast cancer, colon cancer"
        - OR relationship: "Mass spectrometry|MS|Mass spec, NMR|NMR Spectroscopy|Nuclear Magnetic Resonance"
        
        The graph shows the percentage of papers mentioning each keyword per year, allowing you to:
        - Identify emerging research trends
        - Track the rise or decline of specific topics
        - Compare interest in different concepts over time
        
        Hover over any point on the graph to see the exact percentage for that year.
        """)
    
    trend_keywords = st.text_input("Enter Keywords for Trend Analysis (comma-separated, use | for OR logic)").split(',')
    trend_location = st.selectbox("Select Location for Trend Analysis", options=["abstract", "title"])
    
    if st.button("Generate Trend Analysis"):
        if trend_keywords and any(k.strip() for k in trend_keywords):
            trend_fig = analyze_keyword_trends(trend_keywords, trend_location)
            st.plotly_chart(trend_fig)
        else:
            st.error("Please enter at least one keyword for trend analysis")

elif page == "Author Search":
    # Author Search page
    st.header("Search by :blue[_Author_] 🧑‍🔬")
    with st.expander("How to use Author Search"):
        st.write("""
        This tool allows you to visualize research papers by specific authors within the metabolomics landscape.
        
        **How it works:**
        1. Enter an author's name (first and last name) in the text field
        2. Optionally check "Show Other Embeddings" to display papers by other authors in gray
        3. Click "Search" to generate the visualization
         
        The visualization displays papers authored by the specified researcher as larger blue dots, with hover information 
        showing the paper's title, journal, and publication year. The search algorithm matches the author's last name and 
        first initial against the author lists in our database.
        
        You can also try one of the pre-defined notable researchers in metabolomics by clicking their name buttons below the search field.
        """)

    # Initialize session state variables if they don't exist
    if 'author_search_state' not in st.session_state:
        st.session_state.author_search_state = {
            'author_name': "",
            'show_other': False,
            'search_clicked': False,
            'current_fig': None
        }
    
    # Create form to prevent auto-rerun on every input change
    with st.form(key='author_search_form'):
        author_input = st.text_input("Enter First and Last Name", 
                                    value=st.session_state.author_search_state['author_name'])
        show_other = st.checkbox("Show Other Embeddings", 
                                value=st.session_state.author_search_state['show_other'])
        
        # Create columns for buttons
        col1, col2, col3, col4 = st.columns([1, .35, .25, .25])
        
        with col1:
            search_button = st.form_submit_button("Search")
        
        with col2:
            nicholson_button = st.form_submit_button("Jeremy Nicholson")
            
        with col3:
            fiehn_button = st.form_submit_button("Oliver Fiehn")

        with col4:
            fernie_button = st.form_submit_button("Alisdair Fernie")
    
    # Handle form submission
    if search_button:
        if author_input:
            st.session_state.author_search_state['author_name'] = author_input
            st.session_state.author_search_state['show_other'] = show_other
            st.session_state.author_search_state['search_clicked'] = True
            fig = highlightAuthor(author_input, show_other)
            st.session_state.author_search_state['current_fig'] = fig
        else:
            st.error("Please enter an author name")
    
    elif nicholson_button:
        st.session_state.author_search_state['author_name'] = "Jeremy Nicholson"
        st.session_state.author_search_state['show_other'] = show_other
        st.session_state.author_search_state['search_clicked'] = True
        fig = highlightAuthor("Jeremy Nicholson", show_other)
        st.session_state.author_search_state['current_fig'] = fig
        
    elif fiehn_button:
        st.session_state.author_search_state['author_name'] = "Oliver Fiehn"
        st.session_state.author_search_state['show_other'] = show_other
        st.session_state.author_search_state['search_clicked'] = True
        fig = highlightAuthor("Oliver Fiehn", show_other)
        st.session_state.author_search_state['current_fig'] = fig
        
    elif fernie_button:
        st.session_state.author_search_state['author_name'] = "Alisdair Fernie"
        st.session_state.author_search_state['show_other'] = show_other
        st.session_state.author_search_state['search_clicked'] = True
        fig = highlightAuthor("Alisdair Fernie", show_other)
        st.session_state.author_search_state['current_fig'] = fig
    
    # Display the current figure if it exists
    if st.session_state.author_search_state['search_clicked'] and st.session_state.author_search_state['current_fig'] is not None:
        st.plotly_chart(st.session_state.author_search_state['current_fig'])

# Add the "Share Your Findings" section to all pages except Home
if page != "Home":
    st.subheader(":blue[_Feedback?_] 🔍")
    st.write("Share your [feedback](https://www.linkedin.com/in/obifarin/)")
    # findings = st.text_area("Enter your findings here")

    # if st.button("Submit Findings"):
    #     if findings:
    #         email_sent = send_email("Research Findings", findings, "obifarin3@gatech.edu")
    #         if email_sent:
    #             st.success("Email sent successfully")
    #     else:
    #         st.error("Please enter your findings before submitting")









