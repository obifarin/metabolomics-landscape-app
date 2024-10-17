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

    # Apply keyword filtering
    for keyword in keywords:
        cluster_df.loc[cluster_df[location].apply(lambda x: bool(re.search(keyword, x, re.IGNORECASE))), 'keyword_presence'] = keyword

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
        trace.marker.size = 6  # Slightly larger
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
        trace.marker.size = 6  # Slightly larger
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


def highlightAuthor(author_name, show_only_author=False):
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

    # Add scatter plot for non-highlighted points (other embeddings)
    if not show_only_author:
        other_df = df[~df['highlight']]  # DataFrame of non-highlighted points
        fig.add_trace(go.Scattergl(
            x=other_df['tsne_2D_x'], 
            y=other_df['tsne_2D_y'],
            mode='markers',
            marker=dict(color='rgba(160, 160, 160, 0.5)', size=3),
            name='Other',
            hoverinfo='none'
        ))

    # Add scatter plot for highlighted points
    highlight_df = df[df['highlight']]  # DataFrame of highlighted points
    fig.add_trace(go.Scattergl(
        x=highlight_df['tsne_2D_x'], 
        y=highlight_df['tsne_2D_y'],
        mode='markers',
        marker=dict(
            color='rgb(0, 0, 225)',  # Blue
            size=10,
            line=dict(width=2, color='rgb(0, 0, 100)')
        ),
        name=author_name,
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

# Comment out the email function for now
# def send_email(subject, body, to_email):
#     from_email = "adityasimhadri1@gmail.com"  # Replace with your email
#     from_password = "dixy zovy anvu qufz"  # Replace with your app password

#     msg = MIMEMultipart()
#     msg['From'] = from_email
#     msg['To'] = to_email
#     msg['Subject'] = subject

#     msg.attach(MIMEText(body, 'plain'))

#     try:
#         server = smtplib.SMTP('smtp.gmail.com', 587)
#         server.starttls()
#         server.login(from_email, from_password)
#         text = msg.as_string()
#         server.sendmail(from_email, to_email, text)
#         server.quit()
#         return True
#     except Exception as e:
#         st.error(f"Error sending email: {e}")
#         return False

st.title("Landscape of :red[_Metabolomics_] Research 📄")
st.write("**_Aditya Simhadri_** and [**_Olatomiwa O. Bifarin_**](https://www.linkedin.com/in/obifarin/), [Fernández Lab](https://sites.gatech.edu/fernandez/), Georgia Tech")

page = st.sidebar.selectbox("Choose a page", ["Home", "Embeddings Explorer"])

if page == "Home":
    st.subheader("About the Study")
    st.write("""
    The goal of this work is to visualize and explore trends in metabolomics research. 
             We apply natural language processing techniques to analyze scientific paper 
             abstracts in the field. Using PubMedBERT, a transformer-based encoder model, 
             we generate embeddings—numerical representations of the abstracts. 
             This application allows you to interact with these embeddings. Watch a demo of 
             how to use the app below :video_camera:
             
    """)
    st.video("https://www.youtube.com/watch?v=eHrCx2LhdCk")
elif page == "Embeddings Explorer":
    st.header("Embeddings :blue[_Explorer_] 🌐")
    with st.expander("How to use"):
        
        st.write("""On the sidebar, input the desired research cluster and keywords, and choose whether 
                 to search in abstracts or titles. Click 'Generate Plot' to visualize t-SNE embeddings 
                 colored by publication year and keyword presence. **Note**: If an embedding matches multiple 
                 keywords in the document, the value or color corresponding to the last keyword entered will be shown.""")
        

    st.sidebar.header("Parameters")
    cluster_name = st.sidebar.selectbox("Select Research Cluster", options=["All embeddings"] + list(df['predicted_category'].unique()))
    keywords = st.sidebar.text_input("Enter Keywords (comma-separated)").split(',')
    location = st.sidebar.selectbox("Select Location to Search Keywords", options=["abstract", "title"])

    # Add a checkbox for including only papers with keyword matches
    only_matches = st.sidebar.checkbox("Show only papers with keyword matches", value=False)

    if st.sidebar.button("Generate Plot"):
        if keywords and location:
            fig = clusterByKeywords2(cluster_name, keywords, location, only_matches)
            st.plotly_chart(fig)
        else:
            st.error("Please provide all inputs")

    st.header("Search by :blue[_Author_] 🧑‍🔬")
    with st.expander("Author Search Implementation"):
        st.write("""
        The author search identifies and highlights papers authored by a specified individual on a t-SNE scatter plot. 
                 It extracts the author's last name and first initial, then checks each paper's list of authors for matches. 
                 Papers by the specified author are flagged and assigned a distinct color (blue), while other papers remain gray. 
                 The plot displays larger markers for highlighted papers, with hover text showing the paper's title and journal. 
                 Optional display of only the highlighted points can be toggled, and the plot is customized for clarity and aesthetic appeal. 
                 The function returns the final visual representation.""")

    author_name = st.text_input("Enter First and Last Name")

    show_only_author = st.checkbox("Show only papers by author", value=False)

    if st.button("Search"):
        if author_name:
            fig = highlightAuthor(author_name, show_only_author)
            st.plotly_chart(fig)
        else:
            st.error("Please enter an author name")

    # Example buttons
    st.subheader("Example Authors")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Jeremy Nicholson"):
            fig = highlightAuthor("Jeremy Nicholson", show_only_author)
            st.plotly_chart(fig)

    with col2:
        if st.button("Oliver Fiehn"):
            fig = highlightAuthor("Oliver Fiehn", show_only_author)
            st.plotly_chart(fig)

    with col3:
        if st.button("Alisdair Fernie"):
            fig = highlightAuthor("Alisdair Fernie", show_only_author)
            st.plotly_chart(fig)


    st.header("Share Your :blue[_Findings_] 🔍")
    st.write("Kindly share your findings to this email: obifarin3@gatech.edu")
    # findings = st.text_area("Enter your findings here")

    # if st.button("Submit Findings"):
    #     if findings:
    #         email_sent = send_email("Research Findings", findings, "obifarin3@gatech.edu")
    #         if email_sent:
    #             st.success("Email sent successfully")
    #     else:
    #         st.error("Please enter your findings before submitting")











