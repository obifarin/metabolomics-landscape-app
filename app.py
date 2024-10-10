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
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Load your dataframe
@st.cache_data
def load_data():
    df = pd.read_excel('metabolomics_landscape_app_01MAR2024.xlsx')
    return df

df = load_data()

def clusterByKeywords2(cluster_name, keywords, location, include_none):
    if cluster_name != 'All embeddings':
        cluster_df = df[df['predicted_category'] == cluster_name]
    else:
        cluster_df = df

    cluster_df['keyword_presence'] = 'None'

    color_scale_time = [
        (0, 'rgb(70, 1, 83)'),  
        (0.5, 'rgb(159, 23, 124)'), 
        (1, 'rgb(253, 231, 37)')  
    ]

    for keyword in keywords:
        cluster_df.loc[cluster_df[location].apply(lambda x: bool(re.search(keyword, x, re.IGNORECASE))), 'keyword_presence'] = keyword

    if not include_none:
        cluster_df = cluster_df[cluster_df['keyword_presence'] != 'None']

    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=("Colored by Year", "Colored by Keyword Presence"),
        vertical_spacing=0.1
    )

    fig_time = px.scatter(cluster_df, x='tsne_2D_x', y='tsne_2D_y', color='pub_year',
                          color_continuous_scale=color_scale_time, opacity=0.7,
                          hover_data=['title'],
                          range_color=[1998, 2024])

    for trace in fig_time['data']:
        fig.add_trace(trace, row=1, col=1)

    fig_keywords = px.scatter(cluster_df, 
                              x='tsne_2D_x', 
                              y='tsne_2D_y', 
                              color='keyword_presence',
                              color_discrete_sequence=px.colors.qualitative.Bold,
                              #px.colors.qualitative.Plotly px.colors.qualitative.Dark24
                              #px.colors.qualitative.Light24 px.colors.qualitative.Alphabet
                              opacity=0.7, 
                              hover_data=['title'])

    for trace in fig_keywords['data']:
        fig.add_trace(trace, row=2, col=1)

    fig.update_traces(marker=dict(size=7, opacity=0.7))

    fig.update_layout(
        title="Embeddings Explorer",
        plot_bgcolor='white',
        height=700, width=1000,
        title_font=dict(size=24, family='Arial, sans-serif', color='#333333'),
        font=dict(size=14, family='Arial, sans-serif', color='#333333'),
        margin=dict(l=50, r=50, t=80, b=50),
        coloraxis=dict(colorscale=color_scale_time, colorbar=dict(title="Year", y=0.85, thickness=15, len=0.3)),
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
        fig.add_trace(go.Scatter(
            x=other_df['tsne_2D_x'], 
            y=other_df['tsne_2D_y'],
            mode='markers',
            marker=dict(color='rgba(160, 160, 160, 0.5)', size=3),  # Light gray, small, and transparent
            name='Other',
            hoverinfo='none'  # No hover information for other points
        ))

    # Add scatter plot for highlighted points
    highlight_df = df[df['highlight']]  # DataFrame of highlighted points
    fig.add_trace(go.Scatter(
        x=highlight_df['tsne_2D_x'], 
        y=highlight_df['tsne_2D_y'],
        mode='markers',
        marker=dict(
            color='rgb(0, 0, 225)',  # Blue
            size=10,  # Larger size for visibility
            line=dict(width=2, color='rgb(0, 0, 100)')  # Darker blue border
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

    return fig  # Return the constructed figure object

def send_email(subject, body, to_email):
    from_email = "adityasimhadri1@gmail.com"  # Replace with your email
    from_password = "dixy zovy anvu qufz"  # Replace with your app password

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

st.title("App: Landscape of Metabolomics Research")

page = st.sidebar.selectbox("Choose a page", ["Home", "Embeddings Explorer"])

if page == "Home":
    st.header("About the study")
    st.write("""
    The goal of this work is to visualize and understand trends in metabolomics research. 
             We use natural language processing techniques to analyze abstracts of scientific 
             papers in the field. A transformer-based encoder model, PubMedBERT, is employed to 
             generate embeddings — numerical representations of the abstracts. With plans to 
             leverage open source LLM models like Llama 3.1 for topic modelling. **This app allows you 
             to interact with the embeddings.** 
             
            Watch the video below to learn more about the study. 
    """)
    st.video("https://www.youtube.com/watch?v=eHrCx2LhdCk")
elif page == "Embeddings Explorer":
    st.header("Embeddings Explorer")
    with st.expander("How to use"):
        
        st.write("""On the sidebar, input the desired research cluster and keywords, and choose whether 
                 to search in abstracts or titles. Click 'Generate Plot' to visualize t-SNE embeddings 
                 colored by publication year and keyword presence. Note: If an embedding matches multiple keywords in the document, the value or color corresponding to the last keyword entered will be shown.""")
        

    with st.expander("Author Search Implementation"):
        
        st.write("""
The author search identifies and highlights papers authored by a specified individual on a t-SNE scatter plot. 
                 It extracts the author's last name and first initial, then checks each paper's list of authors for matches. 
                 Papers by the specified author are flagged and assigned a distinct color (blue), while other papers remain gray. 
                 The plot displays larger markers for highlighted papers, with hover text showing the paper’s title and journal. 
                 Optional display of non-highlighted points can be toggled, and the plot is customized for clarity and aesthetic appeal. 
                 The function returns the final visual representation.""")
    st.sidebar.header("Parameters")
    cluster_name = st.sidebar.selectbox("Select Research Cluster", options=["All embeddings"] + list(df['predicted_category'].unique()))
    keywords = st.sidebar.text_input("Enter Keywords (comma-separated)").split(',')
    location = st.sidebar.selectbox("Select Location to Search Keywords", options=["abstract", "title"])
    include_none = st.sidebar.checkbox("Include Records Without Keywords")

    if st.sidebar.button("Generate Plot"):
        if keywords and location:
            fig = clusterByKeywords2(cluster_name, keywords, location, include_none)
            st.plotly_chart(fig)
        else:
            st.error("Please provide all inputs")

    st.header("Search by Author")

    author_name = st.text_input("Enter First and Last Name")

    show_other = st.checkbox("Show Other Embeddings")

    # Create two buttons next to each other with minimal space
    col1, col2 = st.columns([.25,.75])

    if st.button("Search"):
        if author_name:
            fig = highlightAuthor(author_name, show_other)
            st.plotly_chart(fig)
        else:
            st.error("Please enter an author name")


    with col1:
        button1 = st.button("Facundo Fernandez")
            

    with col2:
        button2 = st.button("Arthur S. Edison")
            
        
    if button1:
        fig = highlightAuthor("Facundo Fernandez", show_other)
        st.plotly_chart(fig)

    if button2:
        fig = highlightAuthor("Arthur Edison", show_other)
        st.plotly_chart(fig)


    st.header("Submit Your Findings")
    findings = st.text_area("Enter your findings here")

    if st.button("Submit Findings"):
        if findings:
            email_sent = send_email("Research Findings", findings, "olatomiwa.bifarin@chemistry.gatech.edu")
            if email_sent:
                st.success("Email sent successfully")
        else:
            st.error("Please enter your findings before submitting")
