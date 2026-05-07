import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from io import StringIO
import traceback
import numpy as np

from utils.json_processor import JSONProcessor
from utils.text_analyzer import TextAnalyzer
from utils.lda import render_lda_tab
from utils.location_analyzer import LocationAnalyzer
from utils.move_step_analysis import render_move_step_tab

# === Page Configuration ===
st.set_page_config(
    page_title="Qualitative Research Dashboard",
    page_icon="icon.png",
    layout="wide"
)

def load_css():
    """Custom CSS to style st.radio as tabs."""
    st.markdown(
        """
        <style>
        div[role="radiogroup"] {
            display: flex;
            gap: 10px;
            margin-bottom: 16px;
            justify-content: center;
        }
        
        .stRadio div[role="radiogroup"] label {
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #f0f2f6;
            border: 1px solid #ccc;
            border-radius: 6px;     /* The rounded box */
            padding: 8px 12px;
            cursor: pointer;
            transition: all 0.1s ease;
            flex: 1; /* Makes buttons equal width */
        }

        .stRadio div[role="radiogroup"] label > div:first-child {
            display: none;
        }
        
        .stRadio div[role="radiogroup"] label > div:last-child {
            color: #31333F;
        }
        
        .stRadio div[role="radiogroup"] label:hover {
            background-color: #e0e0e0;
            border-color: #aaa;
        }
        
        .stRadio div[role="radiogroup"] label:has(input[type="radio"]:checked) {
            background-color: #0068c9;  /* Streamlit's primary blue */
            border-color: #0068c9;
        }
        
        .stRadio div[role="radiogroup"] label:has(input[type="radio"]:checked) > div:last-child {
            color: white;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Call the function to load the CSS ---
load_css()

# === Session State Initialization ===
if 'json_data' not in st.session_state:
    st.session_state.json_data = None
if 'text_fields' not in st.session_state:
    st.session_state.text_fields = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'last_search_keyword' not in st.session_state:
    st.session_state.last_search_keyword = ""
if 'search_keyword' not in st.session_state:
    st.session_state.search_keyword = ""
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0
if 'lda_model' not in st.session_state:
    st.session_state.lda_model = None
if 'lda_dictionary' not in st.session_state:
    st.session_state.lda_dictionary = None
if 'lda_ran_topics' not in st.session_state:
    st.session_state.lda_ran_topics = 0
# --- New Session State for Location Data ---
if 'location_data' not in st.session_state:
    st.session_state.location_data = None
if 'location_analyzer' not in st.session_state:
    st.session_state.location_analyzer = None
# --- FIX: Add state variables for Geo Analysis tab ---
if 'geo_analysis_run' not in st.session_state:
    st.session_state.geo_analysis_run = False
if 'location_df' not in st.session_state:
    st.session_state.location_df = None
# --- NEW: State for current dataset source ---
if 'current_dataset_source' not in st.session_state:
    st.session_state.current_dataset_source = None
# --- NEW: State for cached stats ---
if 'basic_stats' not in st.session_state:
    st.session_state.basic_stats = None
if 'corpus_word_counts' not in st.session_state:
    st.session_state.corpus_word_counts = None


# === Helper Functions ===

def reset_app_state():
    """Reset the application state variables."""
    st.session_state.json_data = None
    st.session_state.text_fields = []
    st.session_state.search_results = []
    st.session_state.last_search_keyword = ""
    st.session_state.search_keyword = ""
    # --- Reset new state ---
    st.session_state.location_data = None
    st.session_state.location_analyzer = None
    # --- FIX: Reset Geo Analysis state variables ---
    st.session_state.geo_analysis_run = False
    st.session_state.location_df = None
    # --- NEW: Reset dataset source ---
    st.session_state.current_dataset_source = None
    # --- NEW: Reset cached stats ---
    st.session_state.basic_stats = None
    st.session_state.corpus_word_counts = None

def update_active_tab():
    """Callback function to update the active tab index from the radio key."""
    st.session_state.active_tab = tab_titles.index(st.session_state.tab_selector)

@st.cache_data
def convert_df_to_csv(df):
    """Caches the conversion of a DataFrame to CSV."""
    return df.to_csv(index=False).encode('utf-8')

# --- NEW: Cached Statistics Functions ---

@st.cache_data
def get_cached_basic_stats(texts_tuple):
    """
    Runs and caches the basic stats calculation.
    Uses a tuple for hashable input.
    """
    texts_list = list(texts_tuple)
    return text_analyzer.get_basic_stats(texts_list)

@st.cache_data
def get_cached_ngrams(texts_tuple, n, remove_stopwords, min_frequency):
    """
    Runs and caches the n-gram calculation.
    Uses a tuple for hashable input.
    """
    texts_list = list(texts_tuple)
    return text_analyzer.extract_multiple_ngrams(
        texts_list,
        n=n,
        remove_stopwords=remove_stopwords,
        min_frequency=min_frequency
    )

@st.cache_data
def get_cached_word_counts(texts_tuple, remove_stopwords=True):
    """
    Runs and caches the full corpus word count calculation (for keyness).
    Uses a tuple for hashable input.
    """
    texts_list = list(texts_tuple)
    return text_analyzer.get_corpus_word_counts(
        texts_list,
        remove_stopwords=remove_stopwords
    )

# --- End of New Cached Functions ---


def create_sankey_diagram(keyword, collocations):
    """
    Creates a Plotly Sankey diagram with a fixed left/right layout.
    """
    labels = [keyword]
    source = []
    target = []
    value = []
    
    # --- Prepare Node Labels ---
    left_nodes = collocations.get('left', [])
    for word, freq in left_nodes:
        labels.append(f"{word}")
        
    right_nodes = collocations.get('right', [])
    for word, freq in right_nodes:
        labels.append(f"{word}")

    # --- Prepare Flows (Source, Target, Value) ---
    keyword_index = 0
    
    for i, (word, freq) in enumerate(left_nodes):
        node_index = i + 1
        source.append(node_index)
        target.append(keyword_index)
        value.append(freq)
        
    offset = len(left_nodes) + 1
    for i, (word, freq) in enumerate(right_nodes):
        node_index = i + offset
        source.append(keyword_index)
        target.append(node_index)
        value.append(freq)

    # --- Set Node Positions ---
    node_x = [0.5]
    node_y = [0.5]
    
    y_left = np.linspace(0.01, 0.99, len(left_nodes)) if left_nodes else []
    for y in y_left:
        node_x.append(0.01)
        node_y.append(y)
        
    y_right = np.linspace(0.01, 0.99, len(right_nodes)) if right_nodes else []
    for y in y_right:
        node_x.append(0.99)
        node_y.append(y)

    # --- Create the Figure ---
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="#1f77b4",
            x=node_x,
            y=node_y,
            hovertemplate='%{label}<extra></extra>'
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            hovertemplate='%{source.label} -> %{target.label}: %{value:.0f}<extra></extra>'
            ),
        textfont=dict(
            size=14,
            color="black" 
        )
    )])

    fig.update_layout(
        title_text=f"Collocation Flow for '<em>{keyword}</em>'",
        font_size=16,
        height=600,
        font=dict(size=16, color="black"),
        margin=dict(l=100, r=100, t=50, b=50)
    )
    
    return fig

def create_collocation_network(keyword, collocations, min_node_size=10, max_node_size=50, min_line_width=1, max_line_width=15):
    """
    Plotly network graph with a fixed left/right layout.
    """
    fig = go.Figure()
    
    nodes_x = [0]
    nodes_y = [0]
    nodes_text = [f"<b>{keyword}</b>"]
    nodes_size = [1] 
    nodes_color = ['#d62728'] # Red for central keyword
    
    edges_x = []
    edges_y = []
    edges_width = []
    
    all_coll_freqs = []
    
    # Process left nodes
    left_nodes = collocations.get('left', [])
    y_left = np.linspace(1, -1, len(left_nodes)) if left_nodes else []
    
    # --- EXPECTS (word, coll_freq) ---
    for i, (word, coll_freq) in enumerate(left_nodes):
        nodes_x.append(-1)
        nodes_y.append(y_left[i])
        nodes_text.append(f"{word}<br>Freq: {coll_freq}")
        nodes_size.append(coll_freq) # Size based on coll_freq
        nodes_color.append('#1f77b4') # Blue for "before"
        
        edges_x.extend([-1, 0, None])
        edges_y.extend([y_left[i], 0, None])
        edges_width.append(coll_freq)
        
        all_coll_freqs.append(coll_freq)

    # Process right nodes
    right_nodes = collocations.get('right', [])
    y_right = np.linspace(1, -1, len(right_nodes)) if right_nodes else []
    
    # --- EXPECTS (word, coll_freq) ---
    for i, (word, coll_freq) in enumerate(right_nodes):
        nodes_x.append(1)
        nodes_y.append(y_right[i])
        nodes_text.append(f"{word}<br>Freq: {coll_freq}")
        nodes_size.append(coll_freq) # Size based on coll_freq
        nodes_color.append('#ff7f0e') # Orange for "after"

        edges_x.extend([1, 0, None])
        edges_y.extend([y_right[i], 0, None])
        edges_width.append(coll_freq)
        
        all_coll_freqs.append(coll_freq)

    # --- Scale Node Sizes ---
    if len(nodes_size) > 1:
        # Scale keyword size to be average
        avg_size = np.mean(nodes_size[1:])
        nodes_size[0] = avg_size
        
        # Normalize sizes for plotting
        min_val = np.min(nodes_size)
        max_val = np.max(nodes_size)
        if max_val - min_val > 0:
            nodes_size = [min_node_size + (x - min_val) * (max_node_size - min_node_size) / (max_val - min_val) for x in nodes_size]
        else:
            nodes_size = [min_node_size for _ in nodes_size]
    else:
        nodes_size = [min_node_size]

    # --- Add Edge Traces ---
    min_freq = min(edges_width) if edges_width else 0
    max_freq = max(edges_width) if edges_width else 0
    
    if max_freq - min_freq > 0:
        norm_widths = [min_line_width + (w - min_freq) * (max_line_width - min_line_width) / (max_freq - min_freq) for w in edges_width]
    else:
        norm_widths = [min_line_width for _ in edges_width]

    edges_x_coords = []
    edges_y_coords = []
    for i in range(0, len(edges_x), 3):
        edges_x_coords.append(edges_x[i:i+2])
        edges_y_coords.append(edges_y[i:i+2])

    for i in range(len(edges_x_coords)):
        fig.add_trace(go.Scatter(
            x=edges_x_coords[i],
            y=edges_x_coords[i],
            mode='lines',
            line=dict(width=norm_widths[i], color='rgba(0,0,0,0.3)'),
            hoverinfo='none'
        ))

    # --- Add Node Trace ---
    fig.add_trace(go.Scatter(
        x=nodes_x,
        y=nodes_y,
        mode='markers+text',
        text=nodes_text,
        marker=dict(
            color=nodes_color,
            size=nodes_size,
            line=dict(width=1, color='Black')
        ),
        hoverinfo='text',
        textposition='middle center'
    ))

    # --- Layout ---
    fig.update_layout(
        title=f"Collocation Network for '<b>{keyword}</b>'",
        height=600,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.5, 1.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
        hovermode='closest'
    )
    
    return fig

# Initialize text analyzer
text_analyzer = TextAnalyzer()

# === Main App UI ===

st.title("Qualitative Research Dashboard")

st.write("""
Select a dataset of academic texts to begin. The app provides tools 
for corpus analysis, keyword search, topic modeling, and move-step analysis.
""")

# --- Sample Data ---
col1, col2, _ = st.columns([1, 1, 2])

with col1:
    sample_data = st.button("Use PubMed Dataset (fast)", type="secondary", use_container_width=True)
with col2:
    sample_data_plos = st.button("Use PLOS Dataset (slower)", type="secondary", use_container_width=True)

# --- Data Loading Logic ---
if sample_data:
    # --- FIX: Reset state when sample button is clicked ---
    reset_app_state() 
    try:
        with open("attached_assets/methods_pubmed_150.json", "r") as f:
            string_data = f.read()
        json_data = JSONProcessor.parse_json(string_data)
        st.session_state.json_data = json_data
        st.session_state.current_dataset_source = "PubMed" # Set source
        
        if isinstance(json_data, list):
            methods_fields = []
            for item in json_data:
                if isinstance(item, dict) and "materials_and_methods" in item:
                    methods_fields.append(item["materials_and_methods"])
            st.session_state.text_fields = methods_fields
        
        # --- Load corresponding location data ---
        try:
            # --- MODIFIED: Updated filename ---
            with open("attached_assets/methods_pubmed_150_locations.json", "r") as f:
                location_data = json.load(f)
            st.session_state.location_data = location_data
            
            # --- Initialize the analyzer ---
            st.session_state.location_analyzer = LocationAnalyzer()
            
            # --- NEW: Pre-calculate and cache stats ---
            with st.spinner("Analyzing corpus stats..."):
                texts_for_cache = tuple(st.session_state.text_fields)
                st.session_state.basic_stats = get_cached_basic_stats(texts_for_cache)
                st.session_state.corpus_word_counts = get_cached_word_counts(texts_for_cache, remove_stopwords=True)

            st.success(f"Successfully loaded sample biomedical dataset with {len(st.session_state.text_fields)} articles and location data.")
            
        except Exception as loc_e:
            st.warning(f"Could not load location data: {str(loc_e)}")
            st.session_state.location_data = None
            st.session_state.location_analyzer = None
            st.success(f"Successfully loaded sample biomedical dataset with {len(st.session_state.text_fields)} articles (location data failed to load).")
        
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        st.error(traceback.format_exc())
        reset_app_state()

elif sample_data_plos: # <-- NEW BLOCK FOR PLOS DATA
    # --- FIX: Reset state when sample button is clicked ---
    reset_app_state() 
    try:
        # NOTE: Assumes 'methods_plos_1612.json' is in the 'attached_assets' folder
        with open("attached_assets/methods_plos_1612.json", "r") as f:
            string_data = f.read()
        json_data = JSONProcessor.parse_json(string_data)
        st.session_state.json_data = json_data
        st.session_state.current_dataset_source = "PLOS" # Set source
        
        if isinstance(json_data, list):
            methods_fields = []
            for item in json_data:
                if isinstance(item, dict) and "materials_and_methods" in item:
                    methods_fields.append(item["materials_and_methods"])
            st.session_state.text_fields = methods_fields
        
        # --- MODIFIED: Load corresponding PLOS location data ---
        try:
            with open("attached_assets/methods_plos_1612_locations.json", "r") as f:
                location_data = json.load(f)
            st.session_state.location_data = location_data
            
            # --- Initialize the analyzer ---
            st.session_state.location_analyzer = LocationAnalyzer()
            
            # --- NEW: Pre-calculate and cache stats ---
            with st.spinner("Analyzing corpus stats..."):
                texts_for_cache = tuple(st.session_state.text_fields)
                st.session_state.basic_stats = get_cached_basic_stats(texts_for_cache)
                st.session_state.corpus_word_counts = get_cached_word_counts(texts_for_cache, remove_stopwords=True)
            
            st.success(f"Successfully loaded sample PLOS dataset with {len(st.session_state.text_fields)} articles and location data.")
        
        except Exception as loc_e:
            st.warning(f"Could not load PLOS location data: {str(loc_e)}")
            st.session_state.location_data = None
            st.session_state.location_analyzer = None
            
            # --- NEW: Pre-calculate and cache stats (even if location fails) ---
            with st.spinner("Analyzing corpus stats..."):
                texts_for_cache = tuple(st.session_state.text_fields)
                st.session_state.basic_stats = get_cached_basic_stats(texts_for_cache)
                st.session_state.corpus_word_counts = get_cached_word_counts(texts_for_cache, remove_stopwords=True)
                
            st.success(f"Successfully loaded sample PLOS dataset with {len(st.session_state.text_fields)} articles (location data failed to load).")
        
    except Exception as e:
        st.error(f"Error loading sample PLOS data: {str(e)}")
        st.error(traceback.format_exc())
        reset_app_state()

# === Main Analysis Interface (Tabs) ===
if st.session_state.json_data is not None:
    
    # Tab titles
    tab_titles = ["Corpus Analysis", "Keyword Search", "LDA Key Themes", "Geographical Analysis", "Move-Step Analysis"]
    
    # Use st.radio to create a "controlled" tab component
    selected_tab = st.radio(
        "Navigate to section:",
        options=tab_titles,
        index=st.session_state.active_tab,
        key="tab_selector",
        horizontal=True,
        label_visibility="collapsed" # Hides the "Navigate to section:" label
    )

    # Update the session state when the user *manually* clicks a tab
    # st.session_state.active_tab = tab_titles.index(selected_tab)
    
    st.markdown("---") # Add a visual separator

    # --- TAB 1: CORPUS ANALYSIS ---
    if selected_tab == "Corpus Analysis":
        st.header("Corpus Analysis Dashboard")
        
        # --- (Code from old "Overview" tab) ---
        st.subheader("Corpus Overview")
        flat_data = JSONProcessor.flatten_json(st.session_state.json_data)
        
        try:
            df = JSONProcessor.json_to_df(st.session_state.json_data)
            st.write(f"Columns: {', '.join(df.columns)}")
            st.write(f"Records: {len(df)}")
            
            with st.expander("Show Sample Data (First 5 Rows)"):
                display_df = df.head(5).astype(str)
                st.dataframe(display_df)
        except Exception as e:
            st.warning(f"Unable to convert JSON to DataFrame: {str(e)}")
            st.subheader("JSON Structure (Flattened)")
            st.json(flat_data)
        
        if st.session_state.json_data and isinstance(st.session_state.json_data, list):
            st.subheader("Text Field Statistics (Methods Sections)")
            
            # --- MODIFICATION START ---
            
            # Get titles and methods for sample display
            titles = []
            methods_texts = [] # Use a different name to avoid confusion
            for article in st.session_state.json_data:
                if isinstance(article, dict):
                    if "article_title" in article and article["article_title"]:
                        titles.append(article["article_title"])
                    elif "title" in article and article["title"]:
                        titles.append(article["title"])
                    if "materials_and_methods" in article and article["materials_and_methods"]:
                        methods_texts.append(article["materials_and_methods"])
            
            # Pull pre-calculated stats from session state
            if 'basic_stats' in st.session_state and st.session_state.basic_stats:
                methods_stats = st.session_state.basic_stats
                
                col1, col2 = st.columns(2)
                with col1:
                    # Use the 'titles' list we just built
                    st.metric("Number of Articles with Titles", f'{len(titles):,}')
                with col2:
                    # Use the pre-calculated stat
                    st.metric("Total Words in Methods", f'{methods_stats.get("total_words", 0):,}')

                col1, col2 = st.columns(2)
                with col1:
                    # Use the pre-calculated stat
                    st.metric("Unique Words in Methods", f'{methods_stats.get("unique_words", 0):,}')
                with col2:
                    # Use the pre-calculated stat
                    st.metric("Average Words per Methods", f'{methods_stats.get("avg_words_per_text", 0):,.2f}')

                col1, col2 = st.columns(2) 
                with col1:
                    # Use the pre-calculated stat
                    st.metric(
                        "Lexical Diversity", 
                        f'{methods_stats.get("lexical_diversity", 0):,.4f}',
                        help="Ratio of unique words to total words. Higher values indicate more diverse vocabulary (ranges from 0 to 1)."
                    )
                
                if "most_frequent_words" in methods_stats and methods_stats["most_frequent_words"]:
                    st.subheader("Most Frequent Words (Excluding Stop Words)")
                    
                    freq_words = methods_stats["most_frequent_words"]
                    freq_df = pd.DataFrame(freq_words, columns=["Word", "Frequency"])
                    freq_df = freq_df.sort_values("Frequency", ascending=False)
                    
                    fig = px.bar(
                        freq_df.head(15).sort_values("Frequency", ascending=True), 
                        y="Word", x="Frequency", orientation='h',
                        title="Top 15 Most Frequent Words", height=500, text="Frequency" 
                    )

                    fig.update_traces(
                        texttemplate='%{text}', textposition='inside',
                        hovertemplate="%{y} (n=%{x})"
                    )

                    fig.update_layout(
                        uniformtext_minsize=14, uniformtext_mode='hide',
                        font=dict(size=16),
                        yaxis=dict(tickfont=dict(size=16)),
                        xaxis=dict(tickfont=dict(size=16)),
                        title=dict(font=dict(size=20))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("Show Top 15 as Table"):
                        st.dataframe(freq_df.head(15), height=500)
                
            else:
                st.warning("Corpus stats could not be loaded. Please try reloading the data.")
            
            # --- MODIFICATION END ---

            if titles or methods_texts:
                with st.expander("Show Sample Text Fields"):
                    if titles:
                        st.write("**Sample Article Titles:**")
                        sample_size = min(3, len(titles))
                        for i in range(sample_size):
                            st.markdown(f"**{i+1}.** {titles[i]}")
                    if methods_texts:
                        st.write("**Sample Methods Sections:**")
                        sample_size = min(3, len(methods_texts))
                        for i in range(sample_size):
                            st.text_area(f"Method {i+1}", methods_texts[i], height=100, key=f"method_{i}")
        
        st.markdown("---")

        # --- N-gram Analysis ---
        st.header("N-gram Analysis")
        st.markdown("""
        <div style="font-size:0.9em; color:gray; margin-top:0px; margin-bottom:20px;">
        ℹ️ N-grams are sequences of <em>n</em> adjacent words. This helps identify common phrases.
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.text_fields:
            st.warning("No text fields found for N-gram analysis.")
        else:
            st.subheader("N-gram Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                n_value = st.slider("N-gram Size", 2, 10, 3, 1, key="ngram_size")
            with col2:
                min_freq = st.slider("Minimum Frequency", 1, 20, 10, 1, key="min_freq")
            with col3:
                remove_stopwords = st.checkbox("Remove Stop Words", True, key="remove_stopwords")
            
            top_n = st.slider("Show Top n-grams", 5, 50, 10, 5, key="top_n")
            
            if st.button("Generate N-grams", key="generate_ngrams"):
                with st.spinner("Analyzing text and generating n-grams..."):
                    
                    # --- MODIFICATION START ---
                    # Use a tuple of texts for caching
                    texts_for_cache = tuple(st.session_state.text_fields)
                    
                    # Call the new cached function
                    ngram_counter = get_cached_ngrams(
                        texts_for_cache,
                        n=n_value,
                        remove_stopwords=remove_stopwords,
                        min_frequency=min_freq
                    )
                    # --- MODIFICATION END ---
                    
                    if ngram_counter:
                        st.subheader(f"Top {top_n} {n_value}-grams")
                        top_ngrams = ngram_counter.most_common(top_n)
                        display_ngrams = [(text_analyzer.format_ngram(ngram), count) for ngram, count in top_ngrams]
                        ngram_df = pd.DataFrame(display_ngrams, columns=["N-gram", "Frequency"])
                        
                        with st.expander("Show data table"):
                            st.dataframe(ngram_df.astype(str), use_container_width=True)
                        
                        fig = px.bar(
                            ngram_df.head(20).sort_values("Frequency"),
                            x="Frequency", y="N-gram", orientation='h',
                            title=f"Top {n_value}-grams by Frequency",
                            labels={"N-gram": f"{n_value}-gram", "Frequency": "Frequency"},
                            height=600, text="Frequency"
                        )
                        fig.update_layout(
                            yaxis_title="", xaxis_title="Frequency",
                            font=dict(size=16), margin=dict(l=20, r=20, t=40, b=20),
                            yaxis=dict(tickfont=dict(size=16))
                        )
                        fig.update_traces(
                            textposition='outside', textfont_size=16, marker_color='#1f77b4'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No {n_value}-grams found with minimum frequency of {min_freq}.")

    # --- TAB 2: KEYWORD SEARCH ---
    elif selected_tab == "Keyword Search":
        st.header("Keyword Search")
        
        # --- Search Controls ---
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            if 'search_keyword' not in st.session_state:
                st.session_state.search_keyword = ""
            
            def search_on_enter():
                if st.session_state.search_keyword:
                    st.session_state.search_results = JSONProcessor.search_json(
                        st.session_state.json_data, 
                        st.session_state.search_keyword,
                        st.session_state.case_sensitive if 'case_sensitive' in st.session_state else False,
                        st.session_state.whole_word if 'whole_word' in st.session_state else False
                    )
                    st.session_state.last_search_keyword = st.session_state.search_keyword
                    st.session_state.search_triggered = True
                    st.session_state.active_tab = 1 
                else:
                    st.warning("Please enter a keyword to search.")
                    st.session_state.search_results = []
            
            keyword = st.text_input(
                "Enter search keyword", 
                key="search_keyword",
                on_change=search_on_enter
            )
        
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            case_sensitive = st.checkbox("Case sensitive", key="case_sensitive")
        with opt_col2:
            whole_word = st.checkbox("Match whole word only", key="whole_word")
        
        if st.button("Search", key="search_button", type="primary"):
            if keyword:
                st.session_state.search_results = JSONProcessor.search_json(
                    st.session_state.json_data, 
                    keyword,
                    case_sensitive,
                    whole_word
                )
                st.session_state.last_search_keyword = keyword
                st.session_state.active_tab = 1 
            else:
                st.warning("Please enter a keyword to search.")
                st.session_state.search_results = []
        
        # --- Search Results Display ---
        if st.session_state.search_results:
            
            # --- Year Visualization ---
            search_term = st.session_state.last_search_keyword if keyword == "" else keyword
            
            st.subheader(f"Keyword '{search_term}' Mentions by Publication Year (n={len(st.session_state.search_results)})")
            
            if isinstance(st.session_state.json_data, list):
                try:
                    article_years = {}
                    for idx, article in enumerate(st.session_state.json_data):
                        if isinstance(article, dict) and "year" in article:
                            article_years[idx] = article["year"]
                    
                    year_matches = []
                    for path, _ in st.session_state.search_results:
                        if path.startswith("[") and "]" in path:
                            idx_str = path[1:path.find("]")]
                            if idx_str.isdigit():
                                idx = int(idx_str)
                                if idx in article_years:
                                    year_matches.append(article_years[idx])
                    
                    if year_matches:
                        year_counts = pd.Series(year_matches).value_counts().reset_index()
                        year_counts.columns = ["Year", "Count"]
                        year_counts = year_counts.sort_values("Year")
                        
                        fig = px.bar(
                            year_counts, 
                            x="Year", 
                            y="Count",
                            title=f"Keyword '{search_term}' Mentions by Publication Year",
                            labels={"Year": "Publication Year", "Count": "Number of Mentions"},
                            height=600,
                            text="Count"  
                        )
                        fig.update_layout(xaxis=dict(type='category', title="Publication Year"),
                                          margin=dict(t=80)                                          
                                          )
                        fig.update_traces(texttemplate='%{text}', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No year data available for visualization.")
                        
                except Exception as e:
                    st.warning(f"Could not create year-based visualization: {str(e)}")
            
            # --- COLLOCATION SANKEY DIAGRAM (st.form) ---
            if search_term:
                match_texts = [value for _, value in st.session_state.search_results if isinstance(value, str)]
                
                if match_texts:
                    st.subheader(f"Collocation Flow for '{search_term}'")
                    
                    with st.form(key="sankey_controls_form"):
                        col_settings1, col_settings2, col_settings3 = st.columns(3)
                        with col_settings1:
                            left_window = st.slider("Words before", 1, 7, 2, key="left_window_size")
                        with col_settings2:
                            right_window = st.slider("Words after", 1, 7, 2, key="right_window_size")
                        with col_settings3:
                            coll_top_n = st.slider("Top results", 5, 20, 10, 1, key="coll_top_n")
                        
                        remove_stopwords_coll = st.checkbox(
                            "Remove Stop Words from Collocates", 
                            value=True, 
                            key="coll_stopwords"
                        )
                        
                        generate_sankey = st.form_submit_button("Generate Diagram")

                    if generate_sankey:
                        with st.spinner("Generating collocation diagram..."):
                            collocations = text_analyzer.find_collocations(
                                match_texts, 
                                search_term.lower(), 
                                left_window_size=left_window,
                                right_window_size=right_window, 
                                top_n=coll_top_n,
                                remove_stopwords=remove_stopwords_coll
                            )
                            
                            if collocations.get('left') or collocations.get('right'):
                                try:
                                    sankey_fig = create_sankey_diagram(
                                        search_term, 
                                        collocations
                                    )
                                    st.plotly_chart(sankey_fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error generating Sankey diagram: {e}")
                                    st.error(traceback.format_exc())
                            else:
                                st.info("No common collocations found for this keyword. Try adjusting the settings.")
                    else:
                        st.info("Adjust the settings above and click 'Generate Diagram' to view the collocation flow.")


            # --- KWIC CONCORDANCE (st.form) ---
            st.subheader("Key Word in Context (KWIC) Concordance")
            
            with st.form(key="kwic_controls_form"):
                kwic_max_results = st.slider("Max Results to Show", 30, 300, 50, 10, key="kwic_max")
                generate_kwic = st.form_submit_button("Show Concordance")
            
            # (Make sure this function is defined somewhere, e.g., near other helpers)
            @st.cache_data
            def convert_df_to_csv(df):
                """Caches the conversion of a DataFrame to CSV."""
                return df.to_csv(index=False).encode('utf-8')

            if st.session_state.text_fields and search_term and generate_kwic:
                with st.spinner("Generating concordance..."):
                    concordance_results = text_analyzer.generate_concordance(
                        st.session_state.text_fields,
                        search_term,
                        window_size=10, 
                        max_results=kwic_max_results,
                        case_sensitive=case_sensitive,
                        whole_word=whole_word
                    )
                    
                    if concordance_results:
                        source_col_name = "Source"
                        
                        csv_data = []
                        kwic_data_display = []
                        
                        for res in concordance_results:
                            try:
                                text_id = res['text_id'] # This is the index
                                article = st.session_state.json_data[text_id]
                                
                                pmid = article.get('PMID', article.get('pmid', 'N/A'))
                                doi = article.get('doi', article.get('DOI', 'N/A'))
                                
                                source_raw = "N/A"
                                source_display = "N/A"
                                
                                if pmid != 'N/A' and str(pmid).isdigit():
                                    pmid_num = str(pmid)
                                    source_raw = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_num}"
                                    source_display = f'<a href="{source_raw}" target="_blank" style="color: #0068c9; text-decoration: none;">{pmid_num}</a>'
                                    source_col_name = "Source (PMID)"
                                elif doi != 'N/A':
                                    doi_str = str(doi)
                                    source_raw = f"https://doi.org/{doi_str}"
                                    source_display = f'<a href="{source_raw}" target="_blank" style="color: #0068c9; text-decoration: none;">{doi_str}</a>'
                                    source_col_name = "Source (DOI)"
                                    
                                csv_data.append({
                                    "Left Context": res['left'],
                                    "Keyword": res['keyword'],
                                    "Right Context": res['right'],
                                    "__source_raw": source_raw
                                })
                                
                                kwic_data_display.append({
                                    "Left Context": res['left'],
                                    "Keyword": res['keyword'],
                                    "Right Context": res['right'],
                                    "__source_display": source_display
                                })
                                
                            except Exception as e:
                                st.warning(f"Error processing KWIC line: {e}")
                                
                        # Finalize column names
                        for row in csv_data:
                            row[source_col_name] = row.pop("__source_raw")
                        for row in kwic_data_display:
                            row[source_col_name] = row.pop("__source_display")
                                
                        # Display as an HTML table to strictly enforce text alignment
                        kwic_df = pd.DataFrame(csv_data)
                        display_df = pd.DataFrame(kwic_data_display)
                        
                        # Generate HTML
                        html_table = display_df.to_html(escape=False, index=False, classes="kwic-table")
                        
                        full_html = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                        <style>
                        body {{
                            margin: 0;
                            padding: 0;
                            font-family: sans-serif;
                        }}
                        .kwic-table {{
                            width: 100%;
                            border-collapse: separate;
                            border-spacing: 0;
                            font-size: 14px;
                            margin-bottom: 20px;
                        }}
                        .kwic-table th {{
                            background-color: #f0f2f6;
                            padding: 10px;
                            text-align: center !important;
                            border-bottom: 1px solid #ccc;
                            border-top: 1px solid #f0f2f6;
                            color: #31333F;
                            position: sticky;
                            top: -1px;
                            z-index: 10;
                        }}
                        .kwic-table td {{
                            padding: 8px 10px;
                            border-bottom: 1px solid #eee;
                            vertical-align: middle;
                        }}
                        /* Left Context */
                        .kwic-table td:nth-child(1) {{
                            text-align: right !important;
                            width: 40%;
                        }}
                        /* Keyword */
                        .kwic-table td:nth-child(2) {{
                            text-align: center !important;
                            font-weight: bold;
                            color: #d62728;
                            white-space: nowrap;
                        }}
                        /* Right Context */
                        .kwic-table td:nth-child(3) {{
                            text-align: left !important;
                            width: 40%;
                        }}
                        /* Source */
                        .kwic-table td:nth-child(4) {{
                            text-align: center !important;
                            white-space: nowrap;
                        }}
                        .kwic-table tr:hover {{
                            background-color: #f9f9f9;
                        }}
                        </style>
                        </head>
                        <body>
                        {html_table}
                        </body>
                        </html>
                        """
                        import streamlit.components.v1 as components
                        components.html(full_html, height=500, scrolling=True)
                        
                        csv = convert_df_to_csv(kwic_df)
                        st.download_button(
                            label="Download Table as CSV",
                            data=csv,
                            file_name=f"kwic_{search_term}.csv",
                            mime="text/csv",
                            key="download_kwic_csv"
                        )
                        
                    else:
                        st.info("No concordance lines found for this keyword.")
            
            elif not generate_kwic:
                st.info("Adjust the settings and click 'Show Concordance' to view the results.")
        
        # --- No results found message ---
        elif keyword and ('search_button' in st.session_state or 'search_triggered' in st.session_state):
            st.info("No results found for your search query.")

    # --- TAB 3: LDA KEY THEMES ---    
    elif selected_tab == "LDA Key Themes":
        render_lda_tab(text_analyzer)

    # --- TAB 4: GEOGRAPHICAL ANALYSIS ---
    elif selected_tab == "Geographical Analysis":
        if 'current_dataset_source' in st.session_state and st.session_state.current_dataset_source:
            st.header(f"Geographical Analysis ({st.session_state.current_dataset_source})")
        else:
            st.header("Geographical Analysis")
        
        # Check if the location analyzer and data are available
        if 'location_analyzer' in st.session_state and st.session_state.location_analyzer and \
           'location_data' in st.session_state and st.session_state.location_data:
            
            # This button's only job is to set the flag
            if st.button("Generate Geographical Analysis", key="generate_geo_analysis", type="primary"):
                st.session_state.geo_analysis_run = True
                st.session_state.location_df = None # Force re-analysis
            
            # All display logic is moved inside this block
            if st.session_state.geo_analysis_run:
                analyzer = st.session_state.location_analyzer
                location_data = st.session_state.location_data
                
                # Only run analysis if df is not cached in state
                if st.session_state.location_df is None:
                    try:
                        with st.spinner("Analyzing location data..."):
                            # 1. Get location counts
                            location_counts = analyzer.get_location_counts(location_data)
                            
                            if not location_counts:
                                st.info("No valid, mappable locations found in the dataset.")
                                st.session_state.geo_analysis_run = False # Reset flag
                                st.stop()
                            
                            # 2. Create the DataFrame
                            location_df = analyzer.create_location_dataframe(location_counts)
                            
                            if location_df.empty:
                                st.info("Location data is available but could not be processed into a mappable format.")
                                st.session_state.geo_analysis_run = False # Reset flag
                                st.stop()
                            
                            # Add Rank Column
                            location_df['Rank'] = np.arange(1, len(location_df) + 1)
                            location_df = location_df[['Rank', 'Location', 'Count', 'Continent']]

                            # 3. Store DataFrame in session state
                            st.session_state.location_df = location_df
                    
                    except Exception as e:
                        st.error(f"Error processing location analysis: {str(e)}")
                        st.error(traceback.format_exc())
                        st.session_state.geo_analysis_run = False # Reset flag on error
                        st.session_state.location_df = None
                        st.stop()
                
                # This block now runs from the cached df
                if st.session_state.location_df is not None:
                    location_df = st.session_state.location_df # Get from state
                    
                    # 3. Create the map
                    fig = analyzer.create_choropleth_map(location_df)
                    
                    st.subheader("Geographic Distribution of Article Mentions")

                    # Create 2-column layout (65% / 35%)
                    col1, col2 = st.columns([0.7, 0.3])

                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.dataframe(
                            location_df,
                            use_container_width=True,
                            height=600, 
                            hide_index=True
                        )
                    
                    # This part will now work correctly on reruns
                    st.subheader("Explore Articles by Location")
                    
                    # Create formatted options for selectbox
                    formatted_options = [
                        f"{row.Location} (n={row.Count})" 
                        for index, row in location_df.iterrows()
                    ]
                    
                    options_map = {
                        f"{row.Location} (n={row.Count})": row.Location 
                        for index, row in location_df.iterrows()
                    }

                    selected_formatted_option = st.selectbox(
                        "Select a location to see matching articles:",
                        options=formatted_options
                    )
                    
                    if selected_formatted_option:
                        selected_location = options_map[selected_formatted_option]
                        
                        # This function now returns 'doi' as well
                        matching_articles = analyzer.find_articles_with_location(
                            location_data,
                            selected_location
                        )
                        
                        if matching_articles:
                            st.write(f"**{len(matching_articles)} articles found for {selected_location}:**")
                            
                            articles_df = pd.DataFrame(matching_articles)
                                                        
                            # 1. Define dynamic variables based on dataset source
                            source_col_header = "Source (PMID)" # Default
                            link_col_name = "source_link"
                            display_regex = r".*\/(\d+)$" # Default for PMID

                            if st.session_state.current_dataset_source == "PLOS":
                                source_col_header = "Source (DOI)"
                                display_regex = r".*org\/(.+)" # Regex to capture DOI
                                # --- THIS IS THE KEY CHANGE ---
                                # Use the 'doi' column fetched from the analyzer
                                # --- FIXED: Added https:// ---
                                articles_df[link_col_name] = articles_df['doi'].apply(
                                    lambda x: f"https://doi.org/{x}" if x != "No DOI" and x != "N/A" else "N/A"
                                )
                            else: # Default for PubMed or Uploaded
                                # --- FIXED: Added https:// ---
                                articles_df[link_col_name] = articles_df['pmid'].apply(
                                    lambda x: f"https://pubmed.ncbi.nlm.nih.gov/{x}" if str(x).isdigit() else "N/A"
                                )
                            
                            # 2. Define column order
                            column_order = ["title", "year", link_col_name]
                            
                            # 3. Create the dataframe with new configs
                            st.dataframe(
                                articles_df[column_order], 
                                column_config={
                                    "title": "Article Title",
                                    # Item 1: Format Year as Text to remove comma
                                    "year": st.column_config.TextColumn("Year"),
                                    # Item 2: Use dynamic header and link config
                                    link_col_name: st.column_config.LinkColumn(
                                        source_col_header,
                                        display_text=display_regex,
                                        width="small"
                                    )
                                },
                                hide_index=True,
                                use_container_width=True
                            )
                            
                        else:
                            st.info(f"No articles found for {selected_location}.")
            
            # This shows if the button hasn't been clicked yet
            else:
                st.info("Click the button to load the geographical analysis.")

        # REPLACED this block for better logic
        elif 'json_data' in st.session_state and st.session_state.json_data is not None:
            st.warning("Geographical analysis is not available for this dataset, as no corresponding location file was provided.")
        
        else:
            st.info("Please load a dataset to view the Geographical Analysis.")

    # --- TAB 5: Move-Step Analysis ---
    # elif selected_tab == "Move-Step Analysis":
    #     st.header("Move-Step Analysis")
    #     st.markdown("Under construction.")
    elif selected_tab == "Move-Step Analysis":
        render_move_step_tab()

# --- App Footer ---
st.markdown("---")
st.markdown("Qualitative Research Dashboard by Charles Lam, University of Leeds <c.lam@leeds.ac.uk>")