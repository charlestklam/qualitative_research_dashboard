# utils/lda.py
import streamlit as st
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import traceback
import textwrap
import markdown
import requests
import numpy as np # Make sure numpy is imported
import plotly.graph_objects as go # Make sure plotly.graph_objects is imported

# --- LDA COMPUTATION FUNCTION ---
def perform_lda_analysis(text_analyzer, texts, num_topics=10, passes=20, random_state=42):
    """
    Performs LDA analysis and returns HTML visualization, model, and dictionary.
    Includes more detailed error checking.
    """
    if not texts:
        print("LDA Error: Input text list is empty.")
        return None, None, None # Return tuple on failure

    # --- Preprocessing ---
    custom_stop_words_methods = {'qualitative', 'study', 'method', 'methods', 'research', 'among',
                                 'data', 'analysis', 'used', 'using', 'participants', 'results',
                                 '1','2','3','4','5','6','7','8','9','0', '10', 'one','two','n',
                                 'first', 'second', 'third',
                                 'paper', 'approach', 'design', 'based', 'within', 'also'}
    combined_stopwords = text_analyzer.stop_words.union(custom_stop_words_methods)
    original_stopwords = text_analyzer.stop_words
    text_analyzer.stop_words = combined_stopwords
    try:
        tokenized_docs_list = [text_analyzer.preprocess_text(doc, remove_stopwords=True) for doc in texts]
        tokenized_docs = [doc for doc in tokenized_docs_list if doc]
    finally:
        text_analyzer.stop_words = original_stopwords

    if not tokenized_docs:
        print("Warning: No tokens found after preprocessing for LDA.")
        return None, None, None

    # --- Dictionary and Corpus Creation ---
    try:
        print(f"LDA: Creating dictionary from {len(tokenized_docs)} preprocessed documents...")
        dictionary = Dictionary(tokenized_docs)
        print(f"LDA: Initial dictionary size: {len(dictionary)}")

        if not dictionary:
             print("LDA Error: Dictionary is empty BEFORE filtering. Input texts might lack valid words.")
             return None, None, None

        dictionary.filter_extremes(no_below=5, no_above=0.8)
        print(f"LDA: Dictionary size AFTER filtering: {len(dictionary)}")

        if not dictionary:
             print("LDA Error: Dictionary empty AFTER filtering extremes (no_below=5, no_above=0.8).")
             return None, None, None

        corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
        corpus_check = [doc for doc in corpus if doc]
        if not corpus_check:
             print("LDA Error: Corpus empty after creating Bag-of-Words.")
             return None
        print(f"LDA: Corpus created with {len(corpus_check)} non-empty documents.")

        # --- Model Training ---
        print(f"LDA: Training model with {num_topics} topics...")
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            random_state=random_state,
        )
        print("LDA: Model training complete.")

        # --- Visualization ---
        print("LDA: Preparing visualization...")
        if not hasattr(lda_model, 'get_topics') or not lda_model.get_topics().any():
             print("LDA Error: Model trained but seems invalid (no topics found).")
             return None

        vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
        html_string = pyLDAvis.prepared_data_to_html(vis_data)
        print("LDA: Visualization prepared.")

        return html_string, lda_model, dictionary

    except ValueError as ve:
        print(f"LDA ValueError during analysis: {ve}")
        traceback.print_exc()
        return None, None, None
    except Exception as e:
        print(f"Unexpected error during LDA analysis in lda.py: {e}")
        traceback.print_exc()
        return None, None, None


# --- Extract Topic Words ---
def get_topic_words(lda_model, topic_id, num_words=10):
    """Extracts top N words for a given topic ID."""
    try:
        topic_terms = lda_model.show_topic(topic_id, topn=num_words)
        return [word for word, prob in topic_terms]
    except Exception as e:
        print(f"Error getting words for topic {topic_id}: {e}")
        return []


# --- Collapsible Text Component ---
def render_collapsible_text(key, text_content, num_preview_lines=3):
    """
    Renders a collapsible text block with a preview and fade-out,
    using <details> and <summary> to prevent page reruns.
    """
    # Clean up indentation from multiline strings
    text_content = textwrap.dedent(text_content)
    lines = text_content.splitlines()
    
    # Ensure we don't try to preview more lines than exist
    actual_preview_lines = min(num_preview_lines, len(lines))
    preview_text = "\n".join(lines[:actual_preview_lines])
    
    # The rest of the text to be hidden
    rest_of_text = "\n".join(lines[actual_preview_lines:])

    # Convert markdown to HTML for proper rendering
    preview_html = markdown.markdown(preview_text)
    rest_html = markdown.markdown(rest_of_text)

    # Unique class name to style this instance
    html_key = f"collapsible-{key}"

    # New CSS and HTML structure
    css = f"""
    <style>
        /* Main container */
        details.{html_key} {{
            border: 1px solid #e1e4e8;
            border-radius: 8px;
            margin-bottom: 12px;
            overflow: hidden; /* Important */
        }}
        
        /* The summary is the *always visible* part */
        details.{html_key} > summary {{
            display: block; /* Allows styling */
            cursor: pointer;
            padding: 10px 15px;
            list-style: none; /* Hide default arrow */
        }}
        details.{html_key} > summary::-webkit-details-marker {{
            display: none; /* Hide default arrow (Safari) */
        }}
        details.{html_key} > summary:focus {{
            outline: none;
        }}

        /* The preview content, shown only when *closed* */
        details.{html_key}:not([open]) > summary .preview-content {{
            position: relative;
            max-height: 6.5em; /* ~3-4 lines */
            overflow: hidden;
        }}
        
        /* The fade-out effect, shown only when *closed* */
        details.{html_key}:not([open]) > summary .preview-content::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 3em;
            background: linear-gradient(to bottom, transparent, var(--default-backgroundColor, white));
        }}
        
        /* The "Read more/less" text */
        details.{html_key} > summary .toggle-text {{
            display: block;
            text-align: center;
            font-size: 0.9em;
            color: #0068c9;
            font-weight: 500;
            margin-top: 10px;
        }}

        details.{html_key}[open] > summary .toggle-text::before {{
            content: '... Show less';
        }}
        details.{html_key}:not([open]) > summary .toggle-text::before {{
            content: '... Read more';
        }}
        
        /* The full content, shown only when *open* */
        details.{html_key}[open] .full-content {{
            padding: 0 15px 15px 15px;
        }}

        /* Hide the preview div when the details are open */
        details.{html_key}[open] .preview-content {{
            display: none;
        }}
    </style>
    """
    
    html = f"""
    <details class="{html_key}">
        <summary>
            <div class="preview-content">
                {preview_html}
            </div>
            <div class="toggle-text"></div>
        </summary>
        <div class="full-content">
            {rest_html}
        </div>
    </details>
    """

    st.markdown(css, unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)


# --- RENDER TAB FUNCTION ---
def render_lda_tab(text_analyzer):
    st.header("LDA Key Themes Analysis")
    st.write("""
    This section performs Latent Dirichlet Allocation (LDA) topic modeling
    on the 'materials_and_methods' sections of the corpus to identify key themes.
    Adjust the number of topics and click 'Run Analysis'.
    """)
    
    @st.cache_data(show_spinner=False)
    def run_lda_cached(_analyzer_ref_hack, texts_tuple, num_topics):
        texts_list = list(texts_tuple)
        return perform_lda_analysis(text_analyzer, texts_list, num_topics=num_topics)

    with st.form(key="lda_controls_form"):
        # --- MODIFIED LINE ---
        num_topics_lda = st.slider(
            "Select Number of Topics:",
            min_value=3,
            max_value=20,
            value=6,  # <-- Default value changed to 6
            step=1,
            key="lda_num_topics"
        )
        # --- END MODIFICATION ---
        run_lda_button = st.form_submit_button("Run LDA Analysis")

    col1, col2 = st.columns([7, 3])

    with col1:
        # --- Visualization Area ---
        if run_lda_button:
            if st.session_state.text_fields:
                with st.spinner(f"Running LDA for {num_topics_lda} topics..."):
                    texts_for_cache = tuple(st.session_state.text_fields)
                    lda_html_result, lda_model_obj, lda_dict_obj = run_lda_cached(
                        text_analyzer, texts_for_cache, num_topics_lda
                    )
                    st.session_state.lda_model = lda_model_obj
                    st.session_state.lda_dictionary = lda_dict_obj
                    st.session_state.lda_ran_topics = num_topics_lda if lda_model_obj else 0

                if lda_html_result is not None and isinstance(lda_html_result, str):
                    st.subheader(f"Interactive LDA Visualization ({num_topics_lda} Topics)")
                    st.components.v1.html(lda_html_result, width=None, height=870, scrolling=True)
                else:
                    st.error("LDA analysis failed. Check console logs.")
            else:
                st.warning("No 'materials_and_methods' text fields found to analyze.")
        else:
             if st.session_state.lda_model and st.session_state.lda_dictionary:
                 st.info(f"Previously generated LDA model for {st.session_state.lda_ran_topics} topics is loaded. Re-run if needed.")
             else:
                st.info("Adjust the number of topics and click 'Run LDA Analysis'.")

    with col2:
        # --- Explanation Area (Right Column) ---
        st.subheader("Interpreting the LDA Results")

        # 1. Static Explanation
        explain_interpretingLDA = """
        This interactive chart helps you understand the topics discovered in the text data. It uses **Latent Dirichlet Allocation (LDA)**, a statistical model that assumes each document is a mix of topics, and each topic is a mix of words.
        
        **Main Components:**

        1.  **Left Panel: Topic Map** 
            * **Circles:** Each circle represents a **topic**.
            * **Size:** The **size** of the circle indicates the overall **prevalence** of the topic in the entire corpus (bigger = more common).
            * **Distance:** Circles that are **closer together** are more **similar** in their word distributions. Circles far apart are more distinct.

        2.  **Right Panel: Word Bars** 
            * When you **hover over** or **click** a topic circle on the left:
                * This panel shows the **most relevant words** for that selected topic.
                * **Blue Bars:** Represent the **overall frequency** of a word in the entire corpus.
                * **Red Bars:** Represent the **estimated frequency** of a word *within the selected topic*.
            * **Relevance Metric (λ):** The slider labeled 'λ' controls word ranking.
                * **λ = 1:** Ranks by frequency *within this topic*.
                * **λ = 0:** Ranks words by how **distinctive** they are to this topic vs. others.
                * **Recommendation:** Start with λ around 0.6.

        **Interpretation Steps:**

        1.  **Examine Topic Sizes:** Identify the largest circles (dominant themes).
        2.  **Explore Topic Clusters:** Look for groups of circles close together (related themes).
        3.  **Inspect Individual Topics:** Click a circle. Read the top words (red bars). What concept do they suggest? Adjust λ to see both frequent and distinctive words. Try to give the topic a meaningful name.
        4.  **Compare Topics:** Click between different circles to compare their words and understand similarities/differences.        
        """

        st.markdown(explain_interpretingLDA)
        st.markdown("---")  # Add a separator

        st.subheader("Understanding the Axes (PC1 and PC2)")

        explain_understandingPC1PC2 = """
        The horizontal (PC1) and vertical (PC2) axes represent simplified **semantic dimensions** of the topic space, also known as "Principal Components".  
        Think of them as abstract directions that capture how topics **differ in meaning and word usage** across the entire corpus.

        * **PC1 (x-axis)** often reflects the *main distinction* between broad groups of topics —  
        for example, methodological vs. theoretical language, or clinical vs. computational discussions.

        * **PC2 (y-axis)** captures a *secondary layer of variation*,  
        showing subtler contrasts within those groups — for instance, participant-focused vs. procedure-focused writing.

        These axes don't have fixed meanings like "time" or "frequency."
        Instead, they help visualize how **linguistic patterns** vary across topics so you can interpret clusters qualitatively.

        In short: Topics near each other use similar words; topics far apart differ strongly in their language.        
        """
        st.markdown(explain_understandingPC1PC2)
        st.markdown("---")  # Add a separator

    # 2. Placeholder for AI Explanation
    st.subheader("AI Interpretation")
    st.markdown("Under construction.")
