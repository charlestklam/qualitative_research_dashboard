import streamlit as st
import json
import os

class MoveStepVisualizer:
    # --- CONFIGURATION (Edit colors here) ---
    COLORS = {
        'M1-S1': '#fca5a5', 'M1-S2': '#f87171', 'M1-S3': '#ef4444', 
        'M1-S4': '#dc2626', 'M1-S5': '#b91c1c', 'M1-S6': '#991b1b',
        'M2-S1': '#93c5fd', 'M2-S2': '#60a5fa', 'M2-S3': '#3b82f6', 
        'M2-S4': '#2563eb', 'M2-S5': '#1d4ed8', 'M2-S6': '#1e40af', 'M2-S7': '#1e3a8a',
        'M3-S1': '#86efac', 'M3-S2': '#4ade80', 'M3-S3': '#22c55e',
        'HEADING': '#ffff00', 'OTHER': '#9ca3af'
    }

    @staticmethod
    def get_short_label(label_id):
        if not label_id: return ""
        if label_id == 'HEADING': return 'H'
        if label_id == 'OTHER': return 'O'
        parts = label_id.split('-')
        return parts[-1] if len(parts) > 1 else label_id

    @staticmethod
    def process_data(json_data):
        """Translates the AnnotatedFile[] logic to Python"""
        processed_rows = []
        for file in json_data:
            if not isinstance(file, dict): continue
            
            # Extract Label
            row_label = file.get('articleTitle', file.get('filename', 'Unknown'))
            if len(row_label) > 50: 
                row_label = row_label[:47] + "..."
            
            # Extract PMID
            pmid = file.get('pmid', '')

            # Extract and Sort Segments
            segments = file.get('data', [])
            segments.sort(key=lambda x: x.get('segmentId', 0))

            # Extract Tags using Priority Logic
            row_tags = []
            for seg in segments:
                label_ids = seg.get('labelIds', {})
                primary_tag = "OTHER"
                
                # Priority: M1 > M2 > M3 > Other
                if label_ids.get('move-1'): primary_tag = label_ids['move-1'][0]
                elif label_ids.get('move-2'): primary_tag = label_ids['move-2'][0]
                elif label_ids.get('move-3'): primary_tag = label_ids['move-3'][0]
                elif label_ids.get('other'): primary_tag = label_ids['other'][0]
                
                row_tags.append(primary_tag)
            
            processed_rows.append({
                "label": row_label,
                "pmid": pmid,
                "tags": row_tags,
                "count": len(row_tags)
            })
        return processed_rows

    @classmethod
    def render_html(cls, rows, mode='absolute', show_labels=True, font_size=10, width_pct=100):
        """Generates the HTML string for the visualization"""
        
        # Calculate heights
        row_height = max(20, int(font_size * 2.5))
        cell_font_size = font_size
        
        # CSS Block
        html_content = f"""
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html-to-image/1.11.11/html-to-image.min.js"></script>
        <script>
            function downloadImage(format) {{
                var node = document.getElementById('viz-capture-target');
                var options = {{ backgroundColor: '#ffffff' }};
                
                var func = htmlToImage.toPng;
                if (format === 'jpeg') func = htmlToImage.toJpeg;
                if (format === 'svg') func = htmlToImage.toSvg;
                
                func(node, options)
                    .then(function (dataUrl) {{
                        var link = document.createElement('a');
                        link.download = 'move-step-analysis.' + format;
                        link.href = dataUrl;
                        link.click();
                    }})
                    .catch(function (error) {{
                        console.error('oops, something went wrong!', error);
                        alert('Error generating image: ' + error.message);
                    }});
            }}
        </script>
        <style>
            body {{ margin: 0; padding: 0; }}
            .main-wrapper {{
                width: {width_pct}%;
                margin: 0 auto;
                font-family: sans-serif;
            }}
            
            .controls-bar {{
                display: flex; justify-content: flex-end; gap: 10px;
                margin-bottom: 10px;
                padding: 5px;
                background: #f8f9fa;
                border-radius: 4px;
            }}
            .btn-dl {{
                border: 1px solid #ddd; background: white; 
                padding: 4px 8px; font-size: 12px; cursor: pointer;
                border-radius: 3px;
            }}
            .btn-dl:hover {{ background: #e2e6ea; }}

            .viz-container {{ 
                display: flex; 
                flex-direction: column; 
                gap: 4px; 
                margin-bottom: 10px;
                position: relative;
                background: white; /* Ensure background for export */
                padding: 10px;
            }}
            
            /* Unified scrolling for Standard mode */
            .viz-container.mode-absolute {{
                overflow-x: auto;
                padding-bottom: 12px;
            }}

            .viz-row {{ 
                display: flex; 
                align-items: center; 
                height: {row_height}px; 
                background-color: white;
            }}

            /* In Absolute mode, rows expand to fit content */
            .mode-absolute .viz-row {{
                width: max-content;
                min-width: 100%;
            }}
            /* In Normalized mode, rows are exactly 100% */
            .mode-normalized .viz-row {{
                width: 100%;
            }}

            /* --- COLUMNS --- */
            
            /* 1. PMID Column */
            .pmid-col {{
                width: 80px; min-width: 80px;
                font-size: 11px;
                padding-left: 2px;
                display: flex; align-items: center;
                z-index: 10;
            }}
            .pmid-col a {{
                color: #2563eb; text-decoration: none;
            }}
            .pmid-col a:hover {{
                text-decoration: underline;
            }}

            /* 2. Label Column */
            .row-label {{ 
                width: 200px; min-width: 200px; 
                font-size: 12px; color: #555; 
                text-align: right; padding-right: 12px; 
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                z-index: 10;
            }}

            /* Sticky Logic for Absolute Mode */
            .mode-absolute .pmid-col {{
                position: sticky; 
                left: 0; 
                background: white;
                border-right: 1px solid #f0f0f0;
            }}
            .mode-absolute .row-label {{
                position: sticky; 
                left: 80px; /* Width of pmid-col */
                background: white;
                border-right: 1px solid #ddd;
                box-shadow: 2px 0 5px -2px rgba(0,0,0,0.1); /* Subtle shadow separator */
            }}

            /* 3. Strip Container */
            .strip-container {{ display: flex; height: 100%; align-items: center; }}
            .cell {{ 
                height: {row_height - 2}px; display: flex; align-items: center; justify-content: center; 
                font-size: {cell_font_size}px; color: #111; font-weight: 600; cursor: help;
                transition: opacity 0.2s;
            }}
            .cell:hover {{ opacity: 0.8; }}
            
            /* View Mode Specifics for Strips */
            .mode-absolute .strip-container {{ 
                padding-left: 5px;
            }}
            .mode-absolute .cell {{ width: 30px; flex-shrink: 0; margin-right: 1px; }}
            
            .mode-normalized .strip-container {{ width: 100%; padding-left: 5px; }}
            .mode-normalized .cell {{ width: 100%; flex: 1 1 0px; }}
            
            .legend-container {{
                margin-top: 0px; /* Reduced spacing */
                padding-top: 10px;
                border-top: 1px solid #eee;
                font-size: 12px;
            }}
            .explainer-text {{
                margin-top: 8px;
                padding: 8px;
                background: #f8f9fa;
                border-radius: 4px;
                font-size: 12px;
                color: #555;
                line-height: 1.4;
            }}
        </style>
        <div class="main-wrapper">
            <div class="controls-bar">
                <span style="font-size:12px; align-self:center; color:#666;">Download Image:</span>
                <button class="btn-dl" onclick="downloadImage('jpeg')">JPEG</button>
                <button class="btn-dl" onclick="downloadImage('png')">PNG</button>
                <button class="btn-dl" onclick="downloadImage('svg')">SVG</button>
            </div>
            
            <div id="viz-capture-target">
        """

        # Start Container
        container_class = f"viz-container mode-{mode}"
        html_content += f'<div class="{container_class}">'

        # HTML Generation for Rows
        for row in rows:
            html_content += '<div class="viz-row">'
            
            # 1. PMID Column
            pmid_text = row.get("pmid", "")
            pmid_html = ""
            if pmid_text:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_text}/"
                pmid_html = f'<a href="{url}" target="_blank">{pmid_text}</a>'
            
            html_content += f'<div class="pmid-col" title="PMID: {pmid_text}">{pmid_html}</div>'

            # 2. Label Column
            html_content += f'<div class="row-label" title="{row["label"]}">{row["label"]}</div>'
            
            # 3. DNA Strip
            html_content += '<div class="strip-container">'
            
            total_segs = row['count']
            is_crowded = mode == 'normalized' and total_segs > 30
            should_show_text = show_labels and (mode == 'absolute' or not is_crowded)

            for tag in row['tags']:
                color = cls.COLORS.get(tag, '#cccccc')
                short_code = cls.get_short_label(tag) if should_show_text else ""
                html_content += f'<div class="cell" style="background-color: {color};" title="{tag}">{short_code}</div>'

            html_content += '</div></div>' # End strip / row
            
        html_content += "</div>" # End viz-container
        
        # Add Legend INSIDE the capture target so it downloads with the image
        html_content += cls.render_legend_html()
        
        # Add Explainer Text INSIDE capture target (optional, but helpful in export)
        html_content += """
            <div class="explainer-text">
                <strong>How to read this graph:</strong>
                Each row is an article. Each colored cell is a sentence/segment.
                <strong>Standard View:</strong> Length represents total segment count. 
                <strong>Normalized View:</strong> All articles stretched to 100% width to compare proportions.
                Colors indicate Rhetorical Moves/Steps (Context, Description, Credibility).
            </div>
        """
        
        html_content += "</div>" # End viz-capture-target
        html_content += "</div>" # End main-wrapper
        return html_content

    @classmethod
    def render_legend_html(cls):
        legend_html = '<div class="legend-container" style="display: flex; flex-wrap: wrap; gap: 15px;">'
        groups = {
            "Move 1 (Context)": [k for k in cls.COLORS if k.startswith('M1')],
            "Move 2 (Describing)": [k for k in cls.COLORS if k.startswith('M2')],
            "Move 3 (Credibility)": [k for k in cls.COLORS if k.startswith('M3')],
            "Other": ['HEADING', 'OTHER']
        }
        for group_name, keys in groups.items():
            legend_html += f'<div style="display:flex; align-items:center; gap: 5px;"><strong>{group_name}:</strong>'
            for key in keys:
                color = cls.COLORS[key]
                short = cls.get_short_label(key)
                legend_html += f'<div style="background:{color}; padding: 2px 5px; border-radius:3px; font-weight:bold;">{short}</div>'
            legend_html += '</div>'
        legend_html += '</div>'
        return legend_html

def render_move_step_tab():
    """The main function called by app.py - strictly loads demo file"""
    st.header("Move-Step Analysis (Demo)")
    
    # 1. UI Controls
    # Organized into two rows for better density
    row1_col1, row1_col2, row1_col3 = st.columns([2, 1, 1])
    
    with row1_col1:
        viz_mode = st.radio("View Mode", ["Standard", "Normalized"], horizontal=True, label_visibility="collapsed")
        mode_key = 'absolute' if viz_mode == "Standard" else 'normalized'
    
    with row1_col2:
        show_labels = st.toggle("Show Labels", value=True)
        
    # Second row for visual sliders
    row2_col1, row2_col2 = st.columns([1, 1])
    with row2_col1:
        font_size = st.slider("Font Size (px)", 8, 20, 10)
    with row2_col2:
        width_pct = st.slider("Figure Width (%)", 50, 100, 100)

    # 2. Hardcoded File Loading
    demo_filename = "move_step_analysis_demo_10.json"
    
    # Check if the file is in the attached_assets folder first
    potential_paths = [
        os.path.join("attached_assets", demo_filename),
        demo_filename
    ]
    
    file_path = None
    for path in potential_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    data = []
    
    try:
        if not file_path:
            st.error(f"Configuration Error: The demo file '{demo_filename}' was not found in the application root or 'attached_assets' folder.")
            return

        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            st.error(f"Data Error: The file '{demo_filename}' must contain a list of articles.")
            return

    except Exception as e:
        st.error(f"System Error: Could not load the demo data. {str(e)}")
        return

    # 3. Render
    if data:
        rows = MoveStepVisualizer.process_data(data)
        st.markdown(f"**Analysis of {len(rows)} Documents**")
        
        # Estimate height to avoid inner scrollbars
        # Increased base padding to account for legend and explanation
        height = max(500, len(rows) * 40 + 200)
        
        html = MoveStepVisualizer.render_html(
            rows, 
            mode=mode_key, 
            show_labels=show_labels,
            font_size=font_size,
            width_pct=width_pct
        )
        
        st.components.v1.html(html, height=height, scrolling=True)