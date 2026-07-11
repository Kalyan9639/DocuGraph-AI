import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import streamlit as st
import os
from parsers.file_reader import FileReader
from agents.team import ProjectOrchestrator
from utils.doc_generator import doc_exporter
from utils.ollama_client import ollama_client


# Page Configuration
st.set_page_config(
    page_title="AI Project Doc Architect",
    page_icon="🎓",
    layout="wide"
)

# Custom Professional CSS
st.markdown("""
    <style>
    .main { background-color: #fbfbfb; }
    /* Target only primary action buttons */
    div[data-testid="stButton"] button[kind="primary"] {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4f46e5;
        color: white;
        font-weight: bold;
        font-size: 18px;
        border: none;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #4338ca;
        color: white;
    }
    /* Style small secondary buttons inside expanders to be circular, centered, and premium */
    div[data-testid="stExpander"] div[data-testid="stButton"] button {
        padding: 0px !important;
        width: 36px !important;
        height: 36px !important;
        min-width: 36px !important;
        max-width: 36px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 50% !important;
        border: 1px solid #e5e7eb !important;
        background-color: white !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stExpander"] div[data-testid="stButton"] button:hover {
        background-color: #f3f4f6 !important;
        border-color: #d1d5db !important;
        transform: scale(1.08);
    }
    div[data-testid="stExpander"] div[data-testid="stButton"] button:disabled {
        opacity: 0.4 !important;
        cursor: not-allowed !important;
        transform: none !important;
        background-color: #f9fafb !important;
        border-color: #e5e7eb !important;
    }
    div[data-testid="stExpander"] div[data-testid="stButton"] button p {
        margin: 0px !important;
        padding: 0px !important;
        line-height: 1 !important;
        font-size: 18px !important;
    }
    .step-card {
        padding: 20px;
        border-radius: 12px;
        background-color: white;
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-text {
        font-family: 'Courier New', monospace;
        font-size: 14px;
        color: #4b5563;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    import copy
    st.title("🎓 AI Project Documentation Architect")
    st.markdown("### Transform your raw project notes into a professional B.Tech Thesis.")
    st.divider()

    # Initialize sections in st.session_state if not present
    if "sections" not in st.session_state:
        from prompts.library import DEFAULT_SECTIONS
        st.session_state.sections = copy.deepcopy(DEFAULT_SECTIONS)

    # --- SIDEBAR: SYSTEM CHECK ---
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # User input for API Key
        api_key_input = st.text_input(
            "Ollama Cloud API Key",
            value=st.session_state.get("ollama_api_key", os.getenv("OLLAMA_API_KEY") or (st.secrets.get("OLLAMA_API_KEY", "") if hasattr(st, "secrets") else "")),
            type="password",
            help="Enter your Ollama Cloud API Key here. If already set via secrets or env vars, it will default automatically."
        )
        st.session_state["ollama_api_key"] = api_key_input

        if ollama_client.verify_connection():
            st.success(f"Connected to Ollama Cloud\nModel: {ollama_client.model_name}")
        else:
            st.error("Could not connect to Ollama Cloud. Please enter a valid API key.")
            st.stop() # Stop the app if the engine is missing

        st.divider()
        st.info("This system uses **Granite-Embedding** and **Agno Agents** to generate context-aware reports quickly.")

    # --- TOP INPUTS ---
    col_t1, col_t2 = st.columns([3, 1], gap="medium")
    with col_t1:
        project_topic = st.text_input(
            "Project Topic", 
            placeholder="e.g., AI-based Face Recognition Attendance System"
        )
    with col_t2:
        target_pages = st.slider(
            "Target Page Count",
            min_value=8,
            max_value=45,
            value=20,
            help="Determine the target page count. A larger count dynamically increases word count target per section."
        )

    st.divider()

    # --- MAIN UI LAYOUT ---
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.subheader("📋 Document Structure & Chapter Config")
        st.caption("Customize each chapter's formatting, side-headings (optional), and custom focus area.")
        
        sections = st.session_state.sections
        
        # We display them and capture inputs
        for i, section in enumerate(sections):
            # Expander label
            expander_label = f"📖 Chapter {i+1}: {section['title']} ({section['format']})"
            if section.get("side_headings", "").strip():
                expander_label += f" — Side-headings: {section['side_headings']}"
                
            with st.expander(expander_label, expanded=(i == 0 or i == len(sections)-1)):
                c1, c2 = st.columns([2, 1])
                with c1:
                    new_title = st.text_input(
                        "Chapter Title", 
                        value=section["title"], 
                        key=f"sec_title_{i}"
                    )
                    sections[i]["title"] = new_title
                    
                    new_side_headings = st.text_input(
                        "Side-headings (comma-separated, optional)", 
                        value=section.get("side_headings", ""), 
                        key=f"sec_side_{i}"
                    )
                    sections[i]["side_headings"] = new_side_headings
                
                with c2:
                    new_format = st.selectbox(
                        "Formatting Style",
                        options=["Paragraph", "Mixed", "Bullet Points"],
                        index=["Paragraph", "Mixed", "Bullet Points"].index(section.get("format", "Mixed")),
                        key=f"sec_format_{i}"
                    )
                    sections[i]["format"] = new_format
                
                new_instructions = st.text_area(
                    "Instructions / Guidelines for AI", 
                    value=section.get("instructions", ""), 
                    height=80,
                    key=f"sec_instr_{i}"
                )
                sections[i]["instructions"] = new_instructions
                
                # Control actions (Using icons and consistent alignment)
                btn_cols = st.columns([2, 2, 2, 20])
                up_disabled = (i == 0)
                down_disabled = (i == len(sections) - 1)
                
                if btn_cols[0].button("⬆️", key=f"up_{i}", disabled=up_disabled, help="Move Up"):
                    sections[i], sections[i-1] = sections[i-1], sections[i]
                    st.rerun()
                if btn_cols[1].button("⬇️", key=f"down_{i}", disabled=down_disabled, help="Move Down"):
                    sections[i], sections[i+1] = sections[i+1], sections[i]
                    st.rerun()
                if btn_cols[2].button("🗑️", key=f"del_{i}", help="Delete"):
                    sections.pop(i)
                    st.rerun()

        # Add or Reset buttons
        action_cols = st.columns([1, 1, 3])
        if action_cols[0].button("➕ Add Chapter"):
            sections.append({
                "title": f"Custom Chapter {len(sections) + 1}",
                "format": "Mixed",
                "instructions": "Write detailed analysis of this chapter.",
                "side_headings": ""
            })
            st.rerun()
            
        if action_cols[1].button("🔄 Reset Defaults"):
            from prompts.library import DEFAULT_SECTIONS
            st.session_state.sections = copy.deepcopy(DEFAULT_SECTIONS)
            st.rerun()

    with col2:
        st.subheader("📂 Project Context")
        st.markdown("Upload your README, research papers, or draft docs to guide the AI.")
        
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, TXT, MD)", 
            type=["pdf", "docx", "txt", "md"], 
            accept_multiple_files=True
        )

        if uploaded_files:
            st.write(f"✅ {len(uploaded_files)} files uploaded successfully.")
        
        st.divider()

    st.divider()

    # --- EXECUTION SECTION ---
    if st.button("🚀 Generate Professional Word Document", type="primary"):
        if not project_topic:
            st.error("Please provide a project topic!")
            return

        if not sections:
            st.error("Please add at least one chapter/section!")
            return

        # 1. Extract Context from Files
        all_context = ""
        if uploaded_files:
            with st.status("📖 Extracting context from uploaded files...", expanded=True) as status:
                for file in uploaded_files:
                    st.write(f"Reading {file.name}...")
                    all_context += FileReader.extract_text(file, file.name) + "\n\n"
                status.update(label="Context Extraction Complete!", state="complete")

        # 2. Run the Orchestrator
        try:
            with st.status("🤖 AI Agents are working...", expanded=True) as status:
                orchestrator = ProjectOrchestrator()
                
                st.write("🧠 Building Knowledge Base (RAG)...")
                orchestrator.load_context(all_context)
                
                st.write("✍️ Ghostwriters are drafting sections in parallel...")
                final_md = orchestrator.produce_full_document(
                    project_topic=project_topic, 
                    sections_config=sections, 
                    context_text=all_context,
                    target_pages=target_pages
                )
                status.update(label="Report Generation Complete!", state="complete")

            # 3. Export to Word
            with st.spinner("🎨 Applying professional formatting..."):
                export_result = doc_exporter.export_word_only(final_md, project_topic)

            if export_result["status"] == "success":
                st.balloons()
                st.success("🎉 Your professional project document is ready!")
                
                # Download Button
                with open(export_result["file"], "rb") as f:
                    st.download_button(
                        label="📥 Download Professional Word Doc",
                        data=f,
                        file_name=os.path.basename(export_result["file"]),
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                
                # Preview
                with st.expander("👀 Preview Generated Content"):
                    st.markdown(final_md)
            else:
                st.error(f"Export failed: {export_result['message']}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()