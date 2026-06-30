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
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4f46e5;
        color: white;
        font-weight: bold;
        font-size: 18px;
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
    st.title("🎓 AI Project Documentation Architect")
    st.markdown("### Transform your raw project notes into a professional B.Tech Thesis.")
    st.divider()

    # --- SIDEBAR: SYSTEM CHECK ---
    with st.sidebar:
        st.header("⚙️ System Status")
        if ollama_client.verify_connection():
            st.success(f"Connected to Ollama\nModel: {ollama_client.model_name}")
        else:
            st.error("Ollama not detected. Please start the Ollama server.")
            st.stop() # Stop the app if the engine is missing

        st.divider()
        st.info("This system uses **Granite-Embedding** and **Agno Agents** to generate context-aware reports quickly.")

    # --- MAIN UI LAYOUT ---
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📝 Project Details")
        
        # 1. Project Topic
        project_topic = st.text_input(
            "Project Topic", 
            placeholder="e.g., AI-based Face Recognition Attendance System"
        )

        # 2. Subheadings (The skeleton)
        st.markdown("**Desired Subheadings**")
        st.caption("Enter each chapter/section on a new line")
        subheadings_input = st.text_area(
            "Sections", 
            value="Abstract\nIntroduction\nLiterature Survey\nSystem Design\nImplementation\nResults and Testing\nConclusion",
            height=200
        )

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
        st.subheader("📏 Document Target Length")
        target_pages = st.slider(
            "Target Minimum Page Count",
            min_value=8,
            max_value=45,
            value=20,
            help="Determine the target page count. A larger count dynamically increases subsection granularity and word counts, taking more time to complete the document."
        )

    st.divider()

    # --- EXECUTION SECTION ---
    if st.button("🚀 Generate Professional Word Document"):
        if not project_topic or not subheadings_input:
            st.error("Please provide both a project topic and the subheadings!")
            return

        # Process inputs
        subheadings = [s.strip() for s in subheadings_input.split('\n') if s.strip()]
        
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
                # This step handles the ChromaDB embedding
                orchestrator.load_context(all_context)
                
                st.write("✍️ Ghostwriters are drafting sections in parallel...")
                # This step handles the Agent generation
                final_md = orchestrator.produce_full_document(
                    project_topic=project_topic, 
                    subheadings=subheadings, 
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