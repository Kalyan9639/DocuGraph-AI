import os
import sys
import argparse

# Ensure standard streams are configured to UTF-8
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

# Add current folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.team import ProjectOrchestrator
from utils.doc_generator import doc_exporter
from parsers.file_reader import FileReader
from utils.ollama_client import ollama_client

def main():
    parser = argparse.ArgumentParser(
        description="🎓 AI Project Documentation Architect - CLI Tool\nTransform project notes into professional Word documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--topic", required=True, help="Topic of the project (e.g. 'AI-based Attendance System')")
    parser.add_argument("--subheadings", nargs="+", default=["Abstract", "Introduction", "Literature Survey", "System Design", "Implementation", "Results and Testing", "Conclusion"], help="List of chapter headings")
    parser.add_argument("--context-file", help="Path to text, markdown, or PDF file to load context from")
    parser.add_argument("--pages", type=int, default=20, help="Target minimum pages count (8 to 45)")
    
    args = parser.parse_args()

    # 1. Verify Ollama is running
    print("Checking Ollama connection...")
    if not ollama_client.verify_connection():
        print("❌ Error: Ollama server is not running or the required model is not installed.")
        sys.exit(1)

    # 2. Load context if provided
    context_text = ""
    if args.context_file:
        if os.path.exists(args.context_file):
            print(f"Reading context from {args.context_file}...")
            # Read binary to match file uploader behavior
            with open(args.context_file, "rb") as f:
                context_text = FileReader.extract_text(f, args.context_file)
            print(f"✅ Loaded {len(context_text)} characters of context.")
        else:
            print(f"❌ Error: Context file '{args.context_file}' does not exist.")
            sys.exit(1)

    # 3. Instantiate Orchestrator
    orchestrator = ProjectOrchestrator()

    # 4. Generate Document
    print(f"🚀 Generating document for topic: '{args.topic}' (Target: {args.pages} pages)...")
    final_md = orchestrator.produce_full_document(
        project_topic=args.topic,
        sections_config=args.subheadings,
        context_text=context_text,
        target_pages=args.pages
    )

    # 5. Export to Word
    print("🎨 Applying professional styles and formatting to Word...")
    export_result = doc_exporter.export_word_only(final_md, args.topic)

    if export_result["status"] == "success":
        print(f"🎉 Success! Professional Word document generated at:\n👉 {export_result['file']}")
    else:
        print(f"❌ Export failed: {export_result['message']}")

if __name__ == "__main__":
    main()
