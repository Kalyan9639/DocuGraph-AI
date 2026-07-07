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

"""
Document Exporter Bridge.
Strictly converts the final Markdown into a Professional Word Document.
Enforces Size 12 Font and preserves Bold placeholders.
"""


import os
import re
from markdown_parser import MarkdownConverter

class DocExporter:
    """
    Handles the transformation of raw markdown into a polished .docx file.
    """

    def __init__(self, output_base_dir: str = "outputs/final"):
        self.output_base_dir = output_base_dir
        self.converter = MarkdownConverter()
        
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)

    def _get_clean_filename(self, topic: str) -> str:
        """Converts a project topic into a safe filename."""
        clean_name = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
        return clean_name

    def export_word_only(self, md_content: str, project_topic: str) -> dict:
        """
        Generates the Word Document format exclusively.
        """
        folder_name = self._get_clean_filename(project_topic)
        project_dir = os.path.join(self.output_base_dir, folder_name)
        
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

        word_path = os.path.join(project_dir, f"{folder_name}.docx")

        try:
            # Customizing the presentation via your markdown-parser package
            # Enforcing Font Size 12 for standard body text
            custom_styles = {
                'Body': {
                    'size': 12,           # Requirement: 12 font size
                    'font': 'Calibri',    # Standard academic font
                    'spacing': 1.15       # Clean professional spacing
                },
                'H1': {'size': 24, 'font': 'Arial', 'color': (79, 70, 229), 'align': 'CENTER', 'bold': True},
                'H2': {'size': 18, 'font': 'Arial', 'color': (31, 41, 55), 'bold': True}
            }

            # Trigger the custom package to build the Word Doc
            self.converter.to_word(md_content, filename=word_path, custom_styles=custom_styles)

            return {
                "status": "success",
                "file": word_path,
                "folder": project_dir
            }

        except Exception as e:
            print(f"❌ Export Error: {e}")
            return {"status": "error", "message": str(e)}

# Global singleton instance
doc_exporter = DocExporter()