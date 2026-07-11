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
File Reader Parser.
Extracts text directly from in-memory uploaded files (PDF, DOCX, TXT, MD).
Optimized for Streamlit integration.
"""


import PyPDF2
import docx

class FileReader:
    """
    A utility class to parse user-uploaded project documents.
    """

    @staticmethod
    def extract_text(file_obj, filename: str) -> str:
        """
        Detects the file type and routes it to the correct parser.
        """
        # Ensure we read from the beginning of the file object
        if hasattr(file_obj, "seek"):
            try:
                file_obj.seek(0)
            except Exception:
                pass

        ext = filename.split('.')[-1].lower()
        
        try:
            if ext == 'pdf':
                return FileReader._read_pdf(file_obj)
            elif ext == 'docx':
                return FileReader._read_docx(file_obj)
            elif ext in ['md', 'txt']:
                content = file_obj.read()
                if isinstance(content, str):
                    return content
                return content.decode('utf-8', errors='replace')
            else:
                print(f"⚠️ Unsupported file type: {ext}")
                return ""
        except Exception as e:
            print(f"❌ Error reading file {filename}: {e}")
            return ""

    @staticmethod
    def _read_pdf(file_obj) -> str:
        """Extracts text from a PDF file object."""
        reader = PyPDF2.PdfReader(file_obj)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

    @staticmethod
    def _read_docx(file_obj) -> str:
        """Extracts text from a Word document file object."""
        doc = docx.Document(file_obj)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text