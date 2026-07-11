"""
Prompts Library for AI Project Documentation Generator.
Tailored for Indian B.Tech/M.Tech university standards.
"""

# --- UNIVERSAL WRITING RULES ---
UNIVERSAL_RULES = """
STRICT WRITING GUIDELINES:
1. SIMPLE & DIRECT LANGUAGE: Write in plain, clear, and professional English. Avoid complex textbook theory or generic definitions. Keep it focused entirely on what was implemented and achieved in this project.
2. NO CHATTER: Start writing the content immediately. Do not write introductory statements like "Here is the section" or "In this chapter we will discuss".
3. ACTIVE PROJECT FOCUS: Do not explain generic concepts (e.g., do not define what Python or a database is). Instead, write how they were used to build the specific components of the project.
4. BOLD PLACEHOLDERS: Where appropriate, insert guidance/placeholders for the student in strict bold formatting:
   - For Images/Diagrams: **📸 [IMAGE_PLACEHOLDER: <Detailed description of the image/diagram to insert>. Caption: Figure X: Name]**
   - For Code: **💻 [CODE_PLACEHOLDER: <Core logic or algorithm in programming language>. File: <Name>. Language: <Lang>]**
   - For Citations: **📄 [CITATION_PLACEHOLDER: <Author, Title, IEEE format>]**
"""

# --- DEFAULT SECTIONS CONFIGURATION ---
DEFAULT_SECTIONS = [
    {
        "title": "Abstract",
        "format": "Paragraph",
        "instructions": "Summarize the project's goal, the problem it solves, the methodology used, and the final results/outcome.",
        "side_headings": ""
    },
    {
        "title": "Introduction",
        "format": "Mixed",
        "instructions": "Introduce the project domain, define the problem statement, state the project objectives, and define the scope.",
        "side_headings": "Background, Problem Statement, Objectives, Project Scope"
    },
    {
        "title": "Literature Survey",
        "format": "Mixed",
        "instructions": "Compare existing systems or approaches. Briefly explain what other works did, and highlight the research gap.",
        "side_headings": "Existing Systems, Comparative Analysis, Research Gap"
    },
    {
        "title": "System Design",
        "format": "Mixed",
        "instructions": "Detail the architecture, system flow, and modules of the project. Mention the software/hardware stack.",
        "side_headings": "System Architecture, Module Description, Data Flow"
    },
    {
        "title": "Implementation",
        "format": "Mixed",
        "instructions": "Describe how the project was actually built, code modules, environment setup, and implementation logic.",
        "side_headings": "Technology Stack, Core Logic, Setup and Configuration"
    },
    {
        "title": "Results and Testing",
        "format": "Mixed",
        "instructions": "Present the testing strategy, test cases, and analyze the actual outcomes and performance of the system.",
        "side_headings": "Testing Strategy, Test Cases, Results Analysis"
    },
    {
        "title": "Conclusion",
        "format": "Paragraph",
        "instructions": "Summarize what was achieved, list any technical limitations, and suggest future upgrades.",
        "side_headings": "Conclusion Summary, Limitations, Future Scope"
    }
]

# Keep SECTION_PROMPTS for backward compatibility (e.g. if any old files import it)
SECTION_PROMPTS = {}
for s in DEFAULT_SECTIONS:
    key = s["title"].lower().replace(" ", "_")
    SECTION_PROMPTS[key] = {
        "role": "Senior Project Engineer",
        "instruction": s["instructions"]
    }

def get_worker_prompt(section_key: str) -> str:
    """
    Constructs a full system prompt for a specific worker agent. (Fallback/Backward compatibility)
    """
    # Simple fallback using universal guidelines
    return (
        f"ROLE: Senior Project Engineer\n"
        f"TASK: Write the section: {section_key}.\n"
        f"{UNIVERSAL_RULES}"
    )

# --- SYSTEM ARCHITECT PROMPT ---
ARCHITECT_PROMPT = """
You are a Senior Systems Architect and Advisor. Your task is to analyze the Project Topic and any provided raw context, and output a comprehensive, cohesive Technical Project Blueprint.

This blueprint will serve as the single source of truth to synchronize all Document Modules of a final-year engineering thesis.

Outlines you MUST generate in the blueprint:
1. CORE TECHNICAL MODULES (3-4 distinct modules with exact names and responsibilities).
2. HARDWARE & SOFTWARE SPECIFICATIONS (Exact stack: languages, libraries/frameworks, database engine, hosting/deployment details).
3. SYSTEM WORKFLOW / CORE ALGORITHM (The primary data flows, step-by-step logic, or standard algorithms to be implemented).
4. SYSTEM METRICS & EXPECTED OUTPUTS (What metrics are used to measure success, e.g. Accuracy, F1-Score, Latency, and realistic target values).

Strict Guidelines:
- Do not output generic code.
- Write in a highly technical, concrete academic style.
- Output ONLY the Markdown blueprint (no introductions or meta-talk).
"""