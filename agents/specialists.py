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
Specialists Layer for Context-Aware AI Project Documentation.
Agents here process retrieved facts and format them professionally.
"""


from typing import Dict
from agno.agent import Agent
from utils.ollama_client import ollama_client

class SpecialistFactory:
    """
    Factory to generate Context-Aware Agno Agents.
    """

    @staticmethod
    def create_agent(section_name: str, chapter_role: str = "Senior Technical Ghostwriter", chapter_instruction: str = "") -> Agent:
        """
        Creates an Agent tailored to write a specific section or subsection.
        """
        # Dynamic instruction based on the section and chapter roles
        instruction_detail = chapter_instruction if chapter_instruction else f"Write the content for the '{section_name}' section."
        
        system_prompt = (
            f"You are a {chapter_role}. Your task is to write the '{section_name}' "
            f"section of an engineering project report.\n\n"
            f"DIRECTIVES FOR THIS SECTION:\n{instruction_detail}\n\n"
            "STRICT ACADEMIC RULES:\n"
            "1. FORMAL ACADEMIC TONE: Use third-person passive voice (e.g., 'The model was trained' instead of 'I/We trained the model'). Do not use hyped marketing words.\n"
            "2. EXPLAIN TECHNICAL PRINCIPLES: Anchor specific features and details to the provided facts, but use your deep technical knowledge to explain general architecture principles, UML diagram constructs, database design choices, and software testing theories in complete detail.\n"
            "3. BOLD PLACEHOLDERS: Where appropriate, insert guidance for the student in STRICT BOLD FORMATTING.\n"
            "   - Image Example: **📸 [IMAGE_PLACEHOLDER: Block diagram showing data flow. Caption: Fig 1]**\n"
            "   - Code Example: **💻 [CODE_PLACEHOLDER: Main execution logic in Python]**\n"
            "4. NO INTRODUCTIONS: Do not say 'Here is the section' or 'Based on the facts'. Start writing the content immediately without introductory chatter."
        )

        agent = Agent(
            model=ollama_client.get_chat_model(),
            instructions=[system_prompt],
            markdown=True
        )
        
        return agent

    @classmethod
    def get_all_specialists(cls, subheadings: list) -> Dict[str, Agent]:
        """
        Creates mapping of user-defined subheadings to their respective Agents.
        """
        specialists = {}
        for heading in subheadings:
            specialists[heading] = cls.create_agent(heading)
        return specialists

if __name__ == "__main__":
    test_factory = SpecialistFactory()
    test_agent = test_factory.create_agent("system_design")
    print(f"✅ Agent created successfully for 'system_design'.")