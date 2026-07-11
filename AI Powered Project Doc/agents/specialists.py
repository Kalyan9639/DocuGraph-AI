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
    def create_agent(section_name: str, chapter_role: str = "Senior Project Engineer", chapter_instruction: str = "", format_style: str = "Mixed") -> Agent:
        """
        Creates an Agent tailored to write a specific section or subsection.
        """
        from prompts.library import UNIVERSAL_RULES
        
        # Dynamic instruction based on the section and chapter roles
        instruction_detail = chapter_instruction if chapter_instruction else f"Write the content for the '{section_name}' section."
        
        system_prompt = (
            f"You are a {chapter_role}. Your task is to write the '{section_name}' "
            f"section of an engineering project report.\n\n"
            f"DIRECTIVES FOR THIS SECTION:\n{instruction_detail}\n\n"
            f"FORMATTING STYLE: {format_style}\n\n"
            f"{UNIVERSAL_RULES}"
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