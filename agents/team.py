import os
import sys
import uuid
import hashlib
import threading
from typing import Annotated, TypedDict, List, Dict, Tuple

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

# Verified Agno 2.6.20+ Imports
from agno.knowledge.document import Document
from agno.vectordb.chroma import ChromaDb
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.agent import Agent

# LangGraph Imports
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agents.specialists import SpecialistFactory
from prompts.library import ARCHITECT_PROMPT, SECTION_PROMPTS, get_worker_prompt
from utils.ollama_client import ollama_client

# Define Graph States
class WorkerResult(TypedDict):
    subsection: str
    chapter: str
    content: str

class GraphState(TypedDict):
    topic: str
    target_pages: int
    context_text: str
    blueprint: str
    subheadings: List[str]
    subheadings_map: Dict[str, List[str]]
    chapter_plans: Dict[str, str]
    worker_results: Annotated[List[WorkerResult], operator_add] if 'operator_add' in globals() else List[WorkerResult]
    final_markdown: str

class WorkerState(TypedDict):
    topic: str
    chapter: str
    subsection: str
    blueprint: str
    length_instruction: str
    plan_instruction: str

# Helper custom reducer to merge lists in TypedDict state
import operator
GraphState.__annotations__['worker_results'] = Annotated[List[WorkerResult], operator.add]

class ProjectOrchestrator:
    """
    The Lead Orchestrator utilizing LangGraph Orchestrator-Worker workflow.
    Analyzes user context, designs a plan, dispatches parallel workers, and synthesizes a final document.
    """

    def __init__(self):
        # 1. Initialize the Granite embedding model with correct dimensions (384)
        self.embedder = OllamaEmbedder(id="granite-embedding:30m", dimensions=384)
        
        # 2. Setup local ChromaDB. 
        collection_id = f"project_{uuid.uuid4().hex[:8]}"
        self.vector_db = ChromaDb(
            collection=collection_id,
            path="tmp/chromadb",
            embedder=self.embedder
        )
        self.vector_db.create()
        
        # Concurrency semaphore to prevent Ollama 429 too many concurrent requests errors
        self.concurrency_semaphore = threading.Semaphore(2)

        # Build and compile LangGraph workflow
        workflow = StateGraph(GraphState)
        workflow.add_node("generate_blueprint", self._generate_blueprint_node)
        workflow.add_node("plan_subsections", self._plan_subsections_node)
        workflow.add_node("worker_node", self._worker_node)
        workflow.add_node("synthesize_document", self._synthesize_document_node)

        workflow.add_edge(START, "generate_blueprint")
        workflow.add_edge("generate_blueprint", "plan_subsections")
        workflow.add_conditional_edges("plan_subsections", self._orchestrate_tasks, ["worker_node"])
        workflow.add_edge("worker_node", "synthesize_document")
        workflow.add_edge("synthesize_document", END)

        self.graph = workflow.compile()

    def load_context(self, raw_text: str):
        """
        Chunks the extracted user text and loads it into ChromaDB.
        """
        if not raw_text.strip():
            print("⚠️ No context provided. Agents will generate from baseline knowledge.")
            return

        content_hash = hashlib.md5(raw_text.encode('utf-8')).hexdigest()

        if self.vector_db.content_hash_exists(content_hash):
            print("📚 Context already embedded. Skipping.")
            return

        print("📚 Analyzing document and building Knowledge Base...")
        chunks = [chunk.strip() for chunk in raw_text.split('\n\n') if len(chunk.strip()) > 40]
        documents = [Document(content=chunk) for chunk in chunks]
        self.vector_db.insert(content_hash=content_hash, documents=documents)
        print(f"✅ Embedded {len(documents)} facts into ChromaDB.")

    def _get_facts_for_section(self, subheading: str) -> str:
        """
        Similarity search: Finds the top 4 most relevant chunks for a given heading.
        """
        try:
            results = self.vector_db.search(query=subheading, limit=4)
            if not results:
                return "No specific facts found."
            facts = "\n- ".join([doc.content for doc in results])
            return f"- {facts}"
        except Exception as e:
            print(f"⚠️ RAG Search warning: {e}")
            return "No context available due to search error."

    def _normalize_heading(self, heading: str) -> str:
        h = heading.lower().strip()
        if "abstract" in h:
            return "abstract"
        if "introduction" in h or "motivation" in h:
            return "introduction"
        if "literature" in h or "survey" in h or "existing" in h:
            return "literature_survey"
        if "design" in h or "architecture" in h or "analysis" in h:
            return "system_design"
        if "implementation" in h or "setup" in h:
            return "implementation"
        if "result" in h or "testing" in h or "evaluation" in h:
            return "results_and_testing"
        if "conclusion" in h or "future" in h:
            return "conclusion"
        return ""

    def _decompose_custom_heading(self, heading: str) -> List[str]:
        """
        Decomposes a custom/unknown chapter heading into 3 logical subheadings.
        """
        print(f"🔍 Decomposing custom chapter: '{heading}'...")
        try:
            agent = Agent(
                model=ollama_client.get_chat_model(),
                instructions=["You are an academic outline parser. Split this user-defined chapter heading into 3 logical subheadings for an engineering report. Output only the 3 subheadings, each on a new line. Do not output numbers, bullets, or extra text."]
            )
            res = agent.run(heading)
            lines = [line.strip() for line in res.content.split('\n') if line.strip()]
            return [l for l in lines if not l.startswith('-') and not l.startswith('*')][:4]
        except Exception as e:
            print(f"⚠️ Error decomposing custom heading: {e}")
            return [f"{heading} Overview", f"{heading} Methodology", f"{heading} Summary"]

    # Graph Nodes implementation
    def _generate_blueprint_node(self, state: GraphState) -> Dict:
        print("🏗️ Architect is drawing up the Project Blueprint...")
        try:
            agent = Agent(
                model=ollama_client.get_chat_model(),
                instructions=[ARCHITECT_PROMPT],
                markdown=True
            )
            prompt = f"Project Topic: {state['topic']}\nRaw Context (Reference):\n{state['context_text'][:8000]}"
            res = agent.run(prompt)
            print("✅ Project Blueprint generated successfully.")
            return {"blueprint": res.content}
        except Exception as e:
            print(f"⚠️ Blueprint Generation warning: {e}")
            return {"blueprint": f"Technical Blueprint for {state['topic']}.\nIncludes core modules and software stack using standard technologies."}

    def _plan_subsections_node(self, state: GraphState) -> Dict:
        print("📋 Planning subsections and generating task assignments...")
        subheadings = state["subheadings"]
        target_pages = state["target_pages"]

        # Dynamic subsections mapping based on page targets
        if target_pages < 16:
            standard_subsections = {
                "abstract": ["Abstract Summary"],
                "introduction": ["Background and Objectives", "Problem Statement"],
                "literature_survey": ["Existing Systems and Research Gap"],
                "system_design": ["System Architecture Block Diagram", "Module Description"],
                "implementation": ["Technology Stack and Setup", "Core Implementation Logic"],
                "results_and_testing": ["Testing Strategy and Test Cases Table", "Results Analysis"],
                "conclusion": ["Conclusion and Future Scope"]
            }
        elif target_pages < 28:
            standard_subsections = {
                "abstract": ["Abstract Summary"],
                "introduction": ["Background and Motivation", "Problem Definition", "Project Objectives", "Scope and Exclusions"],
                "literature_survey": ["Review of Existing Methodologies", "Comparative Performance Analysis", "Identified Research Gap"],
                "system_design": ["System Architecture Block Diagram", "Module Description and Responsibility", "Data Flow and UML Diagrams"],
                "implementation": ["Technology Stack Justification", "Core Algorithmic Logic and Math", "Environment Setup and Installation"],
                "results_and_testing": ["Testing Strategy and Plan", "Populated Test Cases Table", "Results and Performance Metrics Analysis"],
                "conclusion": ["Project Conclusion Summary", "Technical Limitations", "Future Scope and Upgrades"]
            }
        else:
            standard_subsections = {
                "abstract": ["Abstract Summary"],
                "introduction": ["Background and Motivation", "Problem Definition", "Project Objectives", "Scope and Exclusions", "Organization of the Report", "Research Methodology Outline"],
                "literature_survey": ["Historical Overview of Domain", "Review of Existing Methodologies", "Comparative Performance Analysis Table", "Identified Research Gap"],
                "system_design": ["System Architecture Block Diagram", "Module Description and Responsibility", "Hardware and Software Specifications", "Data Flow and UML Diagrams", "Database Schema and Entity-Relationship Model"],
                "implementation": ["Technology Stack Justification", "Core Algorithmic Logic and Math", "Core Code Components Walkthrough", "Environment Setup and Installation"],
                "results_and_testing": ["Testing Strategy and Plan", "Populated Test Cases Table", "Unit and Integration Testing Results", "User Acceptance Testing", "Results and Performance Metrics Analysis"],
                "conclusion": ["Project Conclusion Summary", "Technical Limitations", "Future Scope and Upgrades"]
            }

        subheadings_map = {}
        chapter_plans = {}

        for heading in subheadings:
            norm = self._normalize_heading(heading)
            if norm in standard_subsections:
                subs = standard_subsections[norm]
            else:
                subs = self._decompose_custom_heading(heading)
            subheadings_map[norm if norm else heading] = subs
            
            for sub in subs:
                config = SECTION_PROMPTS.get(norm)
                if config:
                    chapter_plans[sub] = config["instruction"]
                else:
                    chapter_plans[sub] = f"Write a comprehensive section detailing '{sub}' for the chapter '{heading}'."

        return {
            "subheadings_map": subheadings_map,
            "chapter_plans": chapter_plans
        }

    def _orchestrate_tasks(self, state: GraphState) -> List[Send]:
        # Calculate dynamic page targets and task list
        total_words_target = state["target_pages"] * 280
        all_tasks = []
        for heading in state["subheadings"]:
            norm = self._normalize_heading(heading)
            subs = state["subheadings_map"].get(norm if norm else heading, [heading])
            for sub in subs:
                all_tasks.append((sub, heading))

        words_per_sub = int(total_words_target / max(1, len(all_tasks)))
        words_per_sub = max(120, words_per_sub)
        length_instruction = (
            f"Write exactly between {int(words_per_sub * 0.9)} and {int(words_per_sub * 1.15)} words for this sub-section. "
            f"Do not pad with unnecessary descriptions, and keep explanations precise and compact."
        )

        sends = []
        for sub, parent in all_tasks:
            plan_instr = state["chapter_plans"].get(sub, f"Write the content for the '{sub}' section of the chapter '{parent}'.")
            sends.append(Send("worker_node", {
                "topic": state["topic"],
                "chapter": parent,
                "subsection": sub,
                "blueprint": state["blueprint"],
                "length_instruction": length_instruction,
                "plan_instruction": plan_instr
            }))

        print(f"🚀 Dispatching {len(sends)} Worker Tasks via LangGraph Send API...")
        return sends

    def _worker_node(self, state: WorkerState) -> Dict:
        subheading = state["subsection"]
        parent_chapter = state["chapter"]
        topic = state["topic"]
        blueprint = state["blueprint"]
        length_instruction = state["length_instruction"]
        plan_instr = state["plan_instruction"]

        # Retrieve RAG facts
        facts = self._get_facts_for_section(subheading)

        try:
            norm_chapter = self._normalize_heading(parent_chapter)
            config = SECTION_PROMPTS.get(norm_chapter)
            role = config["role"] if config else "Senior Technical Ghostwriter"

            agent = SpecialistFactory.create_agent(
                section_name=subheading,
                chapter_role=role,
                chapter_instruction=plan_instr
            )

            worker_prompt = (
                f"Project Topic: {topic}\n"
                f"Main Chapter: {parent_chapter}\n"
                f"Sub-section to write: {subheading}\n\n"
                f"--- SHARED PROJECT BLUEPRINT (Ensure complete consistency with this) ---\n{blueprint}\n\n"
                f"--- FACT SHEET (Ground specific details in these facts) ---\n{facts}\n\n"
                f"--- LENGTH & DEPTH TARGET (Must follow this) ---\n{length_instruction}\n\n"
                f"STRICT FORMATTING RULE: Do NOT include any Markdown headers (like '#', '##', '###') or section titles in your output. "
                f"The assembler handles headings automatically. Start writing the body paragraphs of the section immediately."
            )
            
            with self.concurrency_semaphore:
                response = agent.run(worker_prompt)
            content = response.content
        except Exception as e:
            print(f"❌ Error writing subsection {subheading}: {e}")
            content = f"\n*Error generating content.*"

        return {
            "worker_results": [{
                "subsection": subheading,
                "chapter": parent_chapter,
                "content": content
            }]
        }

    def _synthesize_document_node(self, state: GraphState) -> Dict:
        print("✍️ Synthesizing the final report...")
        worker_results = state["worker_results"]
        subheadings_map = state["subheadings_map"]
        subheadings_list = state["subheadings"]

        results_by_sub = {res["subsection"]: res["content"] for res in worker_results}

        full_markdown = []
        for heading in subheadings_list:
            full_markdown.append(f"# {heading}")
            norm = self._normalize_heading(heading)
            subs = subheadings_map.get(norm if norm else heading, [heading])
            for sub in subs:
                content = results_by_sub.get(sub, "*Content generation failed.*")
                
                # Robustly strip leading headers from agent output
                lines = content.split('\n')
                while lines and (lines[0].strip().startswith('#') or not lines[0].strip()):
                    lines.pop(0)
                cleaned_content = '\n'.join(lines)
                
                full_markdown.append(f"## {sub}\n{cleaned_content}")

        final_doc = "\n\n".join(full_markdown)
        return {"final_markdown": final_doc}

    def produce_full_document(self, project_topic: str, subheadings: List[str], context_text: str = "", target_pages: int = 20) -> str:
        """
        Runs the compiled LangGraph workflow to plan, dispatch parallel workers, and compile the final document.
        """
        # Embed the user's document first (RAG)
        self.load_context(context_text)

        # Initialize Graph state
        initial_state: GraphState = {
            "topic": project_topic,
            "target_pages": target_pages,
            "context_text": context_text,
            "blueprint": "",
            "subheadings": subheadings,
            "subheadings_map": {},
            "chapter_plans": {},
            "worker_results": [],
            "final_markdown": ""
        }

        # Run compiled LangGraph workflow
        final_state = self.graph.invoke(initial_state)
        print("✨ All sections contextually generated and merged.")
        return final_state["final_markdown"]