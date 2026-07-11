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

# Streamlit secrets
import streamlit as st

# LangGraph Imports
from langgraph.graph import StateGraph
from langgraph.constants import START, END
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
    sections_config: List[Dict]
    subheadings: List[str]
    subheadings_map: Dict[str, List[str]]
    chapter_plans: Dict[str, Dict]
    worker_results: Annotated[List[WorkerResult], operator_add] if 'operator_add' in globals() else List[WorkerResult]
    final_markdown: str

class WorkerState(TypedDict):
    topic: str
    chapter: str
    subsection: str
    blueprint: str
    length_instruction: str
    plan_instruction: str
    format_style: str

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
        self.embedder = OllamaEmbedder(
            id="granite-embedding:30m", 
            dimensions=384,
            host="https://ollama.com",
            client_kwargs={
                "headers": {"Authorization": f"Bearer {st.secrets['OLLAMA_API_KEY']}"}
            }
        )
        
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

    # Graph Nodes implementation
    def _generate_blueprint_node(self, state: GraphState) -> Dict:
        print("🏗️ Architect is drawing up the Project Blueprint...")
        try:
            agent = Agent(
                model=ollama_client.get_model(),
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
        sections_config = state["sections_config"]

        subheadings = []
        subheadings_map = {}
        chapter_plans = {}

        for sec in sections_config:
            title = sec["title"]
            format_style = sec["format"]
            instructions = sec["instructions"]
            side_headings_raw = sec.get("side_headings", "")

            # Split side-headings by comma
            if side_headings_raw.strip():
                subs = [sh.strip() for sh in side_headings_raw.split(',') if sh.strip()]
            else:
                subs = []

            subheadings.append(title)
            subheadings_map[title] = subs

            if not subs:
                chapter_plans[title] = {
                    "instruction": instructions,
                    "format": format_style
                }
            else:
                for sub in subs:
                    chapter_plans[sub] = {
                        "instruction": f"{instructions}\nWrite specifically about the side-heading: '{sub}' in relation to the main chapter '{title}'.",
                        "format": format_style
                    }

        return {
            "subheadings": subheadings,
            "subheadings_map": subheadings_map,
            "chapter_plans": chapter_plans
        }

    def _orchestrate_tasks(self, state: GraphState) -> List[Send]:
        # Calculate dynamic page targets and task list
        total_words_target = state["target_pages"] * 280
        all_tasks = []
        
        for heading in state["subheadings"]:
            subs = state["subheadings_map"].get(heading, [])
            if not subs:
                all_tasks.append(("", heading))
            else:
                for sub in subs:
                    all_tasks.append((sub, heading))

        words_per_sub = int(total_words_target / max(1, len(all_tasks)))
        words_per_sub = max(150, words_per_sub)
        length_instruction = (
            f"Write exactly between {int(words_per_sub * 0.85)} and {int(words_per_sub * 1.15)} words. "
            f"Keep explanations precise, direct, and focused on implementation details rather than generic textbook theory."
        )

        sends = []
        for sub, parent in all_tasks:
            plan_key = sub if sub else parent
            plan_info = state["chapter_plans"].get(plan_key, {"instruction": "", "format": "Mixed"})
            plan_instr = plan_info["instruction"]
            format_style = plan_info["format"]

            sends.append(Send("worker_node", {
                "topic": state["topic"],
                "chapter": parent,
                "subsection": sub,
                "blueprint": state["blueprint"],
                "length_instruction": length_instruction,
                "plan_instruction": plan_instr,
                "format_style": format_style
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
        format_style = state["format_style"]

        # Retrieve RAG facts
        facts = self._get_facts_for_section(subheading if subheading else parent_chapter)

        try:
            agent = SpecialistFactory.create_agent(
                section_name=subheading if subheading else parent_chapter,
                chapter_role="Senior Project Engineer",
                chapter_instruction=plan_instr,
                format_style=format_style
            )

            worker_prompt = (
                f"Project Topic: {topic}\n"
                f"Main Chapter: {parent_chapter}\n"
                f"Section to write: {subheading if subheading else parent_chapter}\n\n"
                f"--- SHARED PROJECT BLUEPRINT (Ensure complete consistency with this) ---\n{blueprint}\n\n"
                f"--- FACT SHEET (Ground specific details in these facts) ---\n{facts}\n\n"
                f"--- LENGTH & DEPTH TARGET (Must follow this) ---\n{length_instruction}\n\n"
                f"STRICT FORMATTING RULE: Do NOT include any Markdown headers (like '#', '##', '###') or section titles in your output. "
                f"The assembler handles headings automatically. Start writing the content immediately."
            )
            
            with self.concurrency_semaphore:
                response = agent.run(worker_prompt)
            content = response.content
        except Exception as e:
            print(f"❌ Error writing section {subheading if subheading else parent_chapter}: {e}")
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

        results_by_key = {(res["chapter"], res["subsection"]): res["content"] for res in worker_results}

        full_markdown = []
        for heading in subheadings_list:
            full_markdown.append(f"# {heading}")
            subs = subheadings_map.get(heading, [])

            if not subs:
                content = results_by_key.get((heading, ""), "*Content generation failed.*")
                
                # Robustly strip leading headers from agent output
                lines = content.split('\n')
                while lines and (lines[0].strip().startswith('#') or not lines[0].strip()):
                    lines.pop(0)
                cleaned_content = '\n'.join(lines)
                
                full_markdown.append(cleaned_content)
            else:
                for sub in subs:
                    content = results_by_key.get((heading, sub), "*Content generation failed.*")
                    
                    # Robustly strip leading headers from agent output
                    lines = content.split('\n')
                    while lines and (lines[0].strip().startswith('#') or not lines[0].strip()):
                        lines.pop(0)
                    cleaned_content = '\n'.join(lines)
                    
                    full_markdown.append(f"## {sub}\n{cleaned_content}")

        final_doc = "\n\n".join(full_markdown)
        return {"final_markdown": final_doc}

    def produce_full_document(self, project_topic: str, sections_config: List[Dict], context_text: str = "", target_pages: int = 20) -> str:
        """
        Runs the compiled LangGraph workflow to plan, dispatch parallel workers, and compile the final document.
        """
        from prompts.library import DEFAULT_SECTIONS

        normalized_sections_config = []
        if isinstance(sections_config, list) and len(sections_config) > 0 and isinstance(sections_config[0], str):
            for heading in sections_config:
                found = False
                for ds in DEFAULT_SECTIONS:
                    if ds["title"].lower() == heading.lower():
                        normalized_sections_config.append(ds)
                        found = True
                        break
                if not found:
                    normalized_sections_config.append({
                        "title": heading,
                        "format": "Mixed",
                        "instructions": f"Write a section about {heading}.",
                        "side_headings": ""
                    })
        else:
            normalized_sections_config = sections_config

        # Embed the user's document first (RAG)
        self.load_context(context_text)

        # Initialize Graph state
        initial_state: GraphState = {
            "topic": project_topic,
            "target_pages": target_pages,
            "context_text": context_text,
            "blueprint": "",
            "sections_config": normalized_sections_config,
            "subheadings": [],
            "subheadings_map": {},
            "chapter_plans": {},
            "worker_results": [],
            "final_markdown": ""
        }

        # Run compiled LangGraph workflow
        final_state = self.graph.invoke(initial_state)
        print("✨ All sections contextually generated and merged.")
        return final_state["final_markdown"]