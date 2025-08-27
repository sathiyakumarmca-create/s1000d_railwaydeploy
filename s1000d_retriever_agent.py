import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import fitz
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pinecone import Pinecone
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import StateGraph, END
from lxml import etree
from dotenv import load_dotenv

# ================================================================
# --- Global Configuration and Dependencies ---
# ================================================================
load_dotenv()

CONFIG = {
    "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    "LLM_DEPLOYMENT_NAME": os.getenv("LLM_DEPLOYMENT_NAME", "gpt-4o-mini"),
    "EMBEDDING_MODEL_DEPLOYMENT_NAME": os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002"),
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", "medium-blogs-embedding1-1536"),
    "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT")
}


def get_llm():
    return AzureChatOpenAI(
        api_key=CONFIG["AZURE_OPENAI_API_KEY"],
        azure_endpoint=CONFIG["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=CONFIG["LLM_DEPLOYMENT_NAME"],
        api_version=CONFIG["AZURE_OPENAI_API_VERSION"],
        temperature=0
    )


def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_deployment=CONFIG["EMBEDDING_MODEL_DEPLOYMENT_NAME"],
        api_key=CONFIG["AZURE_OPENAI_API_KEY"],
        azure_endpoint=CONFIG["AZURE_OPENAI_ENDPOINT"],
        api_version=CONFIG["AZURE_OPENAI_API_VERSION"]
    )


def get_pinecone_client():
    """Initializes and returns a Pinecone client."""
    return Pinecone(api_key=CONFIG["PINECONE_API_KEY"], environment=CONFIG["PINECONE_ENVIRONMENT"])


# ================================================================
# --- Graph State ---
# ================================================================
class GraphState(TypedDict, total=False):
    """Represents the state of our document processing graph."""
    pdf_path: str
    parsed_json: Optional[List[Dict]]
    extracted_intent: Optional[Dict]
    synthesized_query: Optional[str]
    retrieved_content: Optional[str]  # Corrected to hold retrieved XML
    generated_xml: Optional[str]
    validation_results: Optional[Dict]
    next_step: Optional[str]
    errors: Optional[str]
    retry_count: int


# ================================================================
# --- Agents ---
# ================================================================

# --- agents/pdf_parser.py ---
def pdf_parser_agent(state: GraphState) -> GraphState:
    """Reads a PDF, extracts text, and structures it into a JSON list."""
    logging.info("--> PDFParserAgent: Starting PDF parsing...")
    pdf_path = state.get('pdf_path', None)
    if not pdf_path or not Path(pdf_path).exists():
        logging.error(f"PDFParserAgent: File not found at {pdf_path}")
        return {"errors": f"File not found: {pdf_path}", "next_step": "error_handler"}
    try:
        doc = fitz.open(pdf_path)
        parsed_data = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                parsed_data.append({"page_number": page_num + 1, "content": text})
        doc.close()
        logging.info("✅ PDFParserAgent: Successfully parsed %s pages.", len(parsed_data))
        return {"parsed_json": parsed_data, "next_step": "intent_extraction_agent"}
    except Exception as e:
        logging.error("PDFParserAgent: Failed to parse PDF - %s", e)
        return {"next_step": "error_handler", "errors": f"Parsing failed: {e}"}


# --- agents/intent_extraction.py ---
intent_extraction_prompt = ChatPromptTemplate.from_template("""
You are an expert technical document analyst. Your task is to analyze raw technical text and extract its primary purpose and key technical components.

<Instructions>
1.  **Analyze the text:** Carefully read the content provided in the `Input Text` JSON array.
2.  **Determine the purpose:** What is the main intent of this document? Is it to provide sequential instructions, describe a component, or list parts? Respond with a short, descriptive phrase.
3.  **Identify keywords:** List the most important technical components, actions, part numbers, and terminology.
4.  **Produce the output:** You MUST return a JSON object. This object MUST contain two keys:
    - `"purpose"`: The descriptive phrase you determined.
    - `"keywords"`: A list of the important terms you identified.
5.  **Final JSON Format:** The entire output MUST be a single, well-formed JSON object and nothing else. Do not include any extra text, explanations, or code outside the JSON object itself.
</Instructions>

<Input Text>
{sections}
</Input Text>
""")


def intent_extraction_agent(state: GraphState) -> GraphState:
    """Agent that uses an LLM to extract high-level intent and key entities."""
    logging.info("--> IntentExtractionAgent: Extracting intent and keywords...")
    parsed_json = state.get('parsed_json')
    if not parsed_json:
        logging.error("IntentExtractionAgent: No parsed JSON found in state.")
        return {"next_step": "error_handler", "errors": "No parsed data to extract intent from."}
    try:
        llm_chain = intent_extraction_prompt | get_llm()
        response = llm_chain.invoke({"sections": json.dumps(parsed_json)})
        raw_content = response.content.strip()
        if raw_content.startswith('```json') and raw_content.endswith('```'):
            raw_content = raw_content[7:-3].strip()
        extracted_intent = json.loads(raw_content)
        print(f"extracted_intent------------{extracted_intent}")
        logging.info("✅ IntentExtractionAgent: Successfully extracted purpose and keywords.")
        return {"extracted_intent": extracted_intent, "next_step": "query_synthesis_agent"}
    except (json.JSONDecodeError, OutputParserException) as e:
        logging.error("IntentExtractionAgent: LLM extraction failed due to invalid JSON. Raw response: %s", raw_content)
        return {"next_step": "error_handler", "errors": f"Intent extraction failed: {e}"}
    except Exception as e:
        logging.error("IntentExtractionAgent: An unexpected error occurred - %s", e)
        return {"next_step": "error_handler", "errors": f"Intent extraction failed: {e}"}


# --- agents/query_synthesis.py ---
# --- agents/query_synthesis.py ---
def query_synthesis_agent(state: GraphState) -> GraphState:
    """Agent that synthesizes a highly-effective query string and metadata for retrieval."""
    logging.info("--> QuerySynthesisAgent: Synthesizing search query...")
    extracted_intent = state.get('extracted_intent')

    if not extracted_intent:
        logging.error("QuerySynthesisAgent: No extracted intent found.")
        return {"next_step": "error_handler", "errors": "No intent data for query synthesis."}

    purpose = extracted_intent.get('purpose', 'unspecified purpose')
    keywords = " ".join(extracted_intent.get('keywords', []))

    # New: Create a more targeted query for both text and filtering
    synthesized_query = f"S1000D documentation for {purpose}."

    # Extract key phrases for filtering
    filter_keywords = extracted_intent.get('keywords', [])

    logging.info(f"✅ QuerySynthesisAgent: Synthesized query: '{synthesized_query}' with keywords: {filter_keywords}")

    return {
        "synthesized_query": synthesized_query,
        "query_keywords": filter_keywords,  # New field for keywords
        "next_step": "s1000d_retriever_agent"
    }


# --- agents/s1000d_retriever.py ---
def s1000d_retriever_agent(state: GraphState) -> GraphState:
    """Queries Pinecone for relevant S1000D XML content using hybrid search."""
    logging.info("--> S1000DRetrieverAgent: Retrieving from Pinecone...")
    synthesized_query = state.get('synthesized_query')
    query_keywords = state.get('query_keywords', [])

    if not synthesized_query:
        logging.error("S1000DRetrieverAgent: No synthesized query found in state.")
        return {"next_step": "error_handler", "errors": "No query for retrieval."}

    try:
        embeddings_model = get_embeddings()
        synthesized_query_vector = embeddings_model.embed_query(synthesized_query)
        pc = get_pinecone_client()
        pinecone_idx = pc.Index(CONFIG["PINECONE_INDEX_NAME"])

        # Create a filter based on the extracted purpose or keywords
        # This will narrow down the search to more relevant documents
        # Note: Your ingested data has `purpose` and `xmlElement` in the metadata
        query_filter = {
            "$or": [
                {"purpose": {"$in": query_keywords}},
                {"xmlElement": {"$in": query_keywords}}
            ]
        }

        # A simplified filter could just be on the purpose if the LLM is good at extracting it
        query_filter = {"purpose": {"$in": [state.get('extracted_intent', {}).get('purpose', '')]}}

        results = pinecone_idx.query(
            vector=synthesized_query_vector,
            top_k=5,  # Get more results to find a better match
            filter=query_filter,
            include_metadata=True
        )

        if not results.matches:
            logging.warning("S1000DRetrieverAgent: No relevant documents found after filtering. Trying with no filter.")
            # Fallback to a wider search if filtering fails
            results = pinecone_idx.query(
                vector=synthesized_query_vector,
                top_k=3,
                include_metadata=True
            )
            if not results.matches:
                logging.warning("S1000DRetrieverAgent: No documents found even without a filter.")
                return {"retrieved_content": "No relevant content found.", "next_step": "xml_builder_agent"}

        # Combine the top results into a single string for the LLM
        combined_content = ""
        for match in results.matches:
            combined_content += match.metadata.get('raw_content', '') + "\n---\n"

        logging.info("✅ S1000DRetrieverAgent: Document(s) retrieved successfully.")
        logging.info(f"Top Result Score: {results.matches[0].score:.4f}")
        print(f"combined_content------------>{combined_content}")
        return {"retrieved_content": combined_content, "next_step": "xml_builder_agent"}

    except Exception as e:
        logging.error("S1000DRetrieverAgent: Retrieval failed - %s", e)
        return {"next_step": "error_handler", "errors": f"Schema retrieval failed: {e}"}


# --- agents/xml_builder.py ---
xml_prompt = ChatPromptTemplate.from_template("""
You are an expert in S1000D XML generation. Your task is to convert unstructured technical content into a valid S1000D XML document.

<Instructions>
1.  **Analyze:** Review the `Retrieved XML Examples` and `Classified Sections` to understand the structure and content required.
2.  **Reason Step-by-Step:**
    - I need to create a valid XML document.
    - The root element must be `<dmodule>`.
    - I must add a `<content>` block inside.
    - The content will be structured based on the `Retrieved XML Examples` and the `Classified Sections`.
    - I must ensure all tags are correctly nested and closed.
3.  **Correct Previous Errors (if applicable):**
    - Pay close attention to the `Previous Validation Errors` data.
    - Identify the specific XML syntax issues or missing tags.
    - When generating the new XML, ensure these exact errors are corrected.
4.  **Generate XML:** Create a well-formed S1000D XML document based on your reasoning.
5.  **Final Output:** The entire output MUST be a single, well-formed XML block. Do not include any extra text, explanations, or code outside the XML block itself.
</Instructions>

Retrieved XML Examples:
<examples>
{retrieved_xml_examples}
</examples>
Classified Sections (JSON):
{classified_sections}
Previous Validation Errors:
{previous_errors}
""")


def xml_builder_agent(state: GraphState) -> GraphState:
    """Generates a full S1000D XML document using an LLM."""
    logging.info("--> XMLBuilderAgent: Generating XML...")
    required_fields = ["retrieved_content", "parsed_json"]
    if not all(state.get(field) for field in required_fields):
        missing = [field for field in required_fields if not state.get(field)]
        logging.error("XMLBuilderAgent: Missing required fields: %s", missing)
        return {"next_step": "error_handler", "errors": f"Missing data for XML generation: {missing}"}
    previous_errors = state.get("validation_results", {}).get("errors", "None")
    try:
        llm_chain = xml_prompt | get_llm()
        response = llm_chain.invoke(
            {
                "retrieved_xml_examples": state["retrieved_content"],  # FIX: Pass the correct retrieved content
                "classified_sections": json.dumps(state["parsed_json"]),
                "previous_errors": previous_errors
            }
        )
        generated_xml = response.content.strip()
        if generated_xml.startswith("```xml"):
            generated_xml = generated_xml.lstrip("```xml").rstrip("```").strip()
        print(f"generated_xml------{generated_xml}")
        return {"generated_xml": generated_xml, "next_step": "validator_agent"}
    except Exception as e:
        logging.error("XMLBuilderAgent: LLM generation failed - %s", e)
        return {"next_step": "error_handler", "errors": f"XML generation failed: {e}"}


# --- agents/validator.py ---
def validator_agent(state: GraphState) -> GraphState:
    """
    Validates the generated XML against a schema and returns the XML
    along with the validation results.
    """
    logging.info("--> ValidatorAgent: Validating XML...")
    xml_content = state.get("generated_xml")
    retry_count = state.get("retry_count", 0)
    if not xml_content:
        logging.error("ValidatorAgent: No XML content to validate.")
        return {"next_step": "error_handler", "errors": "No XML content for validation."}
    try:
        # A simple well-formedness check using lxml
        parser = etree.XMLParser(recover=True, no_network=True)
        root = etree.fromstring(xml_content.encode('utf-8'), parser)
        if root is None or parser.error_log:
            errors = [f"Line {err.line}: {err.message}" for err in parser.error_log]
            if not errors:
                errors.append("XML is empty or completely malformed.")
            logging.error("❌ ValidatorAgent: XML failed validation. Retries left: %s. Errors: %s", 3 - retry_count - 1,
                          errors)
            return {
                "validation_results": {"status": "failed", "errors": errors},
                "retry_count": retry_count + 1,
                "next_step": "supervisor_agent",
                "generated_xml": xml_content  # Keep the XML for potential retries
            }
        is_valid = True
        # Check if the root element is `<dmodule>`
        if root.tag != 'dmodule':
            is_valid = False
            logging.warning("❌ ValidatorAgent: Root element is not <dmodule>.")
        if is_valid:
            logging.info("✅ ValidatorAgent: XML passed basic well-formedness check and root element check.")
            return {
                "validation_results": {"status": "success"},
                "generated_xml": xml_content
            }
        else:
            errors = ["Root element is not <dmodule>"]
            logging.error("❌ ValidatorAgent: XML failed validation. Retries left: %s. Errors: %s", 3 - retry_count - 1,
                          errors)
            return {
                "validation_results": {"status": "failed", "errors": errors},
                "retry_count": retry_count + 1,
                "next_step": "supervisor_agent",
                "generated_xml": xml_content
            }
    except Exception as e:
        logging.error("ValidatorAgent: An unexpected XML parsing error occurred - %s", e)
        return {
            "next_step": "supervisor_agent",
            "errors": f"XML parsing failed: {e}",
            "retry_count": retry_count + 1,
            "generated_xml": xml_content
        }


# --- core/supervisor.py ---
def supervisor_agent(state: GraphState) -> GraphState:
    """A supervisor agent that uses an LLM to decide the next step."""
    logging.info("--> SupervisorAgent: Making routing decision...")
    validation_status = state.get("validation_results", {}).get("status")
    if validation_status == "success":
        logging.info("✅ SupervisorAgent: Validation successful. Routing to final step.")
        return {"next_step": "final_step"}
    retry_count = state.get("retry_count", 0)
    MAX_RETRIES = 3
    if validation_status == "failed" and retry_count < MAX_RETRIES:
        logging.info("🔄 SupervisorAgent: Validation failed. Retrying XML generation. Count: %s/%s", retry_count,
                     MAX_RETRIES)
        return {"next_step": "xml_builder_agent"}
    elif validation_status == "failed" and retry_count >= MAX_RETRIES:
        logging.error("❌ SupervisorAgent: Max retries (%s) reached. Aborting.", MAX_RETRIES)
        return {"next_step": "error_handler"}
    tool_names = [
        "pdf_parser_agent",
        "intent_extraction_agent",
        "query_synthesis_agent",
        "s1000d_retriever_agent",
        "xml_builder_agent",
        "validator_agent"
    ]
    supervisor_prompt = PromptTemplate.from_template("""
        You are a highly intelligent and meticulous supervisor for a document transformation pipeline. Your role is to analyze the current state of a task and decide which specialized agent to call next.
        Here is the plan and the available agents:
        1. **pdf_parser_agent**: Extracts raw PDF text into a JSON format. Use this if the 'parsed_json' field is missing.
        2. **intent_extraction_agent**: Extracts the high-level purpose and keywords from the parsed text. Use this if 'extracted_intent' is missing.
        3. **query_synthesis_agent**: Synthesizes a search query based on the extracted intent. Use this if 'synthesized_query' is missing.
        4. **s1000d_retriever_agent**: Retrieves relevant S1000D schema examples. Use this if 'retrieved_content' is missing.
        5. **xml_builder_agent**: Generates the final S1000D XML content. Use this if 'generated_xml' is missing.
        6. **validator_agent**: Validates the generated XML. Use this if 'validation_results' is missing.
        You have access to the following tools: {tools}
        The current state of the task is:
        {state}
        You must choose ONE and ONLY ONE of the available tools based on the current state. Your output MUST be a JSON object with the key 'tool_name' and the value being the name of the tool to call.
        {{
            "tool_name": "the_name_of_the_tool_to_call"
        }}
        Do NOT include any extra text, explanations, or code outside the JSON object.
        """)
    try:
        llm_chain = supervisor_prompt | get_llm()
        response = llm_chain.invoke({"tools": tool_names, "state": state})
        tool_call = json.loads(response.content)
        tool_name = tool_call["tool_name"]
        logging.info("Supervisor decided to call: %s", tool_name)
        return {"next_step": tool_name}
    except Exception as e:
        logging.error("Supervisor decision failed: %s", e, exc_info=True)
        return {"next_step": "error_handler", "errors": f"Supervisor routing failed: {e}"}


# ================================================================
# --- MAIN PIPELINE RUNNER ---
# ================================================================
def create_pipeline() -> StateGraph:
    """Creates and compiles the LangGraph pipeline."""
    workflow = StateGraph(GraphState)
    workflow.add_node("pdf_parser_agent", pdf_parser_agent)
    workflow.add_node("intent_extraction_agent", intent_extraction_agent)
    workflow.add_node("query_synthesis_agent", query_synthesis_agent)
    workflow.add_node("s1000d_retriever_agent", s1000d_retriever_agent)
    workflow.add_node("xml_builder_agent", xml_builder_agent)
    workflow.add_node("validator_agent", validator_agent)
    workflow.add_node("supervisor_agent", supervisor_agent)
    workflow.add_node("error_handler", lambda state: {"next_step": END})
    workflow.set_entry_point("supervisor_agent")
    workflow.add_conditional_edges(
        "supervisor_agent",
        lambda state: state['next_step'],
        {
            "pdf_parser_agent": "pdf_parser_agent",
            "intent_extraction_agent": "intent_extraction_agent",
            "query_synthesis_agent": "query_synthesis_agent",
            "s1000d_retriever_agent": "s1000d_retriever_agent",
            "xml_builder_agent": "xml_builder_agent",
            "validator_agent": "validator_agent",
            "final_step": END,
            "error_handler": "error_handler"
        }
    )
    workflow.add_edge("pdf_parser_agent", "supervisor_agent")
    workflow.add_edge("intent_extraction_agent", "supervisor_agent")
    workflow.add_edge("query_synthesis_agent", "supervisor_agent")
    workflow.add_edge("s1000d_retriever_agent", "supervisor_agent")
    workflow.add_edge("xml_builder_agent", "supervisor_agent")
    workflow.add_edge("validator_agent", "supervisor_agent")
    workflow.add_edge("error_handler", END)
    return workflow.compile()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        llm = get_llm()
        embeddings = get_embeddings()
    except KeyError as e:
        logging.error("Missing environment variable: %s. Please check your .env configuration.", e)
        exit(1)

    # Define the PDF file to process
    pdf_path = r"C:\Users\1000037601\GEAeroSpace_requirement\Bicycle Pre-Ride Inspection Checklist.pdf"

    # Create and run the pipeline
    app = create_pipeline()
    initial_state = {
        "pdf_path": pdf_path,
        "next_step": "pdf_parser_agent",
        "retry_count": 0
    }
    try:
        logging.info("\n--- Starting the S1000D XML Conversion Pipeline ---")
        final_state = app.invoke(initial_state)
        logging.info("\n--- Pipeline Run Complete ---")
        if final_state.get("validation_results", {}).get("status") == "success":
            logging.info("✅ Final Document Status: Valid")
        else:
            logging.info("❌ Final Document Status: Invalid")
            logging.info("Errors: %s", final_state.get('validation_results', {}).get('errors', 'N/A'))
        print("\nFinal State:")
        print(json.dumps(final_state, indent=2))
    except Exception as e:
        logging.error("Pipeline failed with a critical error: %s", e, exc_info=True)