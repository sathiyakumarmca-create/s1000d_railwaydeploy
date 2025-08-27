import streamlit as st
import os
import shutil
import uuid
import logging
import json
import sys

# IMPORTANT: Make sure your retrieveandconvertXMLpipeline.py file is in the same directory.
# This assumes the file contains the necessary functions (create_pipeline, get_llm, etc.)
from s1000d_retriever_agent import create_pipeline,get_llm,get_embeddings,get_pinecone_client

# --- Set up the page configuration and logging ---
st.set_page_config(
    page_title="S1000D Document Pipeline",
    layout="centered",
    initial_sidebar_state="auto"
)

# Use a temporary directory for file uploads
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Set up logging to display messages in the Streamlit console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)


# --- Define the core pipeline function that runs your logic ---
def run_s1000d_pipeline(pdf_path):
    """
    This function encapsulates the logic from your original __main__ block.
    It takes a PDF path and runs the conversion pipeline.
    """
    try:
        # Initialize LLM, Embeddings, and Chroma client
        llm = get_llm()
        embeddings = get_embeddings()
        chroma_client = get_pinecone_client()
    except KeyError as e:
        st.error(f"❌ Missing environment variable: {e}. Please check your .env configuration.")
        return {"xml_output": None, "validation_status": "failed", "error": f"Missing environment variable: {e}"}

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
        validation_results = final_state.get("validation_results", {})
        validation_status = validation_results.get("status", "unknown")

        # Correctly retrieve the generated XML from the final state
        xml_output = final_state.get("generated_xml", "No XML output found.")

        if validation_status == "success":
            logging.info("✅ Final Document Status: Valid")
            return {"xml_output": xml_output, "validation_status": "success"}
        else:
            logging.info("❌ Final Document Status: Invalid")
            logging.info("Errors: %s", validation_results.get('errors', 'N/A'))
            return {"xml_output": xml_output, "validation_status": "failed", "errors": validation_results.get('errors')}

    except Exception as e:
        logging.error("Pipeline failed with a critical error: %s", e, exc_info=True)


# --- Streamlit UI Layout ---
st.title("S1000D Document Pipeline UI")
st.markdown("Upload a PDF document below to transform it into valid S1000D XML.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Use a unique path for the uploaded file
    unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
    temp_pdf_path = os.path.join(UPLOAD_DIR, unique_filename)

    # Save the uploaded file temporarily
    with open(temp_pdf_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)

    if st.button("Process PDF", key="process_button"):
        with st.spinner("Processing document... this may take a moment."):
            # Call the new function that runs your pipeline logic
            result = run_s1000d_pipeline(temp_pdf_path)

            xml_output = result.get('xml_output', None)
            validation_status = result.get('validation_status', None)

            if validation_status == "success" and xml_output:
                st.success("✨ Validation successful! Here is your S1000D XML.")
                st.code(xml_output, language="xml")
            elif xml_output:
                st.warning("⚠️ Processing completed, but validation was not successful.")
                st.code(xml_output, language="xml")
            else:
                st.error(
                    f"❌ The pipeline failed to generate XML output. Reason: {result.get('error', 'Unknown error.')}")

            # You can optionally display detailed errors for debugging
            if validation_status == "failed" and "errors" in result:
                st.subheader("Validation Errors")
                st.json(result["errors"])

            # Clean up the temporary file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
