# -*- coding: utf-8 -*-
"""
unstructured_langgraph_pipeline.py

This module implements a document processing and chunking pipeline using the
Unstructured library and LangGraph. It partitions documents (PDF, DOCX),
filters irrelevant elements, links related elements (e.g., captions to figures),
chunks the content logically based on titles, and formats the output into
structured Pydantic models with rich metadata.

The pipeline prioritizes high-resolution partitioning for detailed metadata
but includes fallback strategies for robustness. Optional integration points
for Large Language Models (LLMs) like OpenAI or TGI are included for
advanced linking and formatting tasks.
"""

import os
import sys
import uuid
import logging
from typing import List, Dict, Optional, TypedDict, Annotated, Tuple, Any, Literal
from pathlib import Path
import mimetypes

# Third-party Libraries
try:
    from PIL import Image
except ImportError:
    print("Pillow is not installed. Please install it: pip install Pillow")
    # Pillow is often needed for image processing within unstructured, especially hi_res
    pass

try:
    import pydantic
    from pydantic import BaseModel, Field, validator
except ImportError:
    print("Pydantic is not installed. Please install it: pip install pydantic")
    sys.exit(1)

try:
    from unstructured.partition.auto import partition
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.docx import partition_docx
    from unstructured.chunking.title import chunk_by_title
    from unstructured.elements.composite_element import CompositeElement
    from unstructured.elements.fundamental import Element, Text, Table
    from unstructured.elements.structured import FigureCaption # Example, adjust based on actual elements
    from unstructured.cleaners.core import clean_extra_whitespace
except ImportError:
    print("Unstructured library not fully installed. Please install core and potentially extras: "
          "pip install 'unstructured[local-inference]' or relevant components.")
    sys.exit(1)

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.sqlite import SqliteSaver # Example checkpointer
except ImportError:
    print("LangGraph library not installed. Please install it: pip install langgraph")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Configuration ---
# Consider moving these to environment variables or a config file
DEFAULT_MAX_CHUNK_CHARS = 1800  # Sensible default for chunk size
DEFAULT_HI_RES_MODEL_NAME = "yolox" # Or "detectron2_onnx"

# Set environment variables for potential LLM use
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
TGI_ENDPOINT = "http://localhost:8080"

# ==============================================================================
# I. Pydantic Models for Structured Output
# ==============================================================================

class CoordinatesMetadata(BaseModel):
    """Represents coordinate information for an element."""
    points: Optional[List[Tuple[float, float]]] = None
    system: Optional[str] = None # e.g., "PixelSpace", "PageSpace"
    layout_width: Optional[float] = None
    layout_height: Optional[float] = None

class LinkMetadata(BaseModel):
    """Represents linking information between elements."""
    linked_object_id: Optional[str] = Field(None, description="ID of the linked Image/Table element")
    is_ocr_for_image_id: Optional[str] = Field(None, description="ID of the Image this OCR text belongs to")

class ChunkMetadata(BaseModel):
    """Detailed metadata associated with a single chunk."""
    source_document_id: str = Field(description="Unique identifier for the source document")
    chunk_order: int = Field(description="Sequential order of the chunk within the document")
    page_number: Optional[int] = Field(None, description="Page number where the chunk primarily resides") # Can be None or first page
    bounding_box: Optional[CoordinatesMetadata] = Field(None, description="Coordinates bounding the chunk (best effort, e.g., title bbox)")
    chunk_type: str = Field(default="Text", description="Semantic type of the chunk (e.g., 'Text', 'Table', 'SectionTitle')")
    length_chars: int = Field(description="Number of characters in the chunk's Markdown content")
    size_bytes: int = Field(description="Size of the chunk's Markdown content in bytes")
    filename: str = Field(description="Original filename of the source document")
    filetype: str = Field(description="MIME type or file extension of the source document")

    # Linking metadata propagated from elements within the chunk
    linked_object_ids: List[str] = Field(default_factory=list, description="List of IDs for linked objects (Images/Tables) mentioned/contained in the chunk")
    ocr_image_ids: List[str] = Field(default_factory=list, description="List of Image IDs for which OCR text is included in this chunk")

    # Metadata inherited/derived from Unstructured elements
    element_ids: List[str] = Field(default_factory=list, description="List of Unstructured element IDs comprising this chunk")
    # Add other relevant unstructured metadata fields if needed, e.g.:
    # languages: Optional[List[str]] = None
    # detection_origin: Optional[str] = None # e.g., 'pdfminer', 'yolox'
    partitioning_strategy_used: Optional[str] = Field(None, description="The Unstructured partitioning strategy used ('hi_res', 'fast', 'ocr_only')")


class ChunkModel(BaseModel):
    """Represents the final structured chunk output."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this chunk")
    content_markdown: str = Field(description="Content of the chunk formatted as Markdown")
    metadata: ChunkMetadata = Field(description="Detailed metadata associated with the chunk")

    @validator('content_markdown')
    def clean_markdown(cls, v):
        # Basic cleaning, can be expanded
        return clean_extra_whitespace(v)


# ==============================================================================
# II. LangGraph Workflow Definition
# ==============================================================================

# --- A. Graph State ---

class GraphState(TypedDict):
    """Defines the state passed between nodes in the LangGraph workflow."""
    # Input state
    file_path: str
    source_document_id: Optional[str] # Can be provided or generated
    file_name: str
    file_type: str

    # Configuration (optional, can be added for more control)
    max_chunk_chars: int
    use_llm_linking: bool
    use_llm_formatting: bool

    # Processing state
    raw_elements: Optional[List[Element]]
    filtered_elements: Optional[List[Element]]
    linked_elements: Optional[List[Element]]
    chunks: Optional[List[CompositeElement]] # Output from chunk_by_title
    final_chunks: Optional[List[ChunkModel]] # Final structured output

    # Metadata tracking
    partitioning_strategy_used: Optional[Literal['hi_res', 'fast', 'ocr_only']]
    toc_detection_uncertain: bool
    needs_llm_linking_decision: bool # Flag to trigger conditional edge
    needs_llm_formatting_decision: bool # Flag to trigger conditional edge

    # Error handling
    error_message: Optional[str]

# --- B. Placeholder Nodes ---

def load_document(state: GraphState) -> Dict[str, Any]:
    """
    Placeholder node: Loads document information.

    In a real implementation, this would fetch the document content (if not
    already local), generate a unique source_document_id if needed,
    and potentially extract initial metadata.
    """
    logger.info(f"--- (0) Loading document info for: {state['file_path']} ---")
    file_path_obj = Path(state['file_path'])

    if not file_path_obj.is_file():
        logger.error(f"File not found: {state['file_path']}")
        return {"error_message": f"File not found: {state['file_path']}"}

    source_id = state.get('source_document_id') or f"doc_{uuid.uuid4()}"
    file_name = file_path_obj.name
    file_type, _ = mimetypes.guess_type(state['file_path'])

    if file_type is None:
        # Fallback to extension if MIME type guess fails
        file_type = file_path_obj.suffix.lower().strip('.')
        logger.warning(f"Could not guess MIME type for {file_name}, using extension: {file_type}")

    # TODO: Implement actual document loading if needed (e.g., from S3, DB)
    # TODO: Implement robust source_document_id generation/retrieval
    logger.info(f"Document ID: {source_id}, File: {file_name}, Type: {file_type}")

    # Initialize state fields
    return {
        "source_document_id": source_id,
        "file_name": file_name,
        "file_type": file_type,
        "raw_elements": None,
        "filtered_elements": None,
        "linked_elements": None,
        "chunks": None,
        "final_chunks": None,
        "partitioning_strategy_used": None,
        "toc_detection_uncertain": False,
        "needs_llm_linking_decision": False, # Default to false
        "needs_llm_formatting_decision": False, # Default to false
        "error_message": None,
        # Set defaults for config if not provided
        "max_chunk_chars": state.get('max_chunk_chars', DEFAULT_MAX_CHUNK_CHARS),
        "use_llm_linking": state.get('use_llm_linking', False),
        "use_llm_formatting": state.get('use_llm_formatting', False),
    }

def save_chunks(state: GraphState) -> Dict[str, Any]:
    """
    Placeholder node: Saves the final structured chunks.

    In a real implementation, this would persist the ChunkModel objects
    to a database, vector store, file system, etc.
    """
    final_chunks = state.get('final_chunks')
    source_doc_id = state.get('source_document_id')
    logger.info(f"--- (7) Saving Chunks for document: {source_doc_id} ---")

    if final_chunks:
        logger.info(f"Successfully generated {len(final_chunks)} chunks.")
        # TODO: Implement actual saving logic here. Examples:
        # - Save to JSON file:
        #   output_path = f"{source_doc_id}_chunks.json"
        #   with open(output_path, 'w') as f:
        #       import json # Make sure json is imported
        #       json.dump([chunk.dict() for chunk in final_chunks], f, indent=2)
        #   logger.info(f"Chunks saved to {output_path}")
        # - Index into a vector database (e.g., Qdrant, Weaviate)
        # - Store in a relational database
        pass # Replace with actual implementation
    else:
        logger.warning(f"No final chunks generated for document {source_doc_id}.")

    # This node typically marks the end of successful processing for a document.
    return {} # No state update needed, or could return status


# ==============================================================================
# III. LangGraph Node Implementations
# ==============================================================================

# --- A. Node 1: Document Partitioning ---

def partition_document(state: GraphState) -> Dict[str, Any]:
    """
    Partitions the document using Unstructured, selecting the best strategy.

    Prioritizes 'hi_res' for detailed metadata useful for linking, falling
    back to 'fast' or 'ocr_only' if 'hi_res' fails or is unsuitable.
    """
    file_path = state['file_path']
    file_type = state['file_type']
    source_doc_id = state['source_document_id']
    logger.info(f"--- (1) Partitioning Document: {file_path} (Type: {file_type}) ---")

    elements: Optional[List[Element]] = None
    strategy_used: Optional[Literal['hi_res', 'fast', 'ocr_only']] = None
    error_message: Optional[str] = None

    # --- Unstructured Partitioning Strategy Comparison ---
    # | Strategy   | Key Features                      | Metadata Richness    | Dependencies      | Speed      | Use Case                             | Fallback Rationale                          |
    # |------------|-----------------------------------|----------------------|-------------------|------------|--------------------------------------|---------------------------------------------|
    # | hi_res     | Layout detection (models)         | High (coords, types) | Heavy (inference) | Slow       | Complex layouts, PDFs, linking needed | Try first for best data; fallback if fails/slow |
    # | fast       | Rules/heuristics based extraction | Medium (basic types) | Lightweight       | Fast       | Simpler docs (DOCX), basic extraction | Faster, more robust than hi_res for many PDFs |
    # | ocr_only   | OCR text extraction only          | Low (text, maybe page)| Moderate (OCR)    | Variable   | Scanned images, text focus, layout ignored | Last resort when structure parsing fails entirely |
    # -----------------------------------------------------

    # Common parameters
    partition_params = {
        "include_page_breaks": True,
        "infer_table_structure": True,
        "skip_infer_table_types": [], # Infer all table types unstructured supports
        # "languages": ["eng"], # Consider making this configurable via state
        "strategy": "hi_res", # Start with hi_res attempt
        "hi_res_model_name": DEFAULT_HI_RES_MODEL_NAME,
    }

    # File Type Routing and Strategy Selection Logic
    partition_func = None
    if file_type in ["application/pdf", "pdf"]:
        partition_func = partition_pdf
    elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"]:
        partition_func = partition_docx
        # hi_res less critical/stable for DOCX usually, could default to fast
        # partition_params["strategy"] = "fast"
    # Add elif for other types like .html, .epub, .txt using partition()
    elif file_type in ["text/html", "html", "text/plain", "txt"]:
        partition_func = partition # Use generic partition for these
        partition_params.pop("hi_res_model_name", None) # May not apply
        partition_params["strategy"] = "fast" # Usually sufficient
    else:
        logger.warning(f"Unsupported file type '{file_type}' for specialized partitioning. Attempting generic partition.")
        # Fallback to generic partition, which might try 'fast' by default
        partition_func = partition
        partition_params.pop("hi_res_model_name", None)
        partition_params["strategy"] = "fast"


    if partition_func:
        # --- Attempt 1: 'hi_res' (if applicable) ---
        if partition_params.get("strategy") == "hi_res":
            logger.info(f"Attempting partitioning with strategy: 'hi_res'")
            try:
                elements = partition_func(filename=file_path, **partition_params)
                strategy_used = 'hi_res'
                logger.info(f"Successfully partitioned using 'hi_res'. Found {len(elements)} elements.")
            except Exception as e:
                logger.warning(f"'hi_res' partitioning failed: {e}. Falling back...")
                # Potentially check for specific errors here if needed
                partition_params["strategy"] = "fast" # Set for next attempt

        # --- Attempt 2: 'fast' ---
        if strategy_used is None and partition_params.get("strategy") == "fast":
            logger.info(f"Attempting partitioning with strategy: 'fast'")
            # Remove hi_res specific params if any remain
            fast_params = {k: v for k, v in partition_params.items() if k not in ["hi_res_model_name"]}
            fast_params["strategy"] = "fast"
            try:
                elements = partition_func(filename=file_path, **fast_params)
                strategy_used = 'fast'
                logger.info(f"Successfully partitioned using 'fast'. Found {len(elements)} elements.")
            except Exception as e:
                logger.warning(f"'fast' partitioning failed: {e}. Falling back to 'ocr_only' if applicable...")
                if file_type in ["application/pdf", "pdf"]: # ocr_only mainly for PDFs/images
                    partition_params["strategy"] = "ocr_only"
                else:
                    error_message = f"Partitioning failed with 'fast' strategy for non-PDF file: {e}"
                    logger.error(error_message)

        # --- Attempt 3: 'ocr_only' (primarily for PDFs) ---
        if strategy_used is None and partition_params.get("strategy") == "ocr_only":
             if file_type in ["application/pdf", "pdf"]:
                logger.info(f"Attempting partitioning with strategy: 'ocr_only'")
                ocr_params = {"strategy": "ocr_only", "include_page_breaks": True} # Simplified params
                try:
                    # Use generic partition for ocr_only as it handles dispatch
                    elements = partition(filename=file_path, **ocr_params)
                    strategy_used = 'ocr_only'
                    logger.info(f"Successfully partitioned using 'ocr_only'. Found {len(elements)} elements.")
                except Exception as e:
                    error_message = f"Partitioning failed with 'ocr_only' strategy: {e}"
                    logger.error(error_message)
             else:
                 # Should have been caught earlier, but safeguard
                 if not error_message:
                    error_message = f"'ocr_only' strategy not applicable for file type {file_type} after 'fast' failed."
                    logger.error(error_message)
    else:
        error_message = f"No suitable partition function found for file type: {file_type}"
        logger.error(error_message)

    if elements is None and not error_message:
        error_message = "Partitioning finished without error, but produced no elements."
        logger.error(error_message)


    # Update state
    update = {
        "raw_elements": elements if elements else [],
        "partitioning_strategy_used": strategy_used,
        "error_message": error_message
    }
    return update


# --- B. Node 2: Element Filtering ---

def filter_elements(state: GraphState) -> Dict[str, Any]:
    """Filters out unwanted elements like headers, footers, and potential TOCs."""
    raw_elements = state.get('raw_elements')
    if not raw_elements:
        logger.warning("No raw elements to filter.")
        return {"filtered_elements": []}

    logger.info(f"--- (2) Filtering Elements (Initial count: {len(raw_elements)}) ---")
    filtered = []
    toc_elements_found = []
    toc_detection_uncertain = False

    # Categories to filter out completely
    skip_categories = {'Header', 'Footer'}

    # Heuristic TOC detection (simple example)
    # More sophisticated logic could involve checking page numbers, indentation, specific keywords
    potential_toc_keywords = ["contents", "table of contents", "index"]
    max_toc_page = 3 # Assume TOCs usually occur within the first few pages

    for i, element in enumerate(raw_elements):
        # 1. Filter by category
        if element.category in skip_categories:
            # logger.debug(f"Filtering element {i} due to category: {element.category}")
            continue

        # 2. Heuristic TOC Filtering
        is_potential_toc = False
        element_text_lower = element.text.lower().strip()
        page_num = element.metadata.page_number

        # Check if element text matches TOC keywords and is early in the doc
        if any(keyword in element_text_lower for keyword in potential_toc_keywords) and \
           (page_num is None or page_num <= max_toc_page):
            # Check if it looks like a title element (common for TOC headers)
            if isinstance(element, Text) and element.category == "Title":
                is_potential_toc = True
                # logger.debug(f"Potential TOC Title found: {element.text}")

            # Check for list items with page numbers (common TOC structure)
            # This requires more complex pattern matching (e.g., regex)
            # Example: Looking for lines ending in numbers, possibly after dots/spaces
            # import re # Make sure re is imported
            # if isinstance(element, ListItem) and re.search(r'\b\d+$', element_text_lower):
            #     # Check proximity to other similar list items?
            #     is_potential_toc = True # This needs refinement

        # Simple heuristic: Filter elements identified as potential TOC markers
        # A more robust approach might collect potential TOC elements and then decide
        # based on patterns or density on early pages.
        if is_potential_toc:
            toc_elements_found.append(element)
            # For now, we filter aggressively if keywords match title
            # logger.debug(f"Filtering element {i} as potential TOC: {element.text}")
            continue # Skip adding this element

        # If element passed filters, add it
        filtered.append(element)

    # Post-filtering check for uncertainty
    if len(toc_elements_found) > 0:
        # If we found TOC elements but maybe didn't filter perfectly (e.g., missed list items)
        # we could set the uncertainty flag. For this simple version, we just log.
        logger.info(f"Found {len(toc_elements_found)} potential TOC elements (filtered based on title keywords).")
        # toc_detection_uncertain = True # Set flag if heuristics were ambiguous

    logger.info(f"Filtering complete. Retained {len(filtered)} elements.")
    return {
        "filtered_elements": filtered,
        "toc_detection_uncertain": toc_detection_uncertain,
        "linked_elements": filtered # Initialize linked_elements with filtered ones
    }


# --- C. Node 3: Element Linking ---

def link_elements(state: GraphState) -> Dict[str, Any]:
    """
    Links related elements using heuristics (coordinates, OCR metadata).

    Attempts to connect FigureCaptions to nearby Images/Tables and ensures
    OCR text elements are linked back to their source Images.
    Relies heavily on coordinate metadata from 'hi_res' partitioning.
    """
    filtered_elements = state.get('filtered_elements')
    strategy = state.get('partitioning_strategy_used')
    source_doc_id = state['source_document_id'] # Added missing definition

    if not filtered_elements:
        logger.warning("No filtered elements to link.")
        return {} # No changes needed

    logger.info(f"--- (3) Linking Elements (Strategy: {strategy}) ---")

    if strategy != 'hi_res':
        logger.warning(f"Skipping coordinate-based linking as partitioning strategy was '{strategy}', not 'hi_res'.")
        # Basic OCR linking might still be possible if metadata exists
        # For now, we just return the elements as is if not hi_res
        return {"linked_elements": filtered_elements}

    elements_by_page: Dict[int, List[Element]] = {}
    for el in filtered_elements:
        page = getattr(el.metadata, 'page_number', None)
        if page is not None:
            if page not in elements_by_page:
                elements_by_page[page] = []
            elements_by_page[page].append(el)

    linked_count = 0
    ocr_link_count = 0
    needs_llm_linking = False # Flag if heuristics seem insufficient

    # Create a mutable copy to update metadata
    linked_elements = list(filtered_elements) # Shallow copy, elements are mutable
    element_map = {el.id: el for el in linked_elements} # For easy lookup

    for i, element in enumerate(linked_elements):
        # 1. Coordinate-Based Linking (FigureCaption to Image/Table)
        if isinstance(element, FigureCaption):
            caption_coords = getattr(element.metadata, 'coordinates', None)
            page_num = getattr(element.metadata, 'page_number', None)

            if caption_coords and caption_coords.points and page_num is not None: # Fixed typo is not N
                # Define search area (e.g., elements below the caption on the same page)
                # Corrected calculation - needs pairs of points
                if len(caption_coords.points) >= 2: # Basic check
                   caption_y_center = (caption_coords.points[0][1] + caption_coords.points[1][1]) / 2 # Example: using first two points' y
                   caption_x_center = (caption_coords.points[0][0] + caption_coords.points[1][0]) / 2 # Example: using first two points' x
                else: continue # Skip if not enough points

                best_match_obj = None
                min_distance = float('inf')

                # Search elements on the same page
                for potential_obj in elements_by_page.get(page_num, []): # Added default empty list
                    # Check for Table or Image types from unstructured.elements.html
                    # Assuming 'Image' comes from 'unstructured.elements.html' or similar
                    # Check if Table type is correctly imported/used
                    if isinstance(potential_obj, (Table, Image)): # Make sure 'Image' is the correct class
                        obj_coords = getattr(potential_obj.metadata, 'coordinates', None)
                        if obj_coords and obj_coords.points and len(obj_coords.points) >= 2:
                            obj_y_center = (obj_coords.points[0][1] + obj_coords.points[1][1]) / 2
                            obj_x_center = (obj_coords.points[0][0] + obj_coords.points[1][0]) / 2

                            # Heuristic: Object should be spatially close, typically above the caption
                            # This logic needs tuning based on common layouts
                            vertical_distance = caption_y_center - obj_y_center
                            horizontal_distance = abs(caption_x_center - obj_x_center)
                            layout_width = caption_coords.layout_width or 600 # Estimate if not present

                            # Prioritize objects directly above and horizontally aligned
                            if 0 < vertical_distance < (caption_coords.layout_height or 800) * 0.5 and \
                               horizontal_distance < layout_width * 0.3:
                                distance = vertical_distance # Simple vertical distance for now
                                if distance < min_distance:
                                    min_distance = distance
                                    best_match_obj = potential_obj

                if best_match_obj:
                    # Generate/get ID for the linked object
                    # Using element.id directly is better practice if available and stable
                    linked_obj_id = getattr(best_match_obj, 'id', f"obj_{uuid.uuid4()}") # Use element ID if present
                    # Ensure linked_id exists in metadata if needed elsewhere, or just use element.id
                    # if not hasattr(best_match_obj.metadata, 'linked_id'):
                    #    best_match_obj.metadata.linked_id = linked_obj_id # Store it if necessary

                    # Update caption metadata
                    if not hasattr(element.metadata, 'links'):
                        element.metadata.links = [] # Ensure links attribute exists
                    link_info = {"linked_object_id": linked_obj_id} # Use the obj ID

                    # Avoid duplicate links
                    if link_info not in element.metadata.links:
                        element.metadata.links.append(link_info)
                        linked_count += 1
                        logger.debug(f"Linked Caption '{element.text[:30]}...' to {best_match_obj.category} ID {linked_obj_id}")


        # 2. OCR-Image Linking (Ensuring links exist)
        # This depends on how the partitioner handles OCR. Sometimes OCR text is
        # embedded in Image metadata, sometimes it's a separate Text element.
        # We assume separate Text elements might have 'image_source_id' or similar.
        image_source_id = getattr(element.metadata, 'image_source_id', None) # Example metadata field
        if image_source_id and isinstance(element, Text):
            # Check if the source image element exists and update link metadata
            if image_source_id in element_map:
                source_image_element = element_map[image_source_id]
                if not hasattr(element.metadata, 'links'):
                    element.metadata.links = []
                link_info = {"is_ocr_for_image_id": image_source_id}
                if link_info not in element.metadata.links:
                    element.metadata.links.append(link_info)
                    ocr_link_count += 1
                    logger.debug(f"Confirmed/added OCR link for text '{element.text[:30]}...' to Image ID {image_source_id}")
            else:
                logger.warning(f"OCR metadata references missing Image ID: {image_source_id}")

    logger.info(f"Linking complete. Added {linked_count} caption links and {ocr_link_count} OCR links.")

    # Decision point for optional LLM linking
    # Could be based on low link count, specific document types, or the uncertainty flag
    if state.get('use_llm_linking'):
        # Simple trigger: if few links were found heuristically for complex doc types
        file_type = state.get('file_type') # Make sure file_type is available in state
        if linked_count < 1 and file_type == 'application/pdf': # Example condition
            logger.info("Heuristic linking yielded few results; flagging for potential LLM linking.")
            needs_llm_linking = True

    return {
        "linked_elements": linked_elements,
        "needs_llm_linking_decision": needs_llm_linking
    }


# --- D. Node 4: Logical Chunking ---

def chunk_elements(state: GraphState) -> Dict[str, Any]:
    """Chunks the linked elements logically based on titles."""
    linked_elements = state.get('linked_elements')
    max_chars = state.get('max_chunk_chars', DEFAULT_MAX_CHUNK_CHARS)

    if not linked_elements:
        logger.warning("No linked elements to chunk.")
        return {"chunks": []}

    logger.info(f"--- (4) Chunking Elements (Max chars: {max_chars}) ---")
    try:
        # chunk_by_title groups elements under Title elements, creating CompositeElement chunks.
        # It leverages the semantic structure identified during partitioning.
        chunks = chunk_by_title(
            elements=linked_elements,
            max_characters=max_chars,
            multipage_sections=True, # Allow sections defined by titles to span pages
            combine_text_under_n_chars=100, # Combine small text elements under a title
            new_after_n_chars=max_chars, # Hard split if a section exceeds max_chars
            include_orig_elements=True # Keep original elements within the chunk object
        )
        logger.info(f"Chunking complete. Generated {len(chunks)} logical chunks.")
        return {"chunks": chunks}
    except Exception as e:
        logger.error(f"Error during chunk_by_title: {e}")
        return {"chunks": [], "error_message": f"Chunking failed: {e}"}


# --- E. Node 5: Final Formatting and Metadata Assembly ---

def format_chunks(state: GraphState) -> Dict[str, Any]:
    """Formats chunks into Markdown and assembles the final Pydantic models."""
    unstructured_chunks = state.get('chunks')
    source_doc_id = state['source_document_id']
    filename = state['file_name']
    filetype = state['file_type']
    strategy = state['partitioning_strategy_used']

    if not unstructured_chunks:
        logger.warning("No chunks to format.")
        return {"final_chunks": []}

    logger.info(f"--- (5) Formatting Chunks and Assembling Metadata ---")
    final_chunk_models: List[ChunkModel] = []
    needs_llm_formatting = False # Flag if any chunk needs LLM refinement
    needs_llm_formatting_for_this_chunk = False # Per-chunk flag

    for i, chunk in enumerate(unstructured_chunks):
        chunk_order = i + 1
        chunk_content_parts = []
        chunk_metadata = {} # Temp dict to build metadata

        # Aggregate metadata from elements within the chunk
        page_numbers = set()
        element_ids = []
        linked_object_ids = set()
        ocr_image_ids = set()
        chunk_bbox_points = None # Try to get bbox from title or first element
        primary_category = "Text" # Default chunk type

        # Handle non-composite chunks if any
        elements_in_chunk = chunk.metadata.orig_elements if hasattr(chunk.metadata, 'orig_elements') and chunk.metadata.orig_elements else [chunk]

        for element in elements_in_chunk:
            element_ids.append(element.id)
            page_num = getattr(element.metadata, 'page_number', None)
            if page_num is not None:
                page_numbers.add(page_num)

            # Capture BBox from the first element (often the title) as a proxy
            if chunk_bbox_points is None and hasattr(element.metadata, 'coordinates') and element.metadata.coordinates and element.metadata.coordinates.points:
                chunk_bbox_points = element.metadata.coordinates.points
                chunk_metadata['bounding_box_system'] = element.metadata.coordinates.system
                chunk_metadata['layout_width'] = element.metadata.coordinates.layout_width
                chunk_metadata['layout_height'] = element.metadata.coordinates.layout_height

            # Determine primary category (e.g., if it contains a table)
            if isinstance(element, Table) and primary_category == "Text":
                primary_category = "Table"
            elif element.category == "Title" and primary_category == "Text":
                primary_category = "SectionTitle" # Or just use Title

            # Aggregate linking metadata
            element_links = getattr(element.metadata, 'links', []) # Added default empty list
            for link in element_links:
                if 'linked_object_id' in link and link['linked_object_id']:
                    linked_object_ids.add(link['linked_object_id'])
                if 'is_ocr_for_image_id' in link and link['is_ocr_for_image_id']:
                    ocr_image_ids.add(link['is_ocr_for_image_id'])

            # --- Content Formatting ---
            # Attempt basic text or markdown conversion
            content_part = ""
            needs_llm_formatting_for_this_chunk = False # Reset for each element/chunk part

            if isinstance(element, Table):
                # Refinement Logic: Check for HTML table representation
                html_table = getattr(element.metadata, 'text_as_html', None)
                if html_table:
                    # Option 1: Convert HTML table to Markdown (requires library like 'markdownify')
                    try:
                        from markdownify import markdownify as md
                        content_part = md(html_table)
                        logger.debug("Converted HTML table to Markdown.")
                    except ImportError:
                        logger.warning("markdownify library not installed. Cannot convert HTML table to Markdown. Including raw HTML.")
                        content_part = f"\n```html\n{html_table}\n```\n" # Wrap in code block
                        # Flagging for LLM might be useful here if HTML is preferred over raw text
                        needs_llm_formatting_for_this_chunk = True
                    except Exception as md_err:
                        logger.warning(f"markdownify failed: {md_err}. Including raw HTML.")
                        content_part = f"\n```html\n{html_table}\n```\n"
                        needs_llm_formatting_for_this_chunk = True

                else:
                    # Fallback to plain text representation if no HTML
                    content_part = element.text
                    # Check if plain text looks unstructured (heuristic)
                    if '\n' not in content_part.strip() and len(content_part) > 50: # Very basic check
                        logger.warning(f"Table content seems unstructured: '{content_part[:50]}...'. Consider LLM formatting.")
                        needs_llm_formatting_for_this_chunk = True
            else:
                # For other elements, use the text attribute
                content_part = element.text

            chunk_content_parts.append(content_part)

            # Check if this specific chunk triggered need for LLM formatting
            if needs_llm_formatting_for_this_chunk and state.get('use_llm_formatting'):
               needs_llm_formatting = True # Set global flag if any chunk needs it


        # Combine content parts into final Markdown
        final_markdown = "\n\n".join(filter(None, chunk_content_parts))
        final_markdown = clean_extra_whitespace(final_markdown) # Clean whitespace

        # Calculate final metadata
        length_chars = len(final_markdown)
        size_bytes = len(final_markdown.encode('utf-8'))
        page_number = min(page_numbers) if page_numbers else None

        bbox_metadata = None
        if chunk_bbox_points:
            bbox_metadata = CoordinatesMetadata(
                points=chunk_bbox_points,
                system=chunk_metadata.get('bounding_box_system'),
                layout_width=chunk_metadata.get('layout_width'),
                layout_height=chunk_metadata.get('layout_height')
            )

        # Assemble Pydantic Metadata Model
        metadata_obj = ChunkMetadata(
            source_document_id=source_doc_id,
            chunk_order=chunk_order,
            page_number=page_number,
            bounding_box=bbox_metadata,
            chunk_type=primary_category,
            length_chars=length_chars,
            size_bytes=size_bytes,
            filename=filename,
            filetype=filetype,
            linked_object_ids=list(linked_object_ids),
            ocr_image_ids=list(ocr_image_ids),
            element_ids=element_ids,
            partitioning_strategy_used=strategy
        )

        # Assemble Final Pydantic Chunk Model
        chunk_model = ChunkModel(
            content_markdown=final_markdown,
            metadata=metadata_obj
        )
        final_chunk_models.append(chunk_model)


    logger.info(f"Formatting complete. Generated {len(final_chunk_models)} final chunk models.")

    # Decision point for optional LLM formatting (based on aggregated flag)
    if needs_llm_formatting and state.get('use_llm_formatting'):
        logger.info("Flagging for potential LLM formatting based on content checks (e.g., complex tables).")
        # The actual LLM call would happen in a subsequent conditional node
        # needs_llm_formatting flag is already set correctly based on per-chunk checks

    return {
        "final_chunks": final_chunk_models,
        "needs_llm_formatting_decision": needs_llm_formatting
    }

# ==============================================================================
# IV. LLM Integration Points (Placeholders)
# ==============================================================================

def llm_link_elements(state: GraphState) -> Dict[str, Any]:
    """
    (Placeholder) Uses an LLM to perform advanced element linking.

    This node would be conditionally invoked if heuristic linking is deemed
    insufficient (needs_llm_linking_decision is True).

    Requires configuration of LLM client (OpenAI, TGI endpoint).
    """
    linked_elements = state.get('linked_elements', []) # Start with heuristically linked elements
    logger.info(f"--- (LLM) Attempting Advanced Element Linking ---")

    if not state.get('use_llm_linking'):
        logger.warning("LLM Linking requested by graph logic, but 'use_llm_linking' is False in state. Skipping.")
        return {"linked_elements": linked_elements} # Return original elements

    if not linked_elements:
        logger.warning("No elements provided to LLM for linking.")
        return {}

    # TODO: Implement LLM Linking Logic
    # 1. Prepare Context/Prompts:
    #    - Select relevant elements (e.g., captions, tables, images on the same page).
    #    - Format element data (text, type, coordinates) into a prompt.
    #    - Example Prompt: "Given the following document elements from page X with their types and coordinates (pixels, origin top-left, width W, height H), identify which FigureCaption elements belong to which Image or Table elements. Provide the links as pairs of element IDs: {'caption_id': 'linked_object_id'}. Elements: [{'id': 'id1', 'type': 'FigureCaption', 'text': '...', 'coords': [[x1,y1],[x2,y2],...]}, {'id': 'id2', 'type': 'Image', 'coords':...},...]"
    # 2. Configure LLM Client:
    #    - Use os.environ.get("OPENAI_API_KEY") or TGI_ENDPOINT.
    #    - Initialize the client (e.g., OpenAI(), or a custom client for TGI).
    # 3. Invoke LLM:
    #    - Send the prompt to the LLM API.
    #    - Handle potential API errors, timeouts, retries.
    # 4. Parse Response:
    #    - Extract the linked pairs from the LLM's response (e.g., JSON parsing).
    #    - Validate the response format and element IDs.
    # 5. Update Element Metadata:
    #    - Iterate through the LLM-identified links.
    #    - Find the corresponding elements in linked_elements.
    #    - Update the metadata.links attribute similar to the heuristic linking node.
    #    - Be careful about overwriting or merging with existing heuristic links.

    logger.warning("LLM Linking node is a placeholder. No actual LLM call performed.")
    # Simulate finding some links for demonstration
    # updated_elements = linked_elements # Modify this list based on LLM response
    # logger.info("LLM Linking simulation complete.")

    # Return the potentially updated list of elements
    return {"linked_elements": linked_elements} # Return original list for now

def llm_format_markdown(state: GraphState) -> Dict[str, Any]:
    """
    (Placeholder) Uses an LLM to refine the Markdown formatting of chunks.

    This node would be conditionally invoked if heuristic formatting is deemed
    insufficient (needs_llm_formatting_decision is True), especially for
    complex tables or lists.

    Requires configuration of LLM client (OpenAI, TGI endpoint).
    """
    final_chunks = state.get('final_chunks', [])
    logger.info(f"--- (LLM) Attempting Advanced Markdown Formatting ---")

    if not state.get('use_llm_formatting'):
        logger.warning("LLM Formatting requested by graph logic, but 'use_llm_formatting' is False in state. Skipping.")
        return {"final_chunks": final_chunks} # Return original chunks

    if not final_chunks:
        logger.warning("No final chunks provided to LLM for formatting.")
        return {}

    # TODO: Implement LLM Formatting Logic
    # 1. Identify Chunks Needing Refinement:
    #    - This might be pre-flagged (based on needs_llm_formatting_decision),
    #      or identified here based on content analysis (e.g., presence of '```html', complex lists).
    # 2. Prepare Context/Prompts for each chunk:
    #    - Get the raw content (or potentially the initial Markdown attempt).
    #    - Example Prompt: "Convert the following text content (which might contain raw text or HTML fragments, especially for tables) into clean, well-structured GitHub Flavored Markdown. Pay close attention to preserving table structure accurately. Content:\n{chunk_content}\n"
    # 3. Configure LLM Client (as in llm_link_elements).
    # 4. Invoke LLM for each identified chunk:
    #    - Send prompt, handle errors/retries.
    # 5. Parse Response:
    #    - Extract the cleaned Markdown content.
    # 6. Update ChunkModel:
    #    - Find the corresponding ChunkModel in final_chunks.
    #    - Update the content_markdown field.
    #    - Recalculate metadata.length_chars and metadata.size_bytes.

    logger.warning("LLM Formatting node is a placeholder. No actual LLM call performed.")
    # Simulate formatting for demonstration
    # updated_chunks = final_chunks # Modify this list based on LLM responses
    # logger.info("LLM Formatting simulation complete.")

    # Return the potentially updated list of chunks
    return {"final_chunks": final_chunks} # Return original list for now

# ==============================================================================
# V. Graph Assembly and Execution
# ==============================================================================

def should_continue(state: GraphState) -> Literal["continue", "__end__"]: # Use __end__ for LangGraph v0.1+
    """Determines if the graph should continue or end based on errors."""
    if state.get("error_message"):
        logger.error(f"Pipeline ending due to error: {state['error_message']}")
        return END # Use END constant from langgraph.graph
    return "continue"

def decide_llm_linking(state: GraphState) -> Literal["llm_link_elements", "chunk_elements"]:
    """Conditional edge: Route to LLM linking or directly to chunking."""
    if state.get("needs_llm_linking_decision", False): # Added default False
        logger.info("Conditional Edge: Routing to LLM Linking.")
        return "llm_link_elements"
    else:
        logger.info("Conditional Edge: Skipping LLM Linking, proceeding to Chunking.")
        return "chunk_elements"

def decide_llm_formatting(state: GraphState) -> Literal["llm_format_markdown", "save_chunks"]:
    """Conditional edge: Route to LLM formatting or directly to saving."""
    if state.get("needs_llm_formatting_decision", False): # Added default False
        logger.info("Conditional Edge: Routing to LLM Formatting.")
        return "llm_format_markdown"
    else:
        logger.info("Conditional Edge: Skipping LLM Formatting, proceeding to Save.")
        return "save_chunks"

# --- Define the Graph ---
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("load_document", load_document)
workflow.add_node("partition_document", partition_document)
workflow.add_node("filter_elements", filter_elements)
workflow.add_node("link_elements", link_elements)
workflow.add_node("llm_link_elements", llm_link_elements) # Placeholder LLM node
workflow.add_node("chunk_elements", chunk_elements)
workflow.add_node("format_chunks", format_chunks)
workflow.add_node("llm_format_markdown", llm_format_markdown) # Placeholder LLM node
workflow.add_node("save_chunks", save_chunks) # Placeholder save node

# Define Edges
workflow.set_entry_point("load_document")

# Check for errors after loading
workflow.add_conditional_edges(
    "load_document",
    should_continue,
    {
        "continue": "partition_document",
        END: END # Use END constant
    }
)

# Check for errors after partitioning
workflow.add_conditional_edges(
    "partition_document",
    should_continue,
    {
        "continue": "filter_elements",
        END: END # Use END constant
    }
)

workflow.add_edge("filter_elements", "link_elements")

# Conditional edge for LLM Linking
workflow.add_conditional_edges(
    "link_elements",
    decide_llm_linking,
    {
        "llm_link_elements": "llm_link_elements",
        "chunk_elements": "chunk_elements",
    }
)

# Edge from LLM linking (if used) to chunking
workflow.add_edge("llm_link_elements", "chunk_elements")

# Check for errors after chunking
workflow.add_conditional_edges(
    "chunk_elements",
    should_continue,
    {
        "continue": "format_chunks",
        END: END # Use END constant
    }
)

# Conditional edge for LLM Formatting
workflow.add_conditional_edges(
    "format_chunks",
    decide_llm_formatting,
    {
        "llm_format_markdown": "llm_format_markdown",
        "save_chunks": "save_chunks",
    }
)

# Edge from LLM formatting (if used) to saving
workflow.add_edge("llm_format_markdown", "save_chunks")

# Final edge to END after saving
workflow.add_edge("save_chunks", END) # Use END constant

# --- Compile the Graph ---
# Optional: Add memory/checkpointing
# memory = SqliteSaver.from_conn_string(":memory:")
# app = workflow.compile(checkpointer=memory)

app = workflow.compile()


# ==============================================================================
# VI. Usage and Customization
# ==============================================================================

# --- A. Running the Pipeline ---

if __name__ == "__main__":
    # Example Usage:
    # Provide the path to a document file as a command-line argument
    # python unstructured_langgraph_pipeline.py /path/to/your/document.pdf
    # Or:
    # python unstructured_langgraph_pipeline.py /path/to/your/document.docx

    if len(sys.argv) < 2:
        print("Usage: python unstructured_langgraph_pipeline.py <file_path> [--use-llm-linking] [--use-llm-formatting]")
        # Example with a dummy file for demonstration if no arg provided
        # Create a dummy file if it doesn't exist
        dummy_file = "dummy_document.txt"
        if not os.path.exists(dummy_file):
            with open(dummy_file, "w") as f:
                f.write("This is the first section title.\n\nThis is paragraph one.\n\nThis is paragraph two.\n\n")
                f.write("This is the second section title.\n\nThis contains a simple list:\n- Item 1\n- Item 2")
            print(f"Created dummy file: {dummy_file}")
        file_to_process = dummy_file
        print(f"No file path provided. Using dummy file: {file_to_process}")
        # sys.exit(1) # Commented out to allow running with dummy file
    else:
        file_to_process = sys.argv[1] # Get the first argument as file path

    # Check for optional LLM flags
    use_llm_linking_flag = "--use-llm-linking" in sys.argv
    use_llm_formatting_flag = "--use-llm-formatting" in sys.argv

    # --- B. Configuration and Tuning ---
    # Key parameters can be adjusted here or passed via config:
    initial_state = {
        "file_path": file_to_process,
        # "source_document_id": "my_custom_id_123", # Optional: Provide an ID
        "max_chunk_chars": 1500, # Adjust chunk size target
        "use_llm_linking": use_llm_linking_flag, # Control optional LLM step
        "use_llm_formatting": use_llm_formatting_flag, # Control optional LLM step
    }

    # --- Execute the Graph ---
    # For streaming results and seeing state changes:
    # config = {"configurable": {"thread_id": "doc-process-thread-1"}} # Example thread ID for checkpointing
    # for event in app.stream(initial_state, config=config):
    #     for key, value in event.items():
    #         print(f"--- Event: {key} ---")
    #         # print(value) # Print the full state update (can be verbose)
    #         # Check if final_chunks key exists in the output message's content
    #         # This check might need adjustment based on LangGraph version and stream output format
    #         if isinstance(value, dict) and 'messages' in value:
    #              # Example check - adjust based on actual structure
    #              if any('final_chunks' in msg.content for msg in value.get('messages', []) if hasattr(msg, 'content')):
    #                  print("Final chunks generated (content omitted for brevity).")

    # For final result only:
    try:
        final_state = app.invoke(initial_state)

        # Print final status and output summary
        if final_state.get("error_message"):
            print(f"\n--- Pipeline Failed ---")
            print(f"Error: {final_state['error_message']}")
        elif final_state.get("final_chunks"):
            print(f"\n--- Pipeline Completed Successfully ---")
            print(f"Document ID: {final_state.get('source_document_id')}") # Use .get for safety
            print(f"Partitioning Strategy Used: {final_state.get('partitioning_strategy_used')}")
            print(f"Number of Chunks Generated: {len(final_state['final_chunks'])}")
            # print("\n--- First Chunk Example ---")
            # print(final_state['final_chunks'][0].json(indent=2)) # Print first chunk as JSON
        else:
            print("\n--- Pipeline Completed ---")
            print("No final chunks were generated (this might be expected for empty docs or filtering).")

    except Exception as e:
        print(f"\n--- An unexpected error occurred during graph execution ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # --- C. Potential Extensions ---
    # - Implement actual logic in load_document and save_chunks.
    # - Add support for more file types (HTML, EPUB, etc.) via unstructured.partition.
    # - Enhance heuristic TOC filtering (e.g., regex, positional analysis).
    # - Implement more sophisticated coordinate-based linking logic (proximity thresholds, geometric checks).
    # - Integrate actual LLM clients (OpenAI, HuggingFace TGI) in the LLM nodes.
    # - Add more robust error handling and retry mechanisms within nodes or the graph.
    # - Extract additional metadata (e.g., named entities) within the format_chunks node.
    # - Make configurations (chunk size, strategies, LLM endpoints) more dynamic (e.g., via config files or env vars).
    # - Add specific handling for multi-column documents if hi_res struggles.


# ==============================================================================
# VII. Conclusion (Summary)
# ==============================================================================
"""
This module provides a modular and robust document chunking pipeline leveraging
Unstructured for parsing and LangGraph for workflow orchestration. Key features include:

 * Adaptive Parsing: Prioritizes high-quality 'hi_res' partitioning with automatic
   fallbacks to 'fast' or 'ocr_only' for improved robustness across diverse documents.
 * Context-Aware Chunking: Uses chunk_by_title for semantically coherent chunks
   respecting document structure.
 * Rich Metadata Output: Generates structured Pydantic models containing detailed
   metadata (source info, coordinates, links, types, etc.) alongside Markdown content.
 * Element Linking: Includes heuristic logic to link related elements like captions
   and figures, preserving contextual relationships.
 * Extensibility: Offers optional, conditional integration points for LLMs to handle
   complex linking or formatting tasks where heuristics may fall short.
 * Modularity: Built with LangGraph, allowing easy modification, extension, or
   integration into larger AI systems.

This pipeline is suitable for building advanced RAG systems, document analysis tools,
or any application requiring high-quality, structured data extraction from documents.
"""