"""Pydantic models and prompt-chaining runner for a document-generation demo.

Workflow mapping (example):
 - DocumentOutline: produced by the "strategist" step (LLM generates an outline)
 - OutlineValidation: produced by the "editor/reviewer" step (LLM checks quality)
 - FinalDocument: produced by the "writer" step (LLM writes full content)

Usage notes:
 - Keep these models stable (backwards compatible) because they may be stored
   or passed between services. Add new optional fields instead of changing
   existing field types.
 - Field descriptions are included to make generated OpenAPI docs or CLI help
   more useful.
"""

from typing import List, Optional, cast
import logging
import os

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# Load environment variables from .env file (e.g., OPENAI_API_KEY)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

model = "gpt-4.1"

# Initialize the OpenAI client using API key from .env (via load_dotenv)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Document outline produced by the first LLM step (strategist)
# ---------------------------------------------------------------------------
class DocumentOutline(BaseModel):
    """Structured outline produced by an LLM given a document topic.

    This model captures the minimal information required to progress to the
    next step in a prompt chain: a topic and a list of ordered section titles.

    Example:
        DocumentOutline(
            topic="Design patterns for agentic systems",
            sections=["Introduction", "Prompt Chaining", "Routing", "Conclusion"]
        )

    Validation/semantics:
    - `sections` is an ordered list; order matters because the writer will
      expand them in sequence.
    - Keep this model small to allow multiple downstream consumers to use it.
    """
    """First LLM call: Generate a structured outline for a document."""
    topic: str = Field(description="The main topic of the document.")
    sections: List[str] = Field(
        description="A list of section titles for the document outline."
    )


# ---------------------------------------------------------------------------
# Outline validation result produced by a reviewer/editor LLM step
# ---------------------------------------------------------------------------
class OutlineValidation(BaseModel):
    """Result of an LLM-based quality check against the generated outline.

    This model conveys whether the outline meets quality criteria and the
    rationale. It is useful for human-in-the-loop flows and for automated
    branching (e.g., re-run strategist when `is_valid` is False).

    Fields:
    - is_valid: boolean decision that can be used to short-circuit the chain.
    - reasoning: a short human-readable explanation of the decision.
    - confidence_score: numeric confidence (0.0-1.0) about the decision.
    """
    """Second LLM call: Validate the generated outline against quality criteria."""
    is_valid: bool = Field(
        description="Whether the outline is logical, comprehensive, and well-structured."
    )
    reasoning: str = Field(
        description="A brief explanation for why the outline is or is not valid."
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1 on the validity of the outline.",
    )


# ---------------------------------------------------------------------------
# Final document produced by the writer LLM step
# ---------------------------------------------------------------------------
class FinalDocument(BaseModel):
    """Complete document content generated from a validated outline.

    The writer step consumes a validated `DocumentOutline` (or the outline
    plus review feedback) and returns fully expanded content. Keep the
    `full_content` field as plain string so downstream consumers can store or
    further process it (e.g., convert to markdown, publish to CMS).
    """
    """Third LLM call: Generate the full document content from the outline."""
    title: str = Field(description="A suitable title for the final document.")
    full_content: str = Field(
        description="The complete, well-written content of the document, based on the provided outline."
    )

# generate_document_outline which takes a raw topic and turns it into a structured plan
# This function represents the "strategist" step in the prompt chain.
# It takes a raw topic and produces a structured outline that can be validated and then expanded into a full document.
def generate_document_outline(topic: str) -> DocumentOutline:
    """First LLM call: generate a structured outline from a topic."""
    logger.info(f"Starting outline generation for topic: '{topic}'")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    """You are an expert content strategist.
                       Create a logical and comprehensive outline for a document on the given topic.
                       The outline should include an introduction, several body sections, and a conclusion."""
                ),
            },
            {"role": "user", "content": topic},
        ],
        response_format=DocumentOutline,
    )

    result = completion.choices[0].message.parsed
    logger.info("Outline generated successfully.")
    return result

# validate_document_outline function doesn’t create new content;
# its only job is to judge the output of the previous LLM call.
# This function represents the "editor/reviewer" step in the prompt chain.
# It takes the generated outline and evaluates its quality based on predefined criteria.
# The output is a structured validation result that indicates whether the outline is acceptable and provides reasoning for the decision.
# Gatekeeper. This step perfectly implements another core agentic design pattern: the reflection.
# The LLM is reflecting on the output of the previous step and making a judgment about its quality.
# This is a critical part of many agentic systems, as it allows for self-correction and improvement over time.
# By having a separate validation step, we can ensure that only high-quality outlines are passed on to the writing stage, which can save time and resources in the long run.
def validate_document_outline(outline: DocumentOutline) -> OutlineValidation:
    """Second LLM call to validate the quality of the generated outline."""
    logger.info("Starting outline validation.")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a critical quality assurance editor. Your primary goal is to REJECT "
                    "low-quality or vague outlines. An outline is considered invalid if the "
                    "original topic is too vague, ambiguous, or lacks a clear focus (e.g., 'stuff', "
                    "'things', 'an article'). Be strict. If the topic is bad, the outline is bad. "
                    "Provide a brief reason for your decision."
                )
            },
            {"role": "user", "content": str(outline.model_dump())},
        ],
        response_format=OutlineValidation,
    )
    result = completion.choices[0].message.parsed
    logger.info(
        f"Validation complete - Is valid: {result.is_valid}, Confidence: {result.confidence_score:.2f}"
    )
    # Log the reasoning, especially for failures
    if not result.is_valid:
        logger.warning(f"Validation failed. Reasoning: {result.reasoning}")
    return result

# generate_final_document, takes the approved plan and does the heavy lifting: writing the full article.
def generate_final_document(outline: DocumentOutline) -> FinalDocument:
    """Third LLM call: expand the validated outline into a full document."""
    logger.info("Generating final document from outline.")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    """You are a skilled author.
                    Write a comprehensive, well-structured document based on the provided outline.
                    Include an engaging title, clear section headings, and a concise conclusion."""
                ),
            },
            {"role": "user", "content": str(outline.model_dump())},
        ],
        response_format=FinalDocument,
    )

    result = completion.choices[0].message.parsed
    logger.info(f"Final document generated with title: '{result.title}'")
    return result

def create_document_from_topic(topic: str) -> Optional[FinalDocument]:
    """Main function implementing the prompt chain with a validation gate."""
    logger.info(f"Starting document creation process for topic: '{topic}'")

    # First LLM call: Generate the outline
    document_outline = generate_document_outline(topic)

    # Second LLM call: Validate the outline
    validation_result = validate_document_outline(document_outline)

    # Gate check: Verify if the outline is valid with sufficient confidence
    if not validation_result.is_valid or validation_result.confidence_score < 0.8:
        logger.warning(
            f"Gate check failed - Outline not valid or confidence too low ({validation_result.confidence_score:.2f})."
        )
        logger.warning(f"Reasoning: {validation_result.reasoning}")
        return None

    logger.info("Gate check passed, proceeding with final document generation.")

    # Third LLM call: Generate the full document
    final_document = generate_final_document(document_outline)

    logger.info("Document creation process completed successfully.")
    return final_document


# topic_input = "The benefits of remote work for small businesses"
topic_input = "Umbrella"

final_document_result = create_document_from_topic(topic_input)
if final_document_result:
    print(f"\nTitle: {final_document_result.title}")
    print("\n--- Document Content ---")
    # Printing only the first 500 characters for brevity
    print(final_document_result.full_content[:500] + "...")
else:
    print("Failed to generate a valid document for the topic.")