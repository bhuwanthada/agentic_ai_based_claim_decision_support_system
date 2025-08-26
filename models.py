from pydantic import BaseModel, Field
from typing import Literal

from dotenv import load_dotenv
import logging
from logging_config import setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger("utils")


class LLMResponseSchema(BaseModel):
    summary: str = Field(description="A concise summary behind claim status")
    claim_status: Literal["approve", "reject", "human review"] = Field(
        description="To provide claim status based upon provided literal value"
    )


class LLMResponseSchemaForDetailedSummery(BaseModel):
    detailed_summary: str = Field(
        description="A brief summary with recent "
        "advancements in oncology guided by PubMed and Guideline api"
    )
