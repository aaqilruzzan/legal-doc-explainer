"""Pydantic models for data validation and structure."""

from pydantic import BaseModel, Field
from typing import Literal

class ClauseDetail(BaseModel):
    """Details of a specific contract clause."""
    heading: str = Field(description="A concise, descriptive title for the specific clause found in the context.")
    description: str = Field(description="A summary of the key terms found (max 30 words).")

class ClauseAnalysis(BaseModel):
    """Analysis result for a contract clause."""
    clause: ClauseDetail
    recommendation: str = Field(description="A brief, actionable recommendation (max 20 words).")
    risk: Literal["critical", "high", "medium", "low", ""] = Field(description="The assessed risk level.")
    confidence: Literal["high", "medium", "low", ""] = Field(description="The confidence level of the analysis.")