from typing import List, Dict, Any
from pydantic import BaseModel, Field


class BatchPredictionRequest(BaseModel):
    """
    JSON batch prediction request.
    """
    records: List[Dict[str, Any]] = Field(
        ..., description="List of customer feature dictionaries"
    )

class PredictionSummary(BaseModel):
    model_config = {
        "protected_namespaces": ()
    }

    total_records: int
    churn_rate: float
    model_version: str
    latency_ms: float
    request_id: str

class BatchPredictionResponse(BaseModel):
    summary: PredictionSummary
    preview: List[Dict[str, Any]]