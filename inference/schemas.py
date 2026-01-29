from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    source_ip: str
    protocol: str
    log_type: str
    bytes_transferred: int
    user_agent: str
    request_path: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
