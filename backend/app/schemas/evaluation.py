from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict


class EvaluationResultSchema(BaseModel):
    id: Optional[int] = None
    timestamp: datetime
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    faithfulness: float
    answer_relevance: float
    total_samples: int

    model_config = ConfigDict(from_attributes=True)



class EvaluationRunRequestSchema(BaseModel):
    sample_dataset_path: Optional[str] = None
