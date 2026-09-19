from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ChunkResponseSchema(BaseModel):
    id: int
    page_number: int
    chunk_index: int
    text: str
    token_count: int

    model_config = ConfigDict(from_attributes=True)


class DocumentResponseSchema(BaseModel):
    id: int
    filename: str
    file_path: str
    file_hash: str
    file_size: int
    page_count: int
    chunk_count: int
    status: str
    upload_time: datetime

    model_config = ConfigDict(from_attributes=True)



class DocumentDetailSchema(DocumentResponseSchema):
    chunks: List[ChunkResponseSchema] = []
