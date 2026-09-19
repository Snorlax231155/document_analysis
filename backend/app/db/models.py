from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_hash = Column(String(64), unique=True, index=True, nullable=False)
    file_size = Column(Integer, nullable=False, default=0)
    page_count = Column(Integer, nullable=False, default=0)
    chunk_count = Column(Integer, nullable=False, default=0)
    status = Column(String(50), nullable=False, default="processed")
    upload_time = Column(DateTime, default=datetime.utcnow, nullable=False)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    page_number = Column(Integer, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False, default=0)

    document = relationship("Document", back_populates="chunks")


class Query(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    question = Column(Text, nullable=False)
    top_k = Column(Integer, nullable=False, default=5)
    enable_reranking = Column(Boolean, nullable=False, default=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    responses = relationship("Response", back_populates="query", cascade="all, delete-orphan")


class Response(Base):
    __tablename__ = "responses"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey("queries.id", ondelete="CASCADE"), nullable=False, index=True)
    answer = Column(Text, nullable=False)
    answer_found = Column(Boolean, nullable=False, default=True)
    retrieval_ms = Column(Float, nullable=False, default=0.0)
    generation_ms = Column(Float, nullable=False, default=0.0)
    total_ms = Column(Float, nullable=False, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    query = relationship("Query", back_populates="responses")
    citations = relationship("Citation", back_populates="response", cascade="all, delete-orphan")


class Citation(Base):
    __tablename__ = "citations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    response_id = Column(Integer, ForeignKey("responses.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    chunk_id = Column(Integer, nullable=False)
    document_name = Column(String(255), nullable=False)
    relevance_score = Column(Float, nullable=False, default=0.0)

    response = relationship("Response", back_populates="citations")


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    recall_at_1 = Column(Float, nullable=False)
    recall_at_3 = Column(Float, nullable=False)
    recall_at_5 = Column(Float, nullable=False)
    recall_at_10 = Column(Float, nullable=False)
    mrr = Column(Float, nullable=False)
    faithfulness = Column(Float, nullable=False)
    answer_relevance = Column(Float, nullable=False)
    total_samples = Column(Integer, nullable=False)
