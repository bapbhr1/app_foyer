"""Modèles typés du corpus et de la réponse métier."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ContractArticle(BaseModel):
    """Un article (clause) d'un contrat."""

    article_id: str
    title: str
    text: str


class Contract(BaseModel):
    """Un contrat d'assurance synthétique et ses articles."""

    contract_id: str
    product: str
    product_line: str
    title: str
    policyholder: str
    articles: list[ContractArticle]


class Chunk(BaseModel):
    """Unité de recherche : une clause traçable jusqu'à son article."""

    chunk_id: str
    contract_id: str
    product: str
    product_line: str
    article_id: str
    article_title: str
    text: str

    def to_context_string(self) -> str:
        return (
            f"[SOURCE {self.chunk_id} | contrat={self.contract_id} | "
            f"produit={self.product} | article={self.article_id} - {self.article_title}]\n"
            f"{self.text}"
        )


class RetrievedChunk(BaseModel):
    """Clause retrouvée avec ses scores de pertinence."""

    chunk: Chunk
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0


class CoverageStatus(str, Enum):
    """Statut de prise en charge recommandé par l'assistant."""

    COVERED = "PRIS_EN_CHARGE"
    NOT_COVERED = "NON_PRIS_EN_CHARGE"
    PARTIAL = "PARTIELLEMENT_PRIS_EN_CHARGE"
    NEEDS_REVIEW = "A_VERIFIER_PAR_GESTIONNAIRE"


class Citation(BaseModel):
    """Extrait contractuel validé, rattaché à sa clause d'origine."""

    chunk_id: str
    contract_id: str
    article_id: str
    article_title: str
    quote: str


class ClaimEvidence(BaseModel):
    """Affirmation de la réponse et extrait proposé pour la justifier."""

    claim: str
    chunk_id: str
    quote: str


class EvidenceCheck(BaseModel):
    """Résultat du contrôle d'une affirmation dans les contrats."""

    claim: str
    supported: bool
    citation: Citation | None = None


class ConversationTurn(BaseModel):
    """Échange précédent utile pour comprendre une question de suivi."""

    question: str
    answer: str


class RAGQuery(BaseModel):
    """Requête adressée au moteur RAG."""

    question: str = Field(min_length=3)
    product_line: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)


class LLMAnswer(BaseModel):
    """Sortie demandée au LLM avant contrôles déterministes."""

    status: CoverageStatus
    answer: str
    decision: str
    conditions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    claims: list[ClaimEvidence] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    reasoning_summary: str


class ConfidenceBreakdown(BaseModel):
    """Détail des trois composantes de la fiabilité."""

    factual_support: float = Field(ge=0.0, le=1.0)
    uncertainty_handling: float = Field(ge=0.0, le=1.0)
    source_handling: float = Field(ge=0.0, le=1.0)


class RAGResponse(LLMAnswer):
    """Réponse complète renvoyée à l'interface, contrôles inclus."""

    citations: list[Citation] = Field(default_factory=list)
    evidence_checks: list[EvidenceCheck] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_breakdown: ConfidenceBreakdown
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
