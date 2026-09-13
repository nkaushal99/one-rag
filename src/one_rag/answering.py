from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from one_rag.context import build_context
from one_rag.retrieval import RetrievalService
from one_rag.settings import Settings, get_settings


class AnswerService:
    """Build inspectable retrieved context and answer from that context only."""

    def __init__(self, settings: Settings | None = None, retrieval: RetrievalService | None = None) -> None:
        self.settings = settings or get_settings()
        provider = self.settings.llm_provider or "gemini"
        if provider != "gemini":
            raise ValueError("LLM_PROVIDER must be 'gemini'.")
        self.retrieval = retrieval or RetrievalService(self.settings)
        api_key = self.settings.google_api_key.get_secret_value() if self.settings.google_api_key else None
        self.model = ChatGoogleGenerativeAI(model=self.settings.llm_model or "gemini-3.5-flash-lite", temperature=self.settings.llm_temperature, api_key=api_key)

    def answer(self, question: str, limit: int, neighbor_count: int, include_parent_context: bool, tenant_id: str = "local", rerank_candidate_limit: int | None = None) -> dict[str, object]:
        evidence = self.retrieval.search(question, limit, neighbor_count, include_parent_context, tenant_id, rerank_candidate_limit)
        if not evidence:
            return {"answer": "I do not have retrieved evidence to answer that question.", "context": "", "evidence": [], "trace": self.retrieval.last_trace}
        context = build_context(evidence)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer only from the supplied context. If the context does not answer the question, respond with exactly: I do not know based on the provided context. Cite each factual claim with its bracketed source and chunk label. Do not invent details."),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ])
        try:
            answer = (prompt | self.model | StrOutputParser()).invoke({"question": question, "context": context})
        except Exception as error:
            raise RuntimeError("Gemini could not generate an answer with the configured model.") from error
        return {"answer": answer, "context": context, "evidence": evidence, "trace": self.retrieval.last_trace}
