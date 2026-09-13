from functools import lru_cache


class RerankerUnavailable(RuntimeError):
    """The optional cross-encoder could not be loaded or score candidates."""


@lru_cache
def get_ranker(model_name: str):
    try:
        from flashrank import Ranker

        return Ranker(model_name=model_name)
    except Exception as error:
        raise RerankerUnavailable("FlashRank could not load the configured reranker model.") from error


class FlashRankReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def score(self, question: str, candidates: list[dict[str, object]]) -> dict[str, float]:
        try:
            from flashrank import RerankRequest

            passages = [{"id": str(candidate["_point_id"]), "text": str(candidate["text"])} for candidate in candidates]
            results = get_ranker(self.model_name).rerank(RerankRequest(query=question, passages=passages))
            return {str(result["id"]): float(result["score"]) for result in results}
        except RerankerUnavailable:
            raise
        except Exception as error:
            raise RerankerUnavailable("FlashRank could not score the retrieved candidates.") from error
