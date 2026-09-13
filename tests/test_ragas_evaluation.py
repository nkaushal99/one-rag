import unittest

from one_rag.ragas_evaluation import FastEmbedRagasEmbeddings, aggregate, is_abstention, is_rate_limited, map_metric_scores, retry_delay_seconds, select_winner, validate_dataset, validate_manifest


def manifest():
    return {
        "dataset_version": "1.0.0",
        "judge": {"model": "gemini-3.5-flash-lite", "temperature": 0},
        "embedding": {"model": "BAAI/bge-small-en-v1.5", "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"},
        "rate_limit": {"requests_per_minute": 12, "max_retry_delay_seconds": 60},
        "metrics": ["context_precision", "context_recall", "faithfulness", "answer_correctness", "answer_relevancy"],
        "configurations": [{"name": f"config-{index}"} for index in range(5)],
    }


class FakeEmbedder:
    def embed(self, texts):
        return ([float(len(text)), 1.0] for text in texts)


class RagasEvaluationTests(unittest.TestCase):
    def test_manifest_requires_fixed_model_and_revision(self):
        validate_manifest(manifest())
        invalid = manifest()
        invalid["judge"]["temperature"] = 0.1
        with self.assertRaises(ValueError):
            validate_manifest(invalid)

    def test_dataset_requires_six_answerable_and_two_negatives(self):
        rows = [{"id": str(index), "answerable": True, "question": "q", "reference_answer": "a", "reference_context": "c"} for index in range(6)]
        rows += [{"id": "negative-1", "answerable": False, "question": "q"}, {"id": "negative-2", "answerable": False, "question": "q"}]
        validate_dataset(rows)
        with self.assertRaises(ValueError):
            validate_dataset(rows[:-1])

    def test_fastembed_adapter_maps_embeddings(self):
        adapter = FastEmbedRagasEmbeddings("fake", embedder=FakeEmbedder()).as_ragas()
        self.assertEqual(adapter.embed_query("abc"), [3.0, 1.0])
        self.assertEqual(adapter.embed_documents(["a", "bb"]), [[1.0, 1.0], [2.0, 1.0]])

    def test_aggregation_maps_metrics_and_tie_breaks_on_latency(self):
        scores = [{"context_precision": 1.0, "context_recall": 0.8, "faithfulness": 0.7, "answer_correctness": 0.6, "answer_relevancy": 0.9}]
        fast = aggregate("fast", scores, [30, 40, 50])
        slow = aggregate("slow", [{key: value + 0.01 for key, value in scores[0].items()}], [80, 90, 100])
        self.assertEqual(fast["context_precision"], 1.0)
        self.assertEqual(select_winner([slow, fast])["configuration"], "fast")

    def test_metric_mapping_rejects_incomplete_ragas_scores(self):
        score = {"context_precision": 1, "context_recall": 1, "faithfulness": 1, "answer_correctness": 1, "answer_relevancy": 1}
        self.assertEqual(map_metric_scores([score])[0]["answer_correctness"], 1.0)
        with self.assertRaises(ValueError):
            map_metric_scores([{"context_precision": 1}])

    def test_abstention_is_deterministic(self):
        self.assertTrue(is_abstention("I do not know based on the provided context."))
        self.assertFalse(is_abstention("The handbook does not say, but I think it is 1.29."))

    def test_provider_retry_delay_is_read_and_capped(self):
        self.assertEqual(retry_delay_seconds(Exception("Please retry in 43.2 seconds."), 60), 44)
        self.assertEqual(retry_delay_seconds(Exception("retry after 120 seconds"), 60), 60)
        self.assertTrue(is_rate_limited(Exception("429 ResourceExhausted quota exceeded")))
        self.assertFalse(is_rate_limited(Exception("invalid API key")))
