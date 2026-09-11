from unittest import TestCase

from one_rag.chunking import chunk_text, token_count


class ChunkingTests(TestCase):
    def test_fixed_chunks_use_a_500_token_budget(self) -> None:
        text = " ".join(f"word{number}" for number in range(650))
        chunks = chunk_text(text, strategy="fixed", chunk_size=500)
        self.assertEqual([len(chunk.split()) for chunk in chunks], [500, 150])

    def test_fixed_overlap_repeats_the_last_100_tokens(self) -> None:
        text = " ".join(f"word{number}" for number in range(650))
        chunks = chunk_text(text, strategy="fixed_overlap", chunk_size=500, chunk_overlap=100)
        self.assertEqual([len(chunk.split()) for chunk in chunks], [500, 250])
        self.assertTrue(chunks[1].startswith("word400"))

    def test_sentence_chunking_does_not_cut_a_sentence(self) -> None:
        self.assertEqual(chunk_text("First sentence. Second sentence!", strategy="sentence"), ["First sentence.", "Second sentence!"])

    def test_paragraph_and_section_boundaries_are_preserved(self) -> None:
        text = "# Architecture\n\nFirst paragraph.\n\nSecond paragraph.\n\n# Recovery\n\nRestore service."
        self.assertEqual(len(chunk_text(text, strategy="paragraph")), 5)
        self.assertEqual(chunk_text(text, strategy="section"), ["# Architecture\n\nFirst paragraph.\n\nSecond paragraph.", "# Recovery\n\nRestore service."])

    def test_section_chunking_keeps_numbered_lists_inside_their_top_level_section(self) -> None:
        text = "1. RELIABILITY AND RETRY\nRetry details.\n1. This numbered list item stays here.\n\n2. RECOVERY\nRestore details."
        self.assertEqual(chunk_text(text, strategy="section"), ["1. RELIABILITY AND RETRY\nRetry details.\n1. This numbered list item stays here.", "2. RECOVERY\nRestore details."])

    def test_token_count_is_visible_and_deterministic(self) -> None:
        self.assertEqual(token_count("One, two."), 4)
