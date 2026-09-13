from unittest import TestCase
from unittest.mock import patch

from pydantic import SecretStr

from one_rag.answering import AnswerService
from one_rag.settings import Settings


class GeminiConfigurationTests(TestCase):
    @patch("one_rag.answering.ChatGoogleGenerativeAI")
    def test_uses_the_configured_gemini_model_and_local_key(self, chat_model) -> None:
        settings = Settings(google_api_key=SecretStr("test-key"))
        AnswerService(settings, retrieval=object())
        chat_model.assert_called_once_with(
            model="gemini-3.5-flash-lite",
            temperature=0,
            api_key="test-key",
        )
