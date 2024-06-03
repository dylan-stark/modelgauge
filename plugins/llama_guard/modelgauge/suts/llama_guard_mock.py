from typing import List, Optional
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.sut import PromptResponseSUT, SUTCompletion, SUTResponse
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from pydantic import BaseModel

class LlamaGuardRequest(BaseModel):
    """The behavior of this sut only depends on the Prompt text."""

    text: str

class LogProbs(BaseModel):
    tokens: List[str]
    token_logprobs: List[float]


class LlamaGuardResponse(BaseModel):
    """This SUT is only capable of returning text."""

    class Choice(BaseModel):
        text: str
        logprobs: Optional[LogProbs] = None

    choices: List[Choice]


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class LlamaGuardMockSUT(PromptResponseSUT[LlamaGuardRequest, LlamaGuardResponse]):
    """This SUT provides a mock llama guard for development and testing."""

    def translate_text_prompt(self, prompt: TextPrompt) -> LlamaGuardRequest:
        return LlamaGuardRequest(text=prompt.text)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> LlamaGuardRequest:
        return LlamaGuardRequest(text=format_chat(prompt))

    def evaluate(self, request: LlamaGuardRequest) -> LlamaGuardResponse:
        return LlamaGuardResponse(**{"choices": [{"text": "safe"}]})

    def translate_response(
        self, request: LlamaGuardRequest, response: LlamaGuardResponse
    ) -> SUTResponse:
        completions = []
        for choice in response.choices:
            completions.append(SUTCompletion(text=choice.text))
        return SUTResponse(completions=completions)


SUTS.register(LlamaGuardMockSUT, "llama-guard-2-mock")
