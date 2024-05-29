"""LlamaGuard provider using Together.ai."""

from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from modelgauge.suts.together_client import (
    TogetherCompletionsRequest
) 
from modelgauge.suts.together_client import TogetherCompletionsSUT

class TogetherProvider:
    """Something that implements the Llama Guard Annotator protocol."""
    def __init__(self, model_name, secrets):
        self.sut = TogetherCompletionsSUT("annotator", model_name, secrets)

    @staticmethod
    def make_provider_completions_request(prompt, model, max_tokens, n):
        return TogetherCompletionsRequest(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            n=n,
        )