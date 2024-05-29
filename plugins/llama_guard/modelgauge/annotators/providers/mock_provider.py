"""Mock LlamaGuard provider."""

from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from modelgauge.suts.together_client import (
    TogetherCompletionsRequest,
    TogetherCompletionsResponse
) 
from modelgauge.suts.together_client import TogetherCompletionsSUT

class MockLlamaGuardProviderSUT:

    def evaluate(self, request) -> TogetherCompletionsResponse:
        return _make_response("safe")

class MockProvider:
    """Something that implements the Llama Guard Annotator protocol."""

    @staticmethod
    def make_provider_completions_request(prompt, model, max_tokens, n):
        return TogetherCompletionsRequest(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            n=n,
        )
    
def _make_response(text: str) -> TogetherCompletionsResponse:
    return TogetherCompletionsResponse(
        id="some-id",
        choices=[TogetherCompletionsResponse.Choice(text=text)],
        usage=TogetherCompletionsResponse.Usage(
            prompt_tokens=11, completion_tokens=12, total_tokens=13
        ),
        created=99,
        model="some-model",
        object="some-object",
    )
