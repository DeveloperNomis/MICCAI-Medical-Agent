from langchain_core.runnables import Runnable


class LocalLLMAdapter(Runnable):
    """
    Minimal LangChain-compatible adapter for the local LLM backend.
    """

    def __init__(self, local_llm):
        self.local_llm = local_llm

    def invoke(self, input, config=None):
        """
        Convert LangChain prompt objects to plain text and run local generation.
        """

        # Extract a string from LangChain PromptValue-like inputs.
        if hasattr(input, "to_string"):
            input = input.to_string()

        elif hasattr(input, "text"):
            input = input.text

        elif not isinstance(input, str):
            input = str(input)

        # The input is now a formatted prompt string.
        return self.local_llm.generate(input, max_new_tokens=50)