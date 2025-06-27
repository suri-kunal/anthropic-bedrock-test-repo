"""Message history with token tracking for AWS Bedrock Converse API."""

from typing import Any


class MessageHistory:
    """Manages chat history with token tracking for Bedrock Converse API."""

    def __init__(
        self,
        model: str,
        system: str,
        context_window_tokens: int,
        client: Any,
    ):
        self.model = model
        self.system = system
        self.context_window_tokens = context_window_tokens
        self.messages: list[dict[str, Any]] = []
        self.total_tokens = 0
        self.message_tokens: list[tuple[int, int]] = (
            []
        )  # List of (input_tokens, output_tokens) tuples
        self.client = client

        # Estimate initial tokens for system prompt (rough approximation)
        # Bedrock doesn't provide token counting like Anthropic, so we estimate
        system_token = max(len(self.system) // 4, 10)  # Rough token estimate
        self.total_tokens = system_token

    async def add_message(
        self,
        role: str,
        content: str | list[dict[str, Any]] | dict[str, Any],
        usage: Any | None = None,
    ):
        """Add a message to the history and track token usage."""
        # Handle different content formats
        if isinstance(content, str):
            content = [{"text": content}]
        elif isinstance(content, dict) and "content" in content:
            # Handle the case where content is wrapped in a dict
            content = content["content"]

        message = {"role": role, "content": content}
        self.messages.append(message)

        # Track token usage for assistant responses
        if role == "assistant" and usage:
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
            
            current_turn_input = input_tokens
            self.message_tokens.append((current_turn_input, output_tokens))
            self.total_tokens += current_turn_input + output_tokens

    def truncate(self) -> None:
        """Remove oldest messages when context window limit is exceeded."""
        if self.total_tokens <= self.context_window_tokens:
            return

        TRUNCATION_NOTICE_TOKENS = 25
        TRUNCATION_MESSAGE = {
            "role": "user",
            "content": [
                {
                    "text": "[Earlier history has been truncated.]",
                }
            ],
        }

        def remove_message_pair():
            if len(self.messages) >= 2:
                self.messages.pop(0)
                self.messages.pop(0)

                if self.message_tokens:
                    input_tokens, output_tokens = self.message_tokens.pop(0)
                    self.total_tokens -= input_tokens + output_tokens

        while (
            self.message_tokens
            and len(self.messages) >= 2
            and self.total_tokens > self.context_window_tokens
        ):
            remove_message_pair()

            if self.messages and self.message_tokens:
                original_input_tokens, original_output_tokens = (
                    self.message_tokens[0]
                )
                self.messages[0] = TRUNCATION_MESSAGE
                self.message_tokens[0] = (
                    TRUNCATION_NOTICE_TOKENS,
                    original_output_tokens,
                )
                self.total_tokens += (
                    TRUNCATION_NOTICE_TOKENS - original_input_tokens
                )

    def format_for_bedrock(self) -> list[dict[str, Any]]:
        """Format messages for Bedrock Converse API."""
        result = []
        for message in self.messages:
            # Ensure content is in the right format for Bedrock
            content = message["content"]
            if isinstance(content, str):
                content = [{"text": content}]
            elif not isinstance(content, list):
                content = [content]
            
            # Convert any 'type' + 'text' format to just 'text' format for Bedrock
            bedrock_content = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text" and "text" in block:
                        bedrock_content.append({"text": block["text"]})
                    elif "text" in block:
                        bedrock_content.append({"text": block["text"]})
                    elif "image" in block:
                        bedrock_content.append({"image": block["image"]})
                    else:
                        bedrock_content.append(block)
                else:
                    bedrock_content.append(block)
            
            result.append({
                "role": message["role"],
                "content": bedrock_content
            })
        
        return result
