"""Message history with token tracking for AWS Bedrock Invoke API (Messages API format)."""

from typing import Any


class MessageHistoryInvoke:
    """Manages chat history with token tracking for Bedrock Invoke API using Messages format."""

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
        # Handle different content formats for Messages API
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif isinstance(content, dict) and "content" in content:
            # Handle the case where content is wrapped in a dict
            content = content["content"]
        elif isinstance(content, list):
            # Ensure each content block has the correct format for Messages API
            formatted_content = []
            for block in content:
                if isinstance(block, dict):
                    if "text" in block and "type" not in block:
                        # Convert simple text blocks to proper format
                        formatted_content.append({"type": "text", "text": block["text"]})
                    elif block.get("type") == "text":
                        formatted_content.append(block)
                    elif block.get("type") == "image":
                        formatted_content.append(block)
                    elif block.get("type") == "document":
                        formatted_content.append(block)
                    else:
                        formatted_content.append(block)
                else:
                    formatted_content.append(block)
            content = formatted_content

        message = {"role": role, "content": content}
        self.messages.append(message)

        # Track token usage for assistant responses
        if role == "assistant" and usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            
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
                    "type": "text",
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

    def format_for_invoke_api(self) -> list[dict[str, Any]]:
        """Format messages for Bedrock Invoke API using Messages API format."""
        result = []
        for message in self.messages:
            # Ensure content is in the right format for Messages API
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            elif not isinstance(content, list):
                content = [content]
            
            # Format content blocks for Messages API
            formatted_content = []
            for block in content:
                if isinstance(block, dict):
                    # Handle different block types
                    if block.get("type") == "text":
                        formatted_content.append({
                            "type": "text",
                            "text": block.get("text", "")
                        })
                    elif block.get("type") == "image":
                        # Format image block for Messages API
                        image_block = {"type": "image"}
                        if "source" in block:
                            image_block["source"] = block["source"]
                        elif "image" in block:
                            # Handle Converse API format conversion
                            image_block["source"] = {
                                "type": "base64",
                                "media_type": block["image"].get("format", "image/jpeg"),
                                "data": block["image"]["source"]["bytes"]
                            }
                        formatted_content.append(image_block)
                    elif block.get("type") == "document":
                        # Format document block for Messages API
                        doc_block = {"type": "document"}
                        if "source" in block:
                            doc_block["source"] = block["source"]
                        elif "document" in block:
                            # Handle Converse API format conversion
                            doc_block["source"] = {
                                "type": "base64",
                                "media_type": block["document"].get("format", "text/plain"),
                                "data": block["document"]["source"]["bytes"]
                            }
                        formatted_content.append(doc_block)
                    elif "text" in block:
                        # Handle blocks with text but no type
                        formatted_content.append({
                            "type": "text",
                            "text": block["text"]
                        })
                    else:
                        # Pass through other block types
                        formatted_content.append(block)
                else:
                    # Handle non-dict content
                    formatted_content.append({
                        "type": "text",
                        "text": str(block)
                    })
            
            result.append({
                "role": message["role"],
                "content": formatted_content
            })
        
        return result 