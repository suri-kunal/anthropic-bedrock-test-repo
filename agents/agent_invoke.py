"""Agent implementation with AWS Bedrock Invoke API for extended thinking support."""

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.exceptions import ClientError

from .utils.history_util_invoke import MessageHistoryInvoke


@dataclass
class ModelConfig:
    """Configuration settings for Claude model parameters on Bedrock."""

    model: str = "anthropic.claude-3-7-sonnet-20250219-v1:0"  # Claude 3.7 Sonnet
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.9
    context_window_tokens: int = 180000
    enable_reasoning: bool = True  # Enable reasoning for Claude 3.7/4 models
    reasoning_budget_tokens: int = 2000  # Token budget for reasoning process


class AgentInvoke:
    """Claude-powered agent using AWS Bedrock Invoke API with extended thinking support."""

    def __init__(
        self,
        name: str,
        system: str,
        config: ModelConfig | None = None,
        verbose: bool = False,
        region: str = "us-east-1",
        message_params: dict[str, Any] | None = None,
        show_reasoning: bool = False,
    ):
        """Initialize an Agent with Invoke API.
        
        Args:
            name: Agent identifier for logging
            system: System prompt for the agent
            config: Model configuration with defaults
            verbose: Enable detailed logging
            region: AWS region for Bedrock client
            message_params: Additional parameters for invoke API call
            show_reasoning: Whether to display reasoning output in verbose mode
        """
        self.name = name
        self.system = system
        self.verbose = verbose
        self.show_reasoning = show_reasoning
        self.config = config or ModelConfig()
        self.message_params = message_params or {}
        
        # Initialize Bedrock client
        self.client = boto3.client("bedrock-runtime", region_name=region)
        
        self.history = MessageHistoryInvoke(
            model=self.config.model,
            system=self.system,
            context_window_tokens=self.config.context_window_tokens,
            client=self.client,
        )

        if self.verbose:
            print(f"\n[{self.name}] Agent initialized with Bedrock Invoke API")

    def _prepare_invoke_params(self) -> dict[str, Any]:
        """Prepare parameters for Bedrock invoke_model() call using Messages API format."""
        # Format messages for Anthropic Messages API
        formatted_messages = self.history.format_for_invoke_api()
        
        # Base request body following Anthropic Messages API format
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "messages": formatted_messages,
            **self.message_params,
        }
        
        # Add system prompt if provided
        if self.system:
            request_body["system"] = self.system
        
        # Add reasoning configuration for extended thinking
        if self.config.enable_reasoning:
            request_body["thinking"] = {
                "enabled": True,
                "max_tokens": self.config.reasoning_budget_tokens
            }
            
        return {
            "modelId": self.config.model,
            "body": json.dumps(request_body),
            "contentType": "application/json"
        }

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        """Encode an image file to base64 and determine format."""
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        # Determine image format from file extension
        ext = os.path.splitext(image_path)[1].lower()
        format_map = {
            '.jpg': 'jpeg',
            '.jpeg': 'jpeg', 
            '.png': 'png',
            '.gif': 'gif',
            '.webp': 'webp'
        }
        image_format = format_map.get(ext, 'jpeg')
        
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        return encoded_image, image_format

    def _encode_document(self, document_path: str) -> tuple[str, str]:
        """Encode a document file to base64 and determine format."""
        with open(document_path, "rb") as doc_file:
            doc_data = doc_file.read()
        
        # Check file size (Bedrock limit: 4.5MB per document)
        max_size = 4.5 * 1024 * 1024  # 4.5MB in bytes
        if len(doc_data) > max_size:
            raise ValueError(f"Document {document_path} is too large. Max size: 4.5MB, actual: {len(doc_data)/1024/1024:.1f}MB")
        
        # Determine document format from file extension
        ext = os.path.splitext(document_path)[1].lower()
        format_map = {
            '.txt': 'txt',
            '.md': 'md',
            '.json': 'json',
            '.csv': 'csv',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.html': 'html',
            '.pdf': 'pdf',
            '.docx': 'docx'
        }
        doc_format = format_map.get(ext, 'txt')
        
        encoded_doc = base64.b64encode(doc_data).decode('utf-8')
        return encoded_doc, doc_format

    def _extract_reasoning_and_response(self, response_data: dict[str, Any]) -> tuple[str | None, str]:
        """Extract reasoning and final response from invoke API response."""
        reasoning_text = None
        response_text = ""
        
        # Handle different response formats
        if "content" in response_data:
            content_blocks = response_data["content"]
            for block in content_blocks:
                if block.get("type") == "text":
                    response_text += block.get("text", "")
                elif block.get("type") == "thinking":
                    reasoning_text = block.get("content", "")
        
        # Alternative format - check for thinking at top level
        if "thinking" in response_data:
            reasoning_text = response_data["thinking"]
            
        return reasoning_text, response_text

    async def _agent_loop(self, user_input: str | dict[str, Any]) -> dict[str, Any]:
        """Process user input and get response from Bedrock using Invoke API."""
        if self.verbose:
            if isinstance(user_input, str):
                print(f"\n[{self.name}] Received: {user_input}")
            else:
                content_types = []
                for c in user_input.get('content', []):
                    if c.get('type') == 'text':
                        content_types.append('text')
                    elif c.get('type') == 'image':
                        content_types.append('image')
                    elif c.get('type') == 'document':
                        doc_name = c.get('source', {}).get('name', 'unknown')
                        content_types.append(f'document({doc_name})')
                    else:
                        content_types.append('unknown')
                print(f"\n[{self.name}] Received message with content: {', '.join(content_types)}")
        
        # Add user message to history
        await self.history.add_message("user", user_input, None)

        # Truncate history if needed
        self.history.truncate()
        
        # Prepare parameters for Bedrock Invoke API
        params = self._prepare_invoke_params()

        try:
            # Call Bedrock Invoke API
            response = self.client.invoke_model(**params)
            
            # Parse the response
            response_body = response["body"].read()
            response_data = json.loads(response_body)
            
            if self.verbose:
                # Extract reasoning and response text
                reasoning_text, response_text = self._extract_reasoning_and_response(response_data)
                
                # Display reasoning if enabled and available
                if self.show_reasoning and reasoning_text:
                    print(f"\n[{self.name}] Reasoning:")
                    print(f"<thinking>\n{reasoning_text}\n</thinking>")
                
                # Display final response
                if response_text:
                    print(f"\n[{self.name}] Output: {response_text}")

            # Add assistant response to history
            # Convert response to format expected by history
            if "content" in response_data:
                assistant_content = response_data["content"]
            else:
                # Fallback format
                assistant_content = [{"type": "text", "text": response_text}]
                
            await self.history.add_message(
                "assistant", 
                assistant_content, 
                response_data.get("usage")
            )

            # Return response in format similar to converse API for compatibility
            formatted_response = {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": assistant_content
                    }
                },
                "usage": response_data.get("usage", {}),
                "stopReason": response_data.get("stop_reason", "end_turn"),
                "thinking": reasoning_text  # Include thinking for easy access
            }

            return formatted_response

        except ClientError as e:
            error_msg = f"ERROR: Can't invoke '{self.config.model}'. Reason: {e}"
            print(error_msg)
            raise

    async def run_async(self, user_input: str, image_paths: list[str] = None, document_paths: list[str] = None) -> dict[str, Any]:
        """Run agent asynchronously with optional image and document inputs."""
        # Handle text-only input
        if image_paths is None and document_paths is None:
            return await self._agent_loop(user_input)
        
        # Handle input with images and/or documents
        content = []
        
        # Add text content
        if user_input:
            content.append({"type": "text", "text": user_input})
        
        # Add image content
        if image_paths:
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
                encoded_image, image_format = self._encode_image(image_path)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{image_format}",
                        "data": encoded_image
                    }
                })
        
        # Add document content  
        if document_paths:
            if len(document_paths) > 5:
                raise ValueError("Bedrock supports maximum 5 documents per request")
            
            for doc_path in document_paths:
                if not os.path.exists(doc_path):
                    raise FileNotFoundError(f"Document file not found: {doc_path}")
                
                encoded_doc, doc_format = self._encode_document(doc_path)
                content.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": f"application/{doc_format}",
                        "data": encoded_doc,
                        "name": os.path.basename(doc_path)
                    }
                })
        
        message_input = {"content": content}
        return await self._agent_loop(message_input)

    def run(self, user_input: str, image_paths: list[str] = None, document_paths: list[str] = None) -> dict[str, Any]:
        """Run agent synchronously with optional image and document inputs."""
        return asyncio.run(self.run_async(user_input, image_paths, document_paths))

    def add_image_from_file(self, file_path: str, text: str = "") -> dict[str, Any]:
        """Convenience method to send a message with an image file."""
        return self.run(text, [file_path])

    def add_images_from_files(self, file_paths: list[str], text: str = "") -> dict[str, Any]:
        """Convenience method to send a message with multiple image files."""
        return self.run(text, image_paths=file_paths)

    def add_document_from_file(self, file_path: str, text: str = "") -> dict[str, Any]:
        """Convenience method to send a message with a document file."""
        return self.run(text, document_paths=[file_path])

    def add_documents_from_files(self, file_paths: list[str], text: str = "") -> dict[str, Any]:
        """Convenience method to send a message with multiple document files."""
        return self.run(text, document_paths=file_paths)

    def add_mixed_files(self, text: str = "", image_paths: list[str] = None, document_paths: list[str] = None) -> dict[str, Any]:
        """Convenience method to send a message with both images and documents."""
        return self.run(text, image_paths=image_paths, document_paths=document_paths)

    def get_reasoning_and_response(self, user_input: str, image_paths: list[str] = None, document_paths: list[str] = None) -> tuple[str | None, str]:
        """Get both reasoning and final response separately."""
        response = self.run(user_input, image_paths, document_paths)
        
        # Extract thinking from our formatted response
        reasoning = response.get("thinking")
        
        # Extract response text
        content = response.get("output", {}).get("message", {}).get("content", [])
        response_text = ""
        for block in content:
            if block.get("type") == "text":
                response_text += block.get("text", "")
        
        return reasoning, response_text

    def get_reasoning_only(self, user_input: str, image_paths: list[str] = None, document_paths: list[str] = None) -> str | None:
        """Get only the reasoning process."""
        reasoning, _ = self.get_reasoning_and_response(user_input, image_paths, document_paths)
        return reasoning

    def get_response_only(self, user_input: str, image_paths: list[str] = None, document_paths: list[str] = None) -> str:
        """Get only the final response without reasoning."""
        _, response = self.get_reasoning_and_response(user_input, image_paths, document_paths)
        return response

    # Chat functionality methods
    def chat(self, user_input: str, image_paths: list[str] = None, document_paths: list[str] = None) -> dict[str, Any]:
        """Continue an ongoing conversation with the agent."""
        return self.run(user_input, image_paths, document_paths)

    def chat_with_files(self, user_input: str, image_paths: list[str] = None, document_paths: list[str] = None) -> dict[str, Any]:
        """Add files to the conversation (alias for chat for clarity)."""
        return self.chat(user_input, image_paths, document_paths)

    def start_interactive_chat(self):
        """Start an interactive chat session in the terminal."""
        print(f"\n🤖 Starting interactive chat with {self.name}")
        print("Commands:")
        print("  - Type your message and press Enter")
        print("  - '/image <path>' to add an image")
        print("  - '/doc <path>' to add a document")  
        print("  - '/files <img1,img2> <doc1,doc2>' to add multiple files")
        print("  - '/history' to see conversation history")
        print("  - '/clear' to clear conversation history")
        print("  - '/quit' or '/exit' to end chat")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['/quit', '/exit']:
                    print("👋 Chat ended. Goodbye!")
                    break
                    
                elif user_input.lower() == '/clear':
                    self.clear_chat_history()
                    print("🧹 Chat history cleared.")
                    continue
                    
                elif user_input.lower() == '/history':
                    self.print_chat_history()
                    continue
                    
                elif user_input.startswith('/image '):
                    image_path = user_input[7:].strip()
                    prompt = input("👤 Message for the image: ").strip()
                    if not prompt:
                        prompt = "Please analyze this image."
                    try:
                        self.chat(prompt, image_paths=[image_path])
                    except FileNotFoundError:
                        print(f"❌ Image file not found: {image_path}")
                    continue
                    
                elif user_input.startswith('/doc '):
                    doc_path = user_input[5:].strip()
                    prompt = input("👤 Message for the document: ").strip()
                    if not prompt:
                        prompt = "Please analyze this document."
                    try:
                        self.chat(prompt, document_paths=[doc_path])
                    except FileNotFoundError:
                        print(f"❌ Document file not found: {doc_path}")
                    continue
                    
                elif user_input.startswith('/files '):
                    # Parse /files <img1,img2> <doc1,doc2>
                    parts = user_input[7:].strip().split(' ', 1)
                    image_paths = []
                    document_paths = []
                    
                    if len(parts) >= 1 and parts[0]:
                        image_paths = [p.strip() for p in parts[0].split(',') if p.strip()]
                    if len(parts) >= 2 and parts[1]:
                        document_paths = [p.strip() for p in parts[1].split(',') if p.strip()]
                    
                    prompt = input("👤 Message for the files: ").strip()
                    if not prompt:
                        prompt = "Please analyze these files."
                    
                    try:
                        self.chat(prompt, image_paths=image_paths or None, document_paths=document_paths or None)
                    except FileNotFoundError as e:
                        print(f"❌ {e}")
                    continue
                
                # Regular chat message
                self.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again or type '/quit' to exit.")

    def clear_chat_history(self):
        """Clear the conversation history."""
        self.history = MessageHistoryInvoke(
            model=self.config.model,
            system=self.system,
            context_window_tokens=self.config.context_window_tokens,
            client=self.client,
        )
        if self.verbose:
            print(f"🧹 [{self.name}] Chat history cleared.")

    def get_chat_history(self) -> list[dict[str, Any]]:
        """Get the current chat history."""
        return self.history.messages.copy()

    def print_chat_history(self):
        """Print the current chat history in a readable format."""
        print(f"\n💬 Chat History for {self.name}")
        print("=" * 50)
        
        if not self.history.messages:
            print("No messages in history.")
            return
        
        for i, message in enumerate(self.history.messages, 1):
            role = message["role"]
            content = message["content"]
            
            # Format role
            if role == "user":
                role_symbol = "👤"
            elif role == "assistant":
                role_symbol = "🤖"
            else:
                role_symbol = "❓"
            
            print(f"\n{i}. {role_symbol} {role.title()}:")
            
            # Format content
            for block in content:
                if block.get("type") == "text":
                    text = block["text"]
                    # Truncate long messages
                    if len(text) > 200:
                        text = text[:200] + "..."
                    print(f"   📝 {text}")
                elif block.get("type") == "image":
                    print(f"   🖼️  [Image attached]")
                elif block.get("type") == "document":
                    doc_name = block.get("source", {}).get("name", "unknown")
                    print(f"   📄 [Document: {doc_name}]")
                elif block.get("type") == "thinking":
                    print(f"   🧠 [Reasoning content]")
        
        print(f"\n📊 Total messages: {len(self.history.messages)}")
        print(f"🔢 Total tokens used: {self.history.total_tokens}")

    def export_chat_history(self, filename: str = None) -> str:
        """Export chat history to a file."""
        if filename is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_invoke_{timestamp}.json"
        
        import json
        history_data = {
            "agent_name": self.name,
            "model": self.config.model,
            "system": self.system,
            "total_tokens": self.history.total_tokens,
            "messages": self.history.messages,
            "api_type": "invoke"
        }
        
        with open(filename, "w") as f:
            json.dump(history_data, f, indent=2)
        
        print(f"💾 Chat history exported to: {filename}")
        return filename 