# Migration Guide: Converse API to Invoke API for Extended Thinking

This guide explains how to migrate from the Converse API implementation to the Invoke API implementation for enhanced reasoning capabilities with Claude models on AWS Bedrock.

## 🎯 Why Migrate to Invoke API?

The Invoke API provides more direct control over the Anthropic Messages API format, which can offer enhanced support for extended thinking and reasoning capabilities. While both APIs support reasoning, the Invoke API may provide:

- **More granular control** over reasoning parameters
- **Better thinking format handling** with the native Messages API
- **Enhanced debugging** of reasoning processes
- **Direct access** to Anthropic's native message format

## 📁 New Files Added

### Core Implementation
- `agents/agent_invoke.py` - Main agent class using Invoke API
- `agents/utils/history_util_invoke.py` - Message history management for Invoke API
- `agents/test_reasoning_invoke.py` - Comprehensive testing suite
- `agents/chat_demo_invoke.py` - Interactive chat demo

## 🔄 Key Differences

### API Format Differences

#### Converse API Format
```python
# Converse API uses Bedrock-native format
params = {
    "modelId": self.config.model,
    "messages": self.history.format_for_bedrock(),
    "inferenceConfig": {
        "maxTokens": self.config.max_tokens,
        "temperature": self.config.temperature,
    },
    "additionalModelRequestFields": {
        "thinking": {
            "type": "enabled",
            "budget_tokens": self.config.reasoning_budget_tokens
        }
    }
}
response = self.client.converse(**params)
```

#### Invoke API Format
```python
# Invoke API uses native Anthropic Messages format
request_body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": self.config.max_tokens,
    "temperature": self.config.temperature,
    "messages": formatted_messages,
    "system": self.system,
    "thinking": {
        "enabled": True,
        "max_tokens": self.config.reasoning_budget_tokens
    }
}
params = {
    "modelId": self.config.model,
    "body": json.dumps(request_body),
    "contentType": "application/json"
}
response = self.client.invoke_model(**params)
```

### Class Name Changes
- `Agent` → `AgentInvoke`
- `MessageHistory` → `MessageHistoryInvoke`

### Import Changes
```python
# Old (Converse API)
from agents.agent import Agent, ModelConfig

# New (Invoke API)
from agents.agent_invoke import AgentInvoke, ModelConfig
```

## 🚀 Migration Steps

### Step 1: Update Imports
```python
# Replace this:
from agents.agent import Agent, ModelConfig

# With this:
from agents.agent_invoke import AgentInvoke, ModelConfig
```

### Step 2: Update Class Instantiation
```python
# Replace this:
agent = Agent(
    name="My Agent",
    system="You are a helpful assistant.",
    config=ModelConfig(enable_reasoning=True),
    show_reasoning=True
)

# With this:
agent = AgentInvoke(
    name="My Agent", 
    system="You are a helpful assistant.",
    config=ModelConfig(enable_reasoning=True),
    show_reasoning=True
)
```

### Step 3: All Methods Remain the Same!
The API is designed to be compatible, so all your existing method calls work unchanged:

```python
# These all work the same way:
response = agent.run("Your question here")
reasoning, answer = agent.get_reasoning_and_response("Your question")
agent.chat("Follow-up question")
agent.start_interactive_chat()

# File operations work the same:
agent.add_image_from_file("image.png", "Describe this image")
agent.add_document_from_file("data.json", "Analyze this data")
```

## 🆚 Side-by-Side Comparison

### Basic Usage

#### Converse API
```python
from agents.agent import Agent, ModelConfig

agent = Agent(
    name="Assistant",
    system="You are helpful.",
    config=ModelConfig(
        enable_reasoning=True,
        reasoning_budget_tokens=2000
    ),
    show_reasoning=True
)

response = agent.run("Solve this math problem: 2x + 5 = 15")
```

#### Invoke API
```python
from agents.agent_invoke import AgentInvoke, ModelConfig

agent = AgentInvoke(
    name="Assistant", 
    system="You are helpful.",
    config=ModelConfig(
        enable_reasoning=True,
        reasoning_budget_tokens=2000
    ),
    show_reasoning=True
)

response = agent.run("Solve this math problem: 2x + 5 = 15")
```

### Advanced Usage

Both implementations support the same advanced features:

```python
# Complex reasoning with high token budget
config = ModelConfig(
    model="anthropic.claude-3-7-sonnet-20250219-v1:0",
    enable_reasoning=True,
    reasoning_budget_tokens=4000,  # High budget for complex thinking
    temperature=0.3
)

# Multi-modal inputs
response = agent.run(
    "Analyze this chart and data together",
    image_paths=["chart.png"],
    document_paths=["data.json"]
)

# Separate reasoning and response
reasoning, answer = agent.get_reasoning_and_response(
    "What's the solution to the traveling salesman problem for 5 cities?"
)
```

## 🧪 Testing Both Implementations

### Run Comparison Tests
```bash
# Test Invoke API implementation
python agents/test_reasoning_invoke.py

# Test original Converse API
python agents/test_reasoning.py

# The new test includes a direct comparison!
```

### Interactive Demos
```bash
# Try the new Invoke API chat
python agents/chat_demo_invoke.py

# Compare with original chat
python agents/chat_demo.py
```

## 📊 Configuration Options

Both implementations use the same `ModelConfig`:

```python
@dataclass
class ModelConfig:
    model: str = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.9
    context_window_tokens: int = 180000
    enable_reasoning: bool = True  # Enable extended thinking
    reasoning_budget_tokens: int = 2000  # Token budget for reasoning
```

### Reasoning Budget Recommendations
- **Simple problems**: 500-1000 tokens
- **Medium complexity**: 1500-2500 tokens  
- **Complex reasoning**: 3000-4000 tokens
- **Maximum budget**: 4000 tokens (model limit)

## 🔍 Debugging and Monitoring

### Enhanced Reasoning Display
The Invoke API provides better access to reasoning content:

```python
# Enable detailed reasoning display
agent = AgentInvoke(
    name="Debug Agent",
    system="Think step by step.",
    config=ModelConfig(enable_reasoning=True),
    verbose=True,
    show_reasoning=True  # Shows <thinking> blocks
)

# Get reasoning separately for analysis
reasoning, response = agent.get_reasoning_and_response("Your question")
print(f"Reasoning length: {len(reasoning)} characters")
print(f"Reasoning content: {reasoning}")
```

### Response Format Differences

#### Converse API Response
```python
{
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "response"}, {"reasoning": "thinking"}]
        }
    },
    "usage": {"inputTokens": 100, "outputTokens": 50}
}
```

#### Invoke API Response (Enhanced)
```python
{
    "output": {
        "message": {
            "role": "assistant", 
            "content": [{"type": "text", "text": "response"}]
        }
    },
    "usage": {"input_tokens": 100, "output_tokens": 50},
    "thinking": "detailed reasoning content",  # Direct access!
    "stopReason": "end_turn"
}
```

## 🛡️ Error Handling

Both implementations handle the same error types:

```python
try:
    response = agent.run("Your question")
except ClientError as e:
    print(f"AWS error: {e}")
except FileNotFoundError as e:
    print(f"File error: {e}")
except ValueError as e:
    print(f"Validation error: {e}")
```

## 🎛️ Performance Considerations

### When to Use Invoke API
- ✅ **Complex reasoning tasks** requiring step-by-step thinking
- ✅ **Debugging reasoning** processes
- ✅ **Maximum control** over message formatting
- ✅ **Research and analysis** applications

### When to Use Converse API  
- ✅ **Simple conversational** applications
- ✅ **Multi-model** compatibility (works with more Bedrock models)
- ✅ **Standardized interface** across different model providers
- ✅ **Production applications** requiring consistency

## 📝 Migration Checklist

- [ ] Update imports (`Agent` → `AgentInvoke`)
- [ ] Test reasoning capabilities with complex questions
- [ ] Verify file upload functionality (images/documents)
- [ ] Test chat history and conversation flow
- [ ] Validate error handling
- [ ] Run comparison tests
- [ ] Update any custom integrations

## 🔧 Troubleshooting

### Common Issues

1. **No reasoning output**: Check `enable_reasoning=True` and `reasoning_budget_tokens > 0`
2. **Import errors**: Ensure you're importing from `agent_invoke` not `agent`
3. **Response format differences**: Use the enhanced methods like `get_reasoning_and_response()`

### Debugging Steps
```python
# Enable verbose logging
agent = AgentInvoke(
    name="Debug",
    system="Debug mode",
    verbose=True,
    show_reasoning=True
)

# Check configuration
print(f"Reasoning enabled: {agent.config.enable_reasoning}")
print(f"Budget: {agent.config.reasoning_budget_tokens}")

# Test simple reasoning
reasoning, response = agent.get_reasoning_and_response("What is 2+2?")
print(f"Got reasoning: {reasoning is not None}")
```

## 🚀 Next Steps

1. **Test the implementation**: Run `python agents/test_reasoning_invoke.py`
2. **Try the chat demo**: Run `python agents/chat_demo_invoke.py`
3. **Compare results**: Use complex reasoning prompts to see differences
4. **Integrate gradually**: Migrate one use case at a time
5. **Monitor performance**: Compare reasoning quality and response times

## 📚 Additional Resources

- [AWS Bedrock Invoke API Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html)
- [Anthropic Messages API Guide](https://docs.anthropic.com/en/api/messages)
- [Extended Thinking Examples](agents/test_reasoning_invoke.py)

Both implementations will continue to be maintained, so you can choose based on your specific needs for reasoning capabilities and control over the message format. 