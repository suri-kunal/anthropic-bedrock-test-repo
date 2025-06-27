# Migration Guide: Anthropic API to AWS Bedrock Converse API

This guide helps you migrate from the original Anthropic API implementation to the AWS Bedrock Converse API version.

## Key Changes Overview

| Aspect | Original (Anthropic) | Migrated (Bedrock) |
|--------|---------------------|-------------------|
| **Client** | `anthropic.Anthropic` | `boto3.client("bedrock-runtime")` |
| **Authentication** | `ANTHROPIC_API_KEY` | AWS credentials |
| **Model IDs** | `claude-sonnet-4-20250514` | `anthropic.claude-3-7-sonnet-20250219-v1:0` |
| **API Method** | `client.messages.create()` | `client.converse()` |
| **Image Support** | Limited | Native support with base64 encoding |
| **Reasoning** | Not available | Full reasoning capabilities with Claude 3.7/4 |
| **Tools/MCP** | Full support | Removed (simplified) |
| **Response Format** | Anthropic format | Bedrock format |

## Code Migration Examples

### 1. Basic Agent Creation

**Before (Anthropic):**
```python
from anthropic import Anthropic
from agents.agent import Agent

agent = Agent(
    name="MyAgent",
    system="You are helpful",
    client=Anthropic(api_key="your-key"),
    verbose=True
)
```

**After (Bedrock):**
```python
import boto3
from agents.agent import Agent, ModelConfig

agent = Agent(
    name="MyAgent",
    system="You are helpful",
    config=ModelConfig(
        model="anthropic.claude-3-7-sonnet-20250219-v1:0"
    ),
    region="us-east-1",
    verbose=True
)
```

### 2. Model Configuration

**Before (Anthropic):**
```python
config = ModelConfig(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    temperature=1.0
)
```

**After (Bedrock):**
```python
config = ModelConfig(
    model="anthropic.claude-3-7-sonnet-20250219-v1:0",  # Full Bedrock model ID
    max_tokens=4096,
    temperature=1.0,
    top_p=0.9  # New parameter
)
```

### 3. Response Handling

**Before (Anthropic):**
```python
response = agent.run("Hello")
text = response.content[0].text
usage = response.usage
```

**After (Bedrock):**
```python
response = agent.run("Hello")
text = response["output"]["message"]["content"][0]["text"]
usage = response.get("usage", {})
```

### 4. Image Support (New Feature)

**New in Bedrock:**
```python
# Single image
response = agent.run(
    "Analyze this image", 
    image_paths=["chart.png"]
)

# Multiple images
response = agent.run(
    "Compare these images",
    image_paths=["image1.jpg", "image2.png"]
)

# Convenience methods
response = agent.add_image_from_file("photo.jpg", "What's in this photo?")
response = agent.add_images_from_files(["a.png", "b.png"], "Compare these")
```

### 4.1. Document Support (New Feature)

**New in Bedrock - Document Upload:**
```python
# Single document
response = agent.run(
    "Analyze this data file",
    document_paths=["data.json"]
)

# Multiple documents (max 5 files, 4.5MB each)
response = agent.run(
    "Compare these reports",
    document_paths=["report1.md", "data.json", "config.yaml"]
)

# Mixed content: images + documents
response = agent.run(
    "Analyze the chart and compare with the data file",
    image_paths=["chart.png"],
    document_paths=["data.json"]
)

# Document convenience methods
response = agent.add_document_from_file("report.txt", "Summarize this report")
response = agent.add_documents_from_files(["data.json", "config.yaml"], "Compare these")
response = agent.add_mixed_files("Analyze together", ["chart.png"], ["data.json"])

# Supported document formats: txt, md, json, csv, xml, yaml, html, pdf, docx
```

### 4.2. Chat Functionality (New Feature)

**New in Bedrock - Ongoing Conversations:**
```python
# Create a chat agent
agent = Agent(
    name="Chat Agent",
    system="You are a helpful assistant. Remember our conversation context.",
    config=ModelConfig(
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        enable_reasoning=True
    ),
    verbose=False  # Less verbose for cleaner chat
)

# Have an ongoing conversation
response1 = agent.chat("Hi! I'm working on a Python project.")
response2 = agent.chat("What are some good libraries for data analysis?")  # Context remembered
response3 = agent.chat("I'm specifically working with financial data.")    # Context continues

# Add files during conversation
agent.chat_with_files("Here's my data file", document_paths=["data.json"])
agent.chat_with_files("And here's a chart", image_paths=["chart.png"])

# Interactive terminal chat with commands
agent.start_interactive_chat()  # Starts interactive session with:
# - /image <path> - Add image
# - /doc <path> - Add document  
# - /files <img1,img2> <doc1,doc2> - Add multiple files
# - /history - View chat history
# - /clear - Clear history
# - /quit - Exit chat

# Chat history management
agent.print_chat_history()              # Pretty print conversation
history = agent.get_chat_history()      # Get messages list
agent.export_chat_history("chat.json")  # Export to file
agent.clear_chat_history()              # Reset conversation
```

### 5. Reasoning Capabilities (New Feature)

**New in Bedrock - Reasoning with Claude 3.7/4:**
```python
# Enable reasoning in model config
config = ModelConfig(
    model="anthropic.claude-3-7-sonnet-20250219-v1:0",
    enable_reasoning=True,  # Enable reasoning
    reasoning_budget_tokens=2000  # Token budget for reasoning process
)

# Create agent with reasoning
agent = Agent(
    name="Reasoning Agent",
    system="Use step-by-step reasoning for complex problems",
    config=config,
    show_reasoning=True  # Display reasoning in verbose mode
)

# Complex problem with reasoning
response = agent.run("""
A train leaves Station A at 2 PM traveling east at 60 mph.
Another train leaves Station B (300 miles east of A) at 3 PM traveling west at 80 mph.
At what time and location will they meet?
""")

# Access reasoning and response separately
reasoning, answer = agent.get_reasoning_and_response("What's the derivative of x²?")
print(f"Reasoning: {reasoning}")
print(f"Answer: {answer}")

# Get only reasoning or only response
reasoning_only = agent.get_reasoning_only("Explain photosynthesis")
response_only = agent.get_response_only("Explain photosynthesis")
```

## Available Models

### Claude 3.7 Models
- `anthropic.claude-3-7-sonnet-20250219-v1:0` (Recommended)

### Claude 4 Models (if available in your region)
- `anthropic.claude-sonnet-4-20250514-v1:0`

## AWS Setup Requirements

### 1. AWS Credentials
Configure AWS credentials using one of these methods:

**AWS CLI:**
```bash
aws configure
```

**Environment Variables:**
```bash
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_DEFAULT_REGION=us-east-1
```

**IAM Role (recommended for EC2/Lambda):**
Attach appropriate IAM role with Bedrock permissions.

### 2. Bedrock Model Access
Ensure you have access to the models you want to use:

1. Go to AWS Bedrock Console
2. Navigate to "Model access"
3. Request access to Anthropic Claude models
4. Wait for approval (usually immediate for Claude models)

### 3. Required IAM Permissions
Your AWS credentials need these permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/anthropic.claude*"
            ]
        }
    ]
}
```

## Migration Checklist

- [ ] Install boto3: `pip install boto3`
- [ ] Configure AWS credentials
- [ ] Request Bedrock model access
- [ ] Update imports (remove `anthropic`, add `boto3`)
- [ ] Update model IDs to Bedrock format
- [ ] Update response parsing logic
- [ ] Configure reasoning parameters if using Claude 3.7/4
- [ ] Remove tool/MCP code if not needed
- [ ] Test image functionality if needed
- [ ] Test document upload functionality
- [ ] Test reasoning capabilities with complex problems
- [ ] Test chat functionality and context memory
- [ ] Update error handling for boto3 exceptions

## Cost Considerations

### Bedrock Pricing
- Pay per token (input/output)
- No monthly commitments
- Pricing varies by model and region
- Check [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) for latest rates

### Comparison
- Bedrock often provides cost advantages for high-volume usage
- Direct Anthropic API may be simpler for low-volume development
- Consider data residency and compliance requirements

## Troubleshooting

### Common Issues

**1. AccessDeniedException**
- Ensure proper IAM permissions
- Verify model access is granted in Bedrock console

**2. ModelNotReadyException**
- Request access to the model in Bedrock console
- Wait for approval

**3. Region Issues**
- Not all models available in all regions
- Use `us-east-1` or `us-west-2` for best model availability

**4. Image Format Errors**
- Supported formats: JPEG, PNG, GIF, WebP
- Maximum image size: 5MB (varies by model)
- Ensure proper base64 encoding

### Error Handling Example

```python
from botocore.exceptions import ClientError

try:
    response = agent.run("Hello")
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == 'AccessDeniedException':
        print("Check your IAM permissions and model access")
    elif error_code == 'ModelNotReadyException':
        print("Request access to this model in Bedrock console")
    else:
        print(f"Bedrock error: {e}")
```

## Benefits of Migration

1. **Reasoning Capabilities**: Access Claude's step-by-step thinking process with Claude 3.7/4
2. **Chat Functionality**: Ongoing conversations with context memory and file sharing
3. **Multi-model Support**: Easy switching between Claude versions and other models
4. **AWS Integration**: Better integration with other AWS services
5. **Document Analysis**: Native support for text files, JSON, CSV, markdown, etc.
6. **Image Analysis**: Native image processing capabilities with reasoning
7. **Interactive Tools**: Terminal chat interface with file upload commands
8. **Cost Control**: AWS billing and cost management tools
9. **Enterprise Features**: VPC endpoints, CloudTrail logging, etc.
10. **Scalability**: AWS infrastructure for high-volume applications
11. **Transparency**: See how Claude approaches and solves complex problems

## Need Help?

- Check the `bedrock_agent_demo.ipynb` for working examples
- Review AWS Bedrock documentation
- Test with simple examples before migrating complex workflows 