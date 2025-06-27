# Agents

A minimal educational implementation of LLM agents using the Anthropic API.

> **Note:** This is NOT an SDK, but a reference implementation of key concepts

## Overview & Core Components

This repo demonstrates how to [build effective agents](https://www.anthropic.com/engineering/building-effective-agents) with the Anthropic API. It shows how sophisticated AI behaviors can emerge from a simple foundation: LLMs using tools in a loop. This implementation is not prescriptive - the core logic is <300 lines of code and deliberately lacks production features. Feel free to translate these patterns to your language and production stack ([Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) can help!)

It contains three components:

- `agent.py`: Manages Anthropic API interactions and tool execution
- `tools/`: Tool implementations (both native and MCP tools)
- `utils/`: Utilities for message history and MCP server connections

## 🆕 AWS Bedrock Version

We've migrated the agent to support **AWS Bedrock Converse API** with the following features:

### Key Benefits:
- **Multi-model support**: Use Claude 3.7, Claude 4, and other Bedrock models
- **Image analysis**: Built-in support for analyzing images alongside text
- **AWS integration**: Native integration with AWS services and IAM
- **Simplified architecture**: No tools/MCP dependencies for focused use cases

### Bedrock-Specific Features:
- **Reasoning capabilities**: Access Claude's step-by-step thinking process
- **Document upload support**: Analyze txt, md, json, csv, xml, yaml files (up to 5 files, 4.5MB each)
- Text and image input support with reasoning
- Optimized for Claude 3.7/4 models with reasoning capabilities
- Base64 encoding for images (JPEG, PNG, GIF, WebP) and documents
- Convenience methods for single/multiple file analysis
- Mixed content support (images + documents together)
- Configurable reasoning token budgets for complex problems

### Quick Start with Bedrock:

```python
from agents.agent import Agent, ModelConfig

# Create Bedrock agent with reasoning enabled
agent = Agent(
    name="Bedrock Agent",
    system="You are a helpful AI assistant. Use step-by-step reasoning for complex problems.",
    config=ModelConfig(
        model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        max_tokens=4096,
        temperature=0.7,
        enable_reasoning=True,  # Enable reasoning capabilities
        reasoning_budget_tokens=2000  # Token budget for reasoning
    ),
    region="us-east-1",
    show_reasoning=True  # Display reasoning in verbose mode
)

# Complex reasoning problem
response = agent.run("""
Calculate the compound interest on $10,000 invested at 6% annually 
for 10 years, compounded quarterly. Show your step-by-step calculation.
""")

# Access reasoning and response separately
reasoning, final_answer = agent.get_reasoning_and_response(
    "Solve this logic puzzle: If all roses are flowers, and some flowers are red, can we conclude that some roses are red?"
)

# Image analysis with reasoning
response = agent.run(
    "Analyze this chart step by step and explain the mathematical relationship shown", 
    image_paths=["chart.png"]
)

# Document analysis with reasoning
response = agent.run(
    "Analyze this data file and provide insights with step-by-step reasoning",
    document_paths=["data.json"]
)

# Mixed content analysis
response = agent.run(
    "Compare this chart with the data file and explain the correlation",
    image_paths=["chart.png"],
    document_paths=["data.json"]
)

# Convenience methods
response = agent.add_document_from_file("report.md", "Summarize this report")
response = agent.add_mixed_files("Analyze together", ["img.png"], ["data.json"])
```

See `bedrock_agent_demo.ipynb` for comprehensive examples including image analysis and advanced usage patterns.

**Quick Test**: Run `python test_reasoning.py` to test all reasoning capabilities.

## Usage (Original Anthropic Version)

```python
from agents.agent import Agent
from agents.tools.think import ThinkTool

# Create an agent with both local tools and MCP server tools
agent = Agent(
    name="MyAgent",
    system="You are a helpful assistant.",
    tools=[ThinkTool()],  # Local tools
    mcp_servers=[
        {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "mcp_server"],
        },
    ]
)

# Run the agent
response = agent.run("What should I consider when buying a new laptop?")
```

From this foundation, you can add domain-specific tools, optimize performance, or implement custom response handling. We remain deliberately unopinionated - this backbone simply gets you started with fundamentals.

## Requirements

### For Bedrock Version:
- Python 3.8+
- AWS credentials configured (AWS CLI, environment variables, or IAM role)
- Access to Bedrock models in your AWS account
- `boto3` Python library
- See `requirements_bedrock.txt` for full dependencies

### For Original Anthropic Version:
- Python 3.8+
- Anthropic API key (set as `ANTHROPIC_API_KEY` environment variable)
- `anthropic` Python library
- `mcp` Python library