#!/usr/bin/env python3
"""Interactive chat demo for AWS Bedrock Agent with reasoning capabilities."""

import os
import sys
from agents.agent import Agent, ModelConfig


def main():
    """Run an interactive chat demo with the Bedrock agent."""
    
    print("🤖 AWS Bedrock Agent - Interactive Chat Demo")
    print("=" * 50)
    
    # Check for AWS credentials
    if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.path.exists(os.path.expanduser("~/.aws/credentials")):
        print("⚠️  Warning: AWS credentials not found.")
        print("   Please configure AWS credentials before running:")
        print("   - Run: aws configure")
        print("   - Or set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        choice = input("\nContinue anyway? (y/N): ").strip().lower()
        if choice != 'y':
            sys.exit(1)
    
    print("\n🔧 Setting up your chat agent...")
    
    # Get user preferences
    print("\nChoose your agent configuration:")
    print("1. 💬 Fast Chat (reasoning disabled, faster responses)")
    print("2. 🧠 Smart Chat (reasoning enabled, more thoughtful responses)")
    print("3. 🔬 Research Chat (high reasoning budget, detailed analysis)")
    
    choice = input("\nEnter choice (1-3, default: 2): ").strip()
    
    if choice == "1":
        # Fast chat configuration
        config = ModelConfig(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            max_tokens=2048,
            temperature=0.7,
            enable_reasoning=False
        )
        agent_name = "Fast Chat Agent"
        system_prompt = "You are a helpful and conversational AI assistant. Be friendly and concise."
        show_reasoning = False
        
    elif choice == "3":
        # Research chat configuration
        config = ModelConfig(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            max_tokens=4096,
            temperature=0.3,
            enable_reasoning=True,
            reasoning_budget_tokens=3000
        )
        agent_name = "Research Chat Agent"
        system_prompt = "You are a thoughtful research assistant. Use detailed reasoning for complex questions."
        show_reasoning = True
        
    else:
        # Default smart chat configuration
        config = ModelConfig(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            max_tokens=3072,
            temperature=0.5,
            enable_reasoning=True,
            reasoning_budget_tokens=1500
        )
        agent_name = "Smart Chat Agent"
        system_prompt = "You are a helpful AI assistant. Use reasoning when needed and be conversational."
        show_reasoning = False
    
    # Ask about reasoning display
    if config.enable_reasoning:
        show_choice = input("\nShow reasoning process? (y/N): ").strip().lower()
        show_reasoning = show_choice == 'y'
    
    try:
        # Create the agent
        agent = Agent(
            name=agent_name,
            system=system_prompt,
            config=config,
            verbose=True,
            show_reasoning=show_reasoning
        )
        
        print(f"\n✅ {agent_name} created successfully!")
        print(f"🧠 Reasoning: {'Enabled' if config.enable_reasoning else 'Disabled'}")
        print(f"👁️  Show reasoning: {'Yes' if show_reasoning else 'No'}")
        
        # Start interactive chat
        agent.start_interactive_chat()
        
    except Exception as e:
        print(f"\n❌ Error creating agent: {e}")
        print("Make sure you have:")
        print("  - AWS credentials configured")
        print("  - Access to Bedrock Claude models")
        print("  - boto3 installed (pip install boto3)")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 