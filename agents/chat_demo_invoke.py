#!/usr/bin/env python3
"""Interactive chat demo for AWS Bedrock agent using Invoke API with extended thinking."""

import sys
from agents.agent_invoke import AgentInvoke, ModelConfig


def main():
    """Run an interactive chat demo with extended thinking capabilities."""
    
    print("🤖 AWS Bedrock Agent Chat Demo - Invoke API with Extended Thinking")
    print("=" * 70)
    print("This demo uses the Invoke API for enhanced reasoning capabilities!")
    print()
    
    # Create agent with reasoning enabled
    agent = AgentInvoke(
        name="Claude Assistant",
        system="""You are Claude, a helpful AI assistant created by Anthropic. 
        You excel at complex reasoning and problem-solving. When faced with challenging 
        questions, think through them step-by-step using detailed reasoning.""",
        config=ModelConfig(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            max_tokens=4096,
            temperature=0.7,
            enable_reasoning=True,
            reasoning_budget_tokens=2500  # Higher budget for complex thinking
        ),
        region="us-east-1",
        verbose=True,
        show_reasoning=True  # Show reasoning process in chat
    )
    
    print("✅ Chat agent with extended thinking capabilities initialized!")
    print()
    print("💡 Try asking complex questions that require step-by-step reasoning:")
    print("   - Math problems with multiple steps")
    print("   - Logic puzzles")  
    print("   - Scientific calculations")
    print("   - Business analysis questions")
    print("   - Philosophical thought experiments")
    print()
    print("🔧 Special commands:")
    print("   - '/reasoning on/off' - Toggle reasoning display")
    print("   - '/budget <number>' - Change reasoning token budget")
    print("   - '/simple' - Switch to simple mode (no reasoning)")
    print("   - '/thinking' - Switch to thinking mode (high reasoning budget)")
    print()
    
    # Start interactive chat
    agent.start_interactive_chat()


def handle_special_commands(agent, user_input):
    """Handle special demo commands."""
    
    if user_input.startswith('/reasoning '):
        mode = user_input[11:].strip().lower()
        if mode == 'on':
            agent.show_reasoning = True
            print("✅ Reasoning display enabled")
        elif mode == 'off':
            agent.show_reasoning = False
            print("✅ Reasoning display disabled")
        else:
            print("❓ Use '/reasoning on' or '/reasoning off'")
        return True
        
    elif user_input.startswith('/budget '):
        try:
            new_budget = int(user_input[8:].strip())
            agent.config.reasoning_budget_tokens = new_budget
            print(f"✅ Reasoning budget set to {new_budget} tokens")
        except ValueError:
            print("❓ Use '/budget <number>' (e.g., '/budget 3000')")
        return True
        
    elif user_input == '/simple':
        agent.config.enable_reasoning = False
        agent.show_reasoning = False
        print("✅ Switched to simple mode (reasoning disabled)")
        return True
        
    elif user_input == '/thinking':
        agent.config.enable_reasoning = True
        agent.config.reasoning_budget_tokens = 4000
        agent.show_reasoning = True
        print("✅ Switched to thinking mode (high reasoning budget)")
        return True
        
    return False


def demo_examples():
    """Show some example prompts that work well with extended thinking."""
    
    print("\n🎯 Example Prompts That Showcase Extended Thinking:")
    print("-" * 55)
    
    examples = [
        "Calculate the compound interest on $5,000 invested at 6% annually for 8 years, compounded quarterly.",
        
        "If I have a 20% chance of success on each attempt, and I make 10 attempts, what's the probability I succeed at least once?",
        
        "A train leaves City A at 2 PM traveling 60 mph. Another train leaves City B at 3 PM traveling 80 mph toward City A. If the cities are 300 miles apart, when and where will they meet?",
        
        "Explain the trolley problem and analyze the ethical implications from three different philosophical perspectives.",
        
        "A company's revenue grew from $1M to $4M over 4 years. If growth was exponential, what was the annual growth rate?",
        
        "Design a simple algorithm to determine if a string is a palindrome, considering only alphanumeric characters and ignoring case.",
        
        "If Earth's population is 8 billion and each person uses 50 gallons of water per day, how many Olympic swimming pools worth of water is used globally each day?"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}\n")
    
    print("💭 These prompts will trigger detailed step-by-step reasoning!")
    print("   Try copying and pasting one into the chat.\n")


if __name__ == "__main__":
    try:
        # Check AWS credentials
        import os
        if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.path.exists(os.path.expanduser("~/.aws/credentials")):
            print("⚠️  Warning: AWS credentials not found.")
            print("   Configure AWS credentials before running: aws configure")
            print("   Or set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            
            response = input("\nContinue anyway? (y/n): ").strip().lower()
            if response != 'y':
                sys.exit(1)
        
        # Option to see examples first
        show_examples = input("Would you like to see example prompts first? (y/n): ").strip().lower()
        if show_examples == 'y':
            demo_examples()
            input("Press Enter to continue to chat...")
        
        # Start main chat
        main()
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo ended. Thanks for trying the Invoke API!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have AWS credentials configured and access to Bedrock.")
        sys.exit(1) 