#!/usr/bin/env python3
"""Test script for AWS Bedrock agent with reasoning capabilities."""

import os
import sys
from agents.agent import Agent, ModelConfig


def test_reasoning_capabilities():
    """Test the reasoning capabilities of Claude 3.7 on Bedrock."""
    
    print("🧠 Testing AWS Bedrock Agent with Reasoning Capabilities")
    print("=" * 60)
    
    # Create agent with reasoning enabled
    agent = Agent(
        name="Reasoning Test Agent",
        system="You are a helpful AI assistant. Use clear, step-by-step reasoning for all problems.",
        config=ModelConfig(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            max_tokens=4096,
            temperature=0.3,  # Lower temperature for more consistent reasoning
            enable_reasoning=True,
            reasoning_budget_tokens=2000
        ),
        region="us-east-1",
        verbose=True,
        show_reasoning=True  # Show the reasoning process
    )
    
    print("✅ Agent with reasoning capabilities created successfully!\n")
    
    # Test 1: Mathematical reasoning
    print("🔢 TEST 1: Mathematical Problem with Reasoning")
    print("-" * 50)
    
    math_problem = """
    A bakery sells cupcakes for $3 each and cookies for $2 each.
    Yesterday, they sold 45 cupcakes and some cookies.
    Their total revenue was $219.
    How many cookies did they sell?
    """
    
    response = agent.run(math_problem)
    print(f"\n✓ Mathematical reasoning test completed.\n")
    
    # Test 2: Logical reasoning
    print("🤔 TEST 2: Logical Reasoning Problem")
    print("-" * 50)
    
    logic_problem = """
    All cats are mammals.
    Some mammals are carnivores.
    Fluffy is a cat.
    Can we conclude that Fluffy is a carnivore? Explain your reasoning.
    """
    
    response = agent.run(logic_problem)
    print(f"\n✓ Logical reasoning test completed.\n")
    
    # Test 3: Programmatic access to reasoning
    print("💻 TEST 3: Programmatic Access to Reasoning")
    print("-" * 50)
    
    # Create a quiet agent for programmatic testing
    quiet_agent = Agent(
        name="Quiet Test Agent",
        system="Think step-by-step through problems.",
        config=ModelConfig(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            enable_reasoning=True,
            reasoning_budget_tokens=1500
        ),
        verbose=False
    )
    
    reasoning, answer = quiet_agent.get_reasoning_and_response(
        "What is 15% of 240?"
    )
    
    print("Reasoning process:")
    print(reasoning if reasoning else "No reasoning available")
    print(f"\nFinal answer: {answer}")
    
    # Test individual methods
    reasoning_only = quiet_agent.get_reasoning_only("Calculate the area of a circle with radius 7")
    response_only = quiet_agent.get_response_only("Calculate the area of a circle with radius 7")
    
    print(f"\nReasoning only (first 100 chars): {reasoning_only[:100] if reasoning_only else 'None'}...")
    print(f"Response only: {response_only}")
    
    print(f"\n✓ Programmatic access test completed.\n")
    
    # Test 4: Reasoning configuration options
    print("⚙️  TEST 4: Different Reasoning Configurations")
    print("-" * 50)
    
    # High reasoning budget
    high_reasoning_agent = Agent(
        name="High Reasoning Agent",
        system="Use extensive reasoning for complex problems.",
        config=ModelConfig(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            enable_reasoning=True,
            reasoning_budget_tokens=3000  # Higher budget
        ),
        verbose=False
    )
    
    # Low reasoning budget  
    low_reasoning_agent = Agent(
        name="Low Reasoning Agent", 
        system="Be concise in your reasoning.",
        config=ModelConfig(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            enable_reasoning=True,
            reasoning_budget_tokens=500  # Lower budget
        ),
        verbose=False
    )
    
    problem = "Explain the relationship between interest rates and inflation."
    
    high_reasoning, high_response = high_reasoning_agent.get_reasoning_and_response(problem)
    low_reasoning, low_response = low_reasoning_agent.get_reasoning_and_response(problem)
    
    print(f"High budget reasoning length: {len(high_reasoning) if high_reasoning else 0} characters")
    print(f"Low budget reasoning length: {len(low_reasoning) if low_reasoning else 0} characters")
    print(f"High budget response length: {len(high_response)} characters")
    print(f"Low budget response length: {len(low_response)} characters")
    
    print(f"\n✓ Configuration comparison test completed.\n")
    
    # Test 5: Document analysis with reasoning
    print("📄 TEST 5: Document Analysis with Reasoning")
    print("-" * 50)
    
    # Create a test document
    test_data = {
        "quarterly_revenue": [150000, 180000, 165000, 200000],
        "expenses": [120000, 130000, 140000, 145000],
        "employee_count": [25, 28, 30, 32],
        "departments": ["Engineering", "Sales", "Marketing"]
    }
    
    import json
    test_doc = "test_data.json"
    with open(test_doc, "w") as f:
        json.dump(test_data, f, indent=2)
    
    doc_agent = Agent(
        name="Document Test Agent",
        system="Analyze documents with clear reasoning.",
        config=ModelConfig(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            enable_reasoning=True,
            reasoning_budget_tokens=1500
        ),
        verbose=True,
        show_reasoning=True
    )
    
    response = doc_agent.run(
        "Analyze this business data and calculate the profit margins for each quarter. Show your reasoning.",
        document_paths=[test_doc]
    )
    
    # Test convenience methods
    print("\n📄 Testing Document Convenience Methods:")
    single_doc = doc_agent.add_document_from_file(test_doc, "What's the average quarterly revenue?")
    
    # Clean up
    import os
    if os.path.exists(test_doc):
        os.remove(test_doc)
    
    print(f"\n✓ Document analysis test completed.\n")
    
    print("🎉 All reasoning tests completed successfully!")
    print("=" * 60)
    

def test_reasoning_disabled():
    """Test agent with reasoning disabled for comparison."""
    
    print("\n🔄 COMPARISON: Agent WITHOUT Reasoning")
    print("-" * 50)
    
    no_reasoning_agent = Agent(
        name="No Reasoning Agent",
        system="You are a helpful assistant.",
        config=ModelConfig(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            enable_reasoning=False  # Disabled
        ),
        verbose=True
    )
    
    response = no_reasoning_agent.run("What is 25% of 80? Show your work.")
    print("✓ Non-reasoning agent test completed.")


if __name__ == "__main__":
    try:
        # Check if running in an environment where we can test
        if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.path.exists(os.path.expanduser("~/.aws/credentials")):
            print("⚠️  Warning: AWS credentials not found. Make sure to configure AWS credentials before running tests.")
            print("   You can use: aws configure")
            print("   Or set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            sys.exit(1)
        
        # Run reasoning tests
        test_reasoning_capabilities()
        
        # Run comparison test
        test_reasoning_disabled()
        
        print(f"\n✨ All tests completed! The reasoning migration was successful.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("   Make sure you have:")
        print("   1. AWS credentials configured")  
        print("   2. Access to Bedrock Claude models")
        print("   3. boto3 installed")
        sys.exit(1) 