#!/usr/bin/env python3
"""Test script for AWS Bedrock agent with Invoke API for extended thinking capabilities."""

import os
import sys
from agents.agent_invoke import AgentInvoke, ModelConfig


def test_invoke_reasoning_capabilities():
    """Test the reasoning capabilities of Claude 3.7 on Bedrock using Invoke API."""
    
    print("🧠 Testing AWS Bedrock Agent with Invoke API for Extended Thinking")
    print("=" * 70)
    
    # Create agent with reasoning enabled using Invoke API
    agent = AgentInvoke(
        name="Invoke Reasoning Test Agent",
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
    
    print("✅ Agent with Invoke API reasoning capabilities created successfully!\n")
    
    # Test 1: Mathematical reasoning
    print("🔢 TEST 1: Complex Mathematical Problem with Extended Thinking")
    print("-" * 55)
    
    math_problem = """
    A company has 3 locations: A, B, and C. 
    - Location A produces 150 units/day and has 25 employees
    - Location B produces 200 units/day and has 32 employees  
    - Location C produces 180 units/day and has 28 employees
    
    If they want to increase total production by 25% while maintaining the same 
    efficiency ratio (units per employee) across all locations, how many new 
    employees should they hire at each location?
    """
    
    response = agent.run(math_problem)
    print(f"\n✓ Complex mathematical reasoning test completed.\n")
    
    # Test 2: Logical reasoning with multiple steps
    print("🤔 TEST 2: Multi-Step Logical Reasoning Problem")
    print("-" * 50)
    
    logic_problem = """
    There are 5 people: Alice, Bob, Carol, David, and Eve.
    - Alice is taller than Bob but shorter than Carol
    - David is the tallest person
    - Eve is shorter than Bob
    - Carol is shorter than David
    
    If we need to arrange them in a line from shortest to tallest for a photo,
    what should be the order? Explain your reasoning step by step.
    """
    
    response = agent.run(logic_problem)
    print(f"\n✓ Multi-step logical reasoning test completed.\n")
    
    # Test 3: Scientific reasoning
    print("🔬 TEST 3: Scientific Problem Solving")
    print("-" * 40)
    
    science_problem = """
    A ball is dropped from a height of 100 meters. Each time it bounces, 
    it reaches 80% of its previous height. 
    
    1. How high will it be after the 3rd bounce?
    2. What's the total distance traveled by the ball after 5 complete bounces?
    3. If the ball needs to reach at least 10 meters high to be visible, 
       after how many bounces will it become invisible?
    
    Show detailed calculations and reasoning for each part.
    """
    
    response = agent.run(science_problem)
    print(f"\n✓ Scientific reasoning test completed.\n")
    
    # Test 4: Programmatic access to reasoning
    print("💻 TEST 4: Programmatic Access to Extended Thinking")
    print("-" * 55)
    
    # Create a quiet agent for programmatic testing
    quiet_agent = AgentInvoke(
        name="Quiet Invoke Test Agent",
        system="Think step-by-step through problems with detailed reasoning.",
        config=ModelConfig(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            enable_reasoning=True,
            reasoning_budget_tokens=2500  # Higher budget for complex reasoning
        ),
        verbose=False
    )
    
    reasoning, answer = quiet_agent.get_reasoning_and_response(
        "If I invest $1000 at 8% compound interest annually, and I add $200 every year, how much will I have after 10 years?"
    )
    
    print("Extended Reasoning Process:")
    if reasoning:
        print(f"Length: {len(reasoning)} characters")
        print(f"Preview: {reasoning[:200]}..." if len(reasoning) > 200 else reasoning)
    else:
        print("No reasoning content available")
        
    print(f"\nFinal Answer: {answer}")
    
    # Test individual methods
    reasoning_only = quiet_agent.get_reasoning_only("What is the derivative of x^3 + 2x^2 - 5x + 1?")
    response_only = quiet_agent.get_response_only("What is the derivative of x^3 + 2x^2 - 5x + 1?")
    
    print(f"\nReasoning only (preview): {reasoning_only[:150] if reasoning_only else 'None'}...")
    print(f"Response only: {response_only}")
    
    print(f"\n✓ Programmatic access test completed.\n")
    
    # Test 5: Reasoning with different budget levels
    print("⚙️  TEST 5: Reasoning Budget Comparison")
    print("-" * 45)
    
    # High reasoning budget
    high_budget_agent = AgentInvoke(
        name="High Budget Agent",
        system="Use extensive detailed reasoning for problems.",
        config=ModelConfig(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            enable_reasoning=True,
            reasoning_budget_tokens=4000  # Very high budget
        ),
        verbose=False
    )
    
    # Low reasoning budget  
    low_budget_agent = AgentInvoke(
        name="Low Budget Agent", 
        system="Be concise but thorough in reasoning.",
        config=ModelConfig(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            enable_reasoning=True,
            reasoning_budget_tokens=500  # Lower budget
        ),
        verbose=False
    )
    
    complex_problem = """
    A chess tournament has 16 players. In the first round, they play in pairs (8 matches).
    Winners advance to the next round. This continues until there's a champion.
    
    If each match takes 45 minutes on average, and there's a 15-minute break between rounds,
    how long will the entire tournament take? Consider that matches in each round happen simultaneously.
    """
    
    high_reasoning, high_response = high_budget_agent.get_reasoning_and_response(complex_problem)
    low_reasoning, low_response = low_budget_agent.get_reasoning_and_response(complex_problem)
    
    print(f"High budget reasoning length: {len(high_reasoning) if high_reasoning else 0} characters")
    print(f"Low budget reasoning length: {len(low_reasoning) if low_reasoning else 0} characters")
    print(f"High budget response length: {len(high_response)} characters")
    print(f"Low budget response length: {len(low_response)} characters")
    
    if high_reasoning and low_reasoning:
        print(f"\nReasoning quality comparison:")
        print(f"High budget reasoning depth: {'Detailed' if len(high_reasoning) > 1000 else 'Moderate' if len(high_reasoning) > 500 else 'Brief'}")
        print(f"Low budget reasoning depth: {'Detailed' if len(low_reasoning) > 1000 else 'Moderate' if len(low_reasoning) > 500 else 'Brief'}")
    
    print(f"\n✓ Budget comparison test completed.\n")
    
    # Test 6: Document analysis with reasoning
    print("📄 TEST 6: Document Analysis with Extended Thinking")
    print("-" * 55)
    
    # Create a complex test document
    test_data = {
        "financial_data": {
            "Q1_2023": {"revenue": 1500000, "expenses": 1200000, "marketing": 150000, "r_and_d": 200000},
            "Q2_2023": {"revenue": 1800000, "expenses": 1300000, "marketing": 180000, "r_and_d": 220000},
            "Q3_2023": {"revenue": 1650000, "expenses": 1400000, "marketing": 160000, "r_and_d": 250000},
            "Q4_2023": {"revenue": 2200000, "expenses": 1500000, "marketing": 200000, "r_and_d": 280000}
        },
        "market_data": {
            "competitors": [
                {"name": "CompanyA", "market_share": 25, "growth_rate": 12},
                {"name": "CompanyB", "market_share": 18, "growth_rate": 8},
                {"name": "CompanyC", "market_share": 15, "growth_rate": 15}
            ],
            "industry_trends": ["AI adoption", "Remote work tools", "Sustainability focus"]
        }
    }
    
    import json
    test_doc = "complex_financial_data.json"
    with open(test_doc, "w") as f:
        json.dump(test_data, f, indent=2)
    
    doc_agent = AgentInvoke(
        name="Document Analysis Agent",
        system="Analyze documents with comprehensive step-by-step reasoning.",
        config=ModelConfig(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            enable_reasoning=True,
            reasoning_budget_tokens=2500
        ),
        verbose=True,
        show_reasoning=True
    )
    
    response = doc_agent.run(
        """Analyze this financial data comprehensively:
        1. Calculate profit margins for each quarter
        2. Identify trends in spending patterns  
        3. Compare our implied market position vs competitors
        4. Provide strategic recommendations based on the data
        
        Show detailed reasoning for all calculations and conclusions.""",
        document_paths=[test_doc]
    )
    
    # Clean up
    if os.path.exists(test_doc):
        os.remove(test_doc)
    
    print(f"\n✓ Document analysis with reasoning test completed.\n")
    
    print("🎉 All Invoke API reasoning tests completed successfully!")
    print("=" * 70)
    

def compare_invoke_vs_converse():
    """Compare reasoning capabilities between Invoke and Converse APIs."""
    
    print("\n🔄 COMPARISON: Invoke API vs Converse API Reasoning")
    print("-" * 60)
    
    from agents.agent import Agent as ConverseAgent
    
    # Same problem for both APIs
    test_problem = """
    A factory produces widgets in batches. Each batch takes 3 hours to complete
    and produces 150 widgets. The factory operates 16 hours per day.
    
    If demand is 2000 widgets per day, and the factory currently has 500 widgets
    in inventory, how many days can they meet demand without producing new batches?
    
    After that, if they resume production at full capacity, how many days will it
    take to build up a 1-week safety stock (assuming demand continues at 2000/day)?
    """
    
    print("Testing with Invoke API...")
    invoke_agent = AgentInvoke(
        name="Invoke Test",
        system="Use detailed step-by-step reasoning.",
        config=ModelConfig(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            enable_reasoning=True,
            reasoning_budget_tokens=2000
        ),
        verbose=False
    )
    
    invoke_reasoning, invoke_response = invoke_agent.get_reasoning_and_response(test_problem)
    
    print("Testing with Converse API...")
    converse_agent = ConverseAgent(
        name="Converse Test",
        system="Use detailed step-by-step reasoning.",
        config=ModelConfig(
            model="anthropic.claude-3-7-sonnet-20250219-v1:0",
            enable_reasoning=True,
            reasoning_budget_tokens=2000
        ),
        verbose=False
    )
    
    converse_reasoning, converse_response = converse_agent.get_reasoning_and_response(test_problem)
    
    print("\n📊 Comparison Results:")
    print(f"Invoke API reasoning length: {len(invoke_reasoning) if invoke_reasoning else 0} characters")
    print(f"Converse API reasoning length: {len(converse_reasoning) if converse_reasoning else 0} characters")
    print(f"Invoke API response length: {len(invoke_response)} characters")
    print(f"Converse API response length: {len(converse_response)} characters")
    
    print(f"\nInvoke API reasoning available: {'✅ Yes' if invoke_reasoning else '❌ No'}")
    print(f"Converse API reasoning available: {'✅ Yes' if converse_reasoning else '❌ No'}")
    
    if invoke_reasoning and converse_reasoning:
        print("\n🔍 Both APIs provided reasoning content")
    elif invoke_reasoning and not converse_reasoning:
        print("\n🎯 Only Invoke API provided reasoning content")
    elif not invoke_reasoning and converse_reasoning:
        print("\n🎯 Only Converse API provided reasoning content")
    else:
        print("\n⚠️ Neither API provided reasoning content")


if __name__ == "__main__":
    try:
        # Check if running in an environment where we can test
        if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.path.exists(os.path.expanduser("~/.aws/credentials")):
            print("⚠️  Warning: AWS credentials not found. Make sure to configure AWS credentials before running tests.")
            print("   You can use: aws configure")
            print("   Or set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            sys.exit(1)
        
        # Run Invoke API reasoning tests
        test_invoke_reasoning_capabilities()
        
        # Run comparison test
        compare_invoke_vs_converse()
        
        print(f"\n✨ All tests completed! The Invoke API implementation provides enhanced reasoning capabilities.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("   Make sure you have:")
        print("   1. AWS credentials configured")  
        print("   2. Access to Bedrock Claude models")
        print("   3. boto3 installed")
        sys.exit(1) 