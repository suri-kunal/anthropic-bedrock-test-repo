#!/usr/bin/env python3
"""Simple comparison script to demonstrate differences between Converse and Invoke APIs."""

import os
import sys


def compare_reasoning_apis():
    """Compare reasoning capabilities between Converse and Invoke APIs."""
    
    print("🔍 Comparing Converse API vs Invoke API for Extended Thinking")
    print("=" * 65)
    
    # Test problem that requires step-by-step reasoning
    test_problem = """
    Sarah has $1000 to invest. She can choose between:
    - Option A: 5% simple interest annually
    - Option B: 4.8% compound interest annually
    
    After 10 years, which option gives her more money and by how much?
    Show your step-by-step calculations.
    """
    
    print("Test Problem:")
    print(test_problem)
    print("\n" + "=" * 65)
    
    # Test with Converse API
    print("\n🔄 Testing with Converse API:")
    print("-" * 40)
    
    try:
        from agents.agent import Agent, ModelConfig
        
        converse_agent = Agent(
            name="Converse Agent",
            system="You are a helpful assistant. Show detailed step-by-step reasoning.",
            config=ModelConfig(
                model="anthropic.claude-3-7-sonnet-20250219-v1:0",
                enable_reasoning=True,
                reasoning_budget_tokens=2000
            ),
            verbose=False
        )
        
        print("✅ Converse agent created successfully")
        converse_reasoning, converse_response = converse_agent.get_reasoning_and_response(test_problem)
        
        print(f"📊 Converse API Results:")
        print(f"   Reasoning available: {'✅ Yes' if converse_reasoning else '❌ No'}")
        if converse_reasoning:
            print(f"   Reasoning length: {len(converse_reasoning)} characters")
            print(f"   Reasoning preview: {converse_reasoning[:150]}...")
        print(f"   Response length: {len(converse_response)} characters")
        print(f"   Response preview: {converse_response[:200]}...")
        
    except Exception as e:
        print(f"❌ Converse API test failed: {e}")
        converse_reasoning = None
        converse_response = "Error occurred"
    
    # Test with Invoke API
    print(f"\n🚀 Testing with Invoke API:")
    print("-" * 40)
    
    try:
        from agents.agent_invoke import AgentInvoke
        
        invoke_agent = AgentInvoke(
            name="Invoke Agent", 
            system="You are a helpful assistant. Show detailed step-by-step reasoning.",
            config=ModelConfig(
                model="anthropic.claude-3-7-sonnet-20250219-v1:0",
                enable_reasoning=True,
                reasoning_budget_tokens=2000
            ),
            verbose=False
        )
        
        print("✅ Invoke agent created successfully")
        invoke_reasoning, invoke_response = invoke_agent.get_reasoning_and_response(test_problem)
        
        print(f"📊 Invoke API Results:")
        print(f"   Reasoning available: {'✅ Yes' if invoke_reasoning else '❌ No'}")
        if invoke_reasoning:
            print(f"   Reasoning length: {len(invoke_reasoning)} characters")
            print(f"   Reasoning preview: {invoke_reasoning[:150]}...")
        print(f"   Response length: {len(invoke_response)} characters")
        print(f"   Response preview: {invoke_response[:200]}...")
        
    except Exception as e:
        print(f"❌ Invoke API test failed: {e}")
        invoke_reasoning = None
        invoke_response = "Error occurred"
    
    # Comparison summary
    print(f"\n📋 Comparison Summary:")
    print("-" * 30)
    
    if converse_reasoning and invoke_reasoning:
        print("🔍 Both APIs provided reasoning content")
        print(f"   Converse reasoning: {len(converse_reasoning)} chars")
        print(f"   Invoke reasoning: {len(invoke_reasoning)} chars")
        
        if len(invoke_reasoning) > len(converse_reasoning):
            print("   🎯 Invoke API provided more detailed reasoning")
        elif len(converse_reasoning) > len(invoke_reasoning):
            print("   🎯 Converse API provided more detailed reasoning")
        else:
            print("   🤝 Both APIs provided similar reasoning detail")
            
    elif invoke_reasoning and not converse_reasoning:
        print("🎯 Only Invoke API provided reasoning content")
        
    elif converse_reasoning and not invoke_reasoning:
        print("🎯 Only Converse API provided reasoning content")
        
    else:
        print("⚠️ Neither API provided reasoning content")
    
    print(f"\n💡 Recommendation:")
    if invoke_reasoning:
        print("   ✅ Invoke API is working well for extended thinking")
        print("   🚀 Consider migrating to Invoke API for better reasoning control")
    else:
        print("   ⚠️ Check your reasoning configuration and model access")
    
    return invoke_reasoning, converse_reasoning


def interactive_demo():
    """Run an interactive demo to test both APIs."""
    
    print("\n🎮 Interactive Demo Mode")
    print("=" * 30)
    
    while True:
        print("\nOptions:")
        print("1. Test with a custom problem")
        print("2. Try a math problem")
        print("3. Try a logic puzzle")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == "1":
            problem = input("\nEnter your problem: ").strip()
            if problem:
                test_both_apis(problem)
        
        elif choice == "2":
            problem = "If I invest $5000 at 7% annual compound interest for 6 years, how much will I have? Show calculations."
            print(f"\nTesting math problem: {problem}")
            test_both_apis(problem)
        
        elif choice == "3":
            problem = "There are 5 houses in a row. The red house is to the left of the blue house. The green house is to the right of the red house but to the left of the blue house. The yellow house is at one end. Where is each house?"
            print(f"\nTesting logic puzzle: {problem}")
            test_both_apis(problem)
        
        elif choice == "4":
            print("👋 Thanks for trying the comparison!")
            break
        
        else:
            print("❓ Invalid choice. Please select 1-4.")


def test_both_apis(problem):
    """Test both APIs with a given problem."""
    
    print(f"\n🔄 Testing both APIs with your problem...")
    
    # Quick test with both APIs
    try:
        from agents.agent import Agent, ModelConfig
        from agents.agent_invoke import AgentInvoke
        
        # Test Converse API
        converse_agent = Agent(
            name="Quick Test",
            system="Think step by step.",
            config=ModelConfig(enable_reasoning=True, reasoning_budget_tokens=1500),
            verbose=False
        )
        
        # Test Invoke API
        invoke_agent = AgentInvoke(
            name="Quick Test",
            system="Think step by step.",
            config=ModelConfig(enable_reasoning=True, reasoning_budget_tokens=1500),
            verbose=False
        )
        
        print("Testing Converse API...")
        converse_reasoning, converse_response = converse_agent.get_reasoning_and_response(problem)
        
        print("Testing Invoke API...")
        invoke_reasoning, invoke_response = invoke_agent.get_reasoning_and_response(problem)
        
        print(f"\n📊 Quick Results:")
        print(f"Converse - Reasoning: {'✅' if converse_reasoning else '❌'}, Response: {len(converse_response)} chars")
        print(f"Invoke   - Reasoning: {'✅' if invoke_reasoning else '❌'}, Response: {len(invoke_response)} chars")
        
        if invoke_reasoning:
            print(f"\n🧠 Invoke API Reasoning Preview:")
            print(f"{invoke_reasoning[:300]}...")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")


if __name__ == "__main__":
    try:
        # Check AWS credentials
        if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.path.exists(os.path.expanduser("~/.aws/credentials")):
            print("⚠️  Warning: AWS credentials not found.")
            print("   Configure with: aws configure")
            print("   Or set: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            
            response = input("\nContinue anyway? (y/n): ").strip().lower()
            if response != 'y':
                sys.exit(1)
        
        # Run main comparison
        invoke_reasoning, converse_reasoning = compare_reasoning_apis()
        
        # Offer interactive demo
        if invoke_reasoning or converse_reasoning:
            demo_choice = input("\nWould you like to try the interactive demo? (y/n): ").strip().lower()
            if demo_choice == 'y':
                interactive_demo()
        
    except KeyboardInterrupt:
        print("\n\n👋 Comparison ended. Thanks!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have AWS credentials and Bedrock access.")
        sys.exit(1) 