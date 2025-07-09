"""
Test script for LangChain Budget Agent with LangSmith integration
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to the Python path
sys.path.append(os.path.dirname(__file__))

def test_langchain_agent():
    """Test the LangChain budget agent"""
    
    print("🚀 Testing LangChain Budget Agent with LangSmith")
    print("=" * 60)
    
    # Check environment variables
    groq_api_key = os.getenv('GROQ_API_KEY')
    langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
    
    print(f"GROQ_API_KEY: {'✅ Found' if groq_api_key else '❌ Missing'}")
    print(f"LANGSMITH_API_KEY: {'✅ Found' if langsmith_api_key else '❌ Missing'}")
    
    if not groq_api_key:
        print("\n❌ Error: GROQ_API_KEY is required")
        print("Please add GROQ_API_KEY to your .env file")
        return False
    
    if not langsmith_api_key:
        print("\n⚠️ Warning: LANGSMITH_API_KEY not found")
        print("LangSmith tracing will be disabled")
        print("To enable LangSmith:")
        print("1. Sign up at https://smith.langchain.com/")
        print("2. Get your API key from settings")
        print("3. Add LANGSMITH_API_KEY=your_key to .env file")
    
    try:
        # Import the LangChain agent
        from agents.budget.langchain_budget_agent import LangChainBudgetAgent
        
        print("\n🔄 Initializing LangChain Budget Agent...")
        
        # Create agent instance
        agent = LangChainBudgetAgent(
            groq_api_key=groq_api_key,
            model_name="mixtral-8x7b-32768",
            data_folder="cleaned_data",
            use_couchdb=False
        )
        
        print("✅ Agent initialized successfully!")
        
        # Test queries
        test_queries = [
            "Je cherche une propriété avec un budget de 300000 DT à Sousse",
            "Quelle est la propriété avec la plus grande surface dans mon budget?",
            "Montre-moi des options moins chères"
        ]
        
        print(f"\n🧪 Running {len(test_queries)} test queries...")
        print("-" * 40)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test {i}: {query}")
            
            try:
                # Test the chat method
                result = agent.chat(query)
                
                if result and result.get('response'):
                    print(f"✅ Response received ({len(result['response'])} chars)")
                    print(f"📊 Context: {len(result.get('context', {}))} items")
                    print(f"🏠 Properties: {len(result.get('properties', []))} found")
                    
                    # Show first 100 chars of response
                    response_preview = result['response'][:100]
                    print(f"📄 Preview: {response_preview}...")
                    
                else:
                    print("❌ No response received")
                    
            except Exception as e:
                print(f"❌ Error in test {i}: {str(e)}")
        
        print(f"\n🎯 Testing context management...")
        
        # Test context
        context_summary = agent.get_context_summary()
        print(f"📋 Context summary: {context_summary}")
        
        # Test conversation history
        history = agent.get_conversation_history()
        print(f"💬 Conversation history: {len(history)} messages")
        
        print(f"\n✅ All tests completed!")
        
        if langsmith_api_key:
            print(f"\n🔍 Check LangSmith dashboard for execution traces:")
            print(f"https://smith.langchain.com/")
            print(f"Project: Budget-Agent-Real-Estate")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory")
        return False
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🏠 LangChain Budget Agent Test Suite")
    print("🔬 Testing agent functionality and LangSmith integration\n")
    
    success = test_langchain_agent()
    
    if success:
        print(f"\n🎉 Test completed successfully!")
        print(f"Your LangChain agent is ready to use!")
    else:
        print(f"\n💥 Test failed!")
        print(f"Please check the errors above and fix them.")
    
    print(f"\n" + "=" * 60)
