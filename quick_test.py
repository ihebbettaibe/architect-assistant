"""
Quick verification script for LangChain Budget Agent
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def quick_test():
    """Quick test of the LangChain agent"""
    
    print("🔧 Quick LangChain Agent Test")
    print("=" * 40)
    
    # Check API key
    groq_api_key = os.getenv('GROQ_API_KEY')
    langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
    
    print(f"GROQ_API_KEY: {'✅' if groq_api_key else '❌'}")
    print(f"LANGSMITH_API_KEY: {'✅' if langsmith_api_key else '❌'}")
    
    if not groq_api_key:
        print("❌ GROQ_API_KEY required")
        return False
    
    try:
        # Test import
        print("\n🔄 Testing imports...")
        from agents.budget.langchain_budget_agent import LangChainBudgetAgent
        print("✅ Import successful")
        
        # Test agent creation
        print("\n🔄 Creating agent...")
        agent = LangChainBudgetAgent(
            groq_api_key=groq_api_key,
            model_name="mixtral-8x7b-32768",
            data_folder="cleaned_data",
            use_couchdb=False
        )
        print("✅ Agent created successfully")
        
        # Test simple chat
        print("\n🔄 Testing chat...")
        result = agent.chat("Je cherche une propriété avec un budget de 200000 DT")
        
        if result and result.get('response'):
            print("✅ Chat successful")
            print(f"📝 Response length: {len(result['response'])} chars")
            print(f"📊 Context items: {len(result.get('context', {}))}")
            
            # Show first 100 chars
            preview = result['response'][:100]
            print(f"📄 Preview: {preview}...")
            
            return True
        else:
            print("❌ No response received")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n🎉 LangChain agent is working!")
        print("🚀 Start Streamlit app: streamlit run streamlit_langchain_app.py --server.port 8502")
        
        if not os.getenv('LANGSMITH_API_KEY'):
            print("\n💡 To enable LangSmith tracing:")
            print("1. Sign up at https://smith.langchain.com/")
            print("2. Get API key and add to .env file")
            print("3. Restart the app")
    else:
        print("\n💥 Test failed - check errors above")
