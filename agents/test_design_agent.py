import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.design_agent import DesignAgent

def test_design_agent():
    """Test the design agent with a sample query"""
    print("Initializing Design Agent...")
    design_agent = DesignAgent()
    
    # Sample user query in French (as the agent is designed for French)
    query = "Je voudrais une maison moderne avec beaucoup de lumière naturelle, des espaces ouverts, et une connexion avec la nature. Je préfère un design minimaliste mais avec quelques éléments chaleureux."
    
    print("\nProcessing design message...")
    results = design_agent.process_message(query)
    
    print("\n--- Design Agent Results ---")
    print(f"Stage: {results['stage']}")
    print(f"Response: {results['text']}")
    
    # Show extracted info if available
    if results.get('extracted_info'):
        print(f"\nExtracted Info:")
        for key, value in results['extracted_info'].items():
            print(f"- {key}: {value}")
    
    # Test a follow-up message for technical options
    print("\n--- Testing Technical Options ---")
    followup = "Pour le sol, je préfère le marbre et j'aimerais une porte blindée pour la sécurité."
    
    followup_results = design_agent.process_message(followup)
    print(f"Stage: {followup_results['stage']}")
    print(f"Response: {followup_results['text']}")
    
    if followup_results.get('extracted_info'):
        print(f"\nExtracted Info:")
        for key, value in followup_results['extracted_info'].items():
            print(f"- {key}: {value}")
    
    # Generate final design brief
    print("\n--- Final Design Brief ---")
    brief = design_agent.generate_design_brief()
    print(f"Design Brief: {brief}")
    
    return results

if __name__ == "__main__":
    test_design_agent()
