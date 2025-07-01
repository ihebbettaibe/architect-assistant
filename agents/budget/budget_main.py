from .budget_agent_base import EnhancedBudgetAgent
from .budget_analysis import BudgetAnalysis
from .client_interface import ClientInterface

class FullBudgetAgent(EnhancedBudgetAgent, BudgetAnalysis, ClientInterface):
    pass

if __name__ == "__main__":
    print("🚀 Initializing Enhanced Budget Agent...")
    # Use the correct path to cleaned_data folder (go up two directories from agents/budget/)
    agent = FullBudgetAgent(data_folder="cleaned_data")
    
    # Test the multi-agent interface
    print("\n🤖 Testing Multi-Agent Interface...")
    
    # Simulate client inputs
    test_inputs = [
        "Je souhaite construire une maison avec un budget de 350000 DT",
        "Mon budget est flexible, je peux aller jusqu'à 400000 DT",
        "J'ai besoin d'un crédit pour financer le projet"
    ]
    
    for i, client_input in enumerate(test_inputs, 1):
        print(f"\n📝 Client Input {i}: {client_input}")
        
        # Process with budget agent
        result = agent.process_client_input(client_input)
        
        print(f"🎯 Agent Response:")
        print(f"  - Budget Extracted: {result['budget_analysis']['extracted_budget']}")
        print(f"  - Reliability Score: {result['reliability_score']:.1%}")
        print(f"  - Confidence Level: {result['confidence_level']}")
        
        if result['inconsistencies_detected']:
            print(f"  - Inconsistencies: {len(result['inconsistencies_detected'])}")
        
        if result['targeted_questions']:
            print(f"  - Questions: {result['targeted_questions'][0]}")
        
        print(f"  - Next Actions: {result['next_actions']}")
    
    # Test traditional analysis (for backward compatibility)
    print("\n📊 Testing Traditional Analysis...")
    print()  # Empty line for spacing
    client_profile = {
        "city": "Sousse",
        "budget": 60000,
        "preferences": "terrain habitation construction",
        "min_size": 200,
        "max_price": 60000    }
    
    # Traditional budget analysis
    budget_analysis = agent.analyze_client_budget(client_profile)
    
    print(f"\n=== TRADITIONAL BUDGET ANALYSIS ===")
    print(f"Client: {client_profile['city']} - Budget: {client_profile['budget']:,} DT")
    print(f"Properties Analyzed: {budget_analysis.get('total_properties_analyzed', 0)}")
    print(f"Comparable Properties Found: {budget_analysis['market_statistics']['inventory_count']}")
    
    # Display analysis results if available
    if budget_analysis and budget_analysis['market_statistics']['inventory_count'] > 0:
        market_stats = budget_analysis['market_statistics']
        budget_ai = budget_analysis['budget_analysis']
        
        print(f"\n📈 Market Statistics:")
        print(f"- Price Range: {market_stats['price_stats']['min']:,.0f} - {market_stats['price_stats']['max']:,.0f} DT")
        print(f"- Average Price: {market_stats['price_stats']['mean']:,.0f} DT")
        print(f"- Budget Feasibility: {market_stats['budget_feasibility']['feasibility_ratio']:.1%}")
        
        print(f"\n🎯 AI Recommendations:")
        print(f"- Budget Validation: {budget_ai['budget_validation']}")
        if 'market_position' in budget_ai:
            print(f"- Market Position: {budget_ai['market_position']}")
        print(f"- Recommendations: {budget_ai['recommendations']}")
        if 'price_negotiation_tips' in budget_ai:
            print(f"- Price Negotiation Tips: {budget_ai['price_negotiation_tips']}")
        if 'alternative_suggestions' in budget_ai:
            print(f"- Alternative Suggestions: {budget_ai['alternative_suggestions']}")
        if 'market_trends' in budget_ai:
            print(f"- Market Trends: {budget_ai['market_trends']}")
        if 'risk_assessment' in budget_ai:
            print(f"- Risk Assessment: {budget_ai['risk_assessment']}")
        
        print(f"\n📊 Confidence Score: {budget_ai['confidence_score']:.1%}")
        
        # Display the most compatible property with URL
        if budget_analysis.get('comparable_properties'):
            most_compatible = agent.get_most_compatible_property(client_profile, budget_analysis['comparable_properties'])
            if most_compatible:
                print(f"\n🏆 MOST COMPATIBLE PROPERTY:")
                prop = most_compatible['property_details']
                print(f"- Title: {prop['Title']}")
                print(f"- Price: {prop['Price']:,.0f} DT")
                print(f"- Surface: {prop['Surface']:.0f} m²")
                print(f"- Price per m²: {prop['price_per_m2']:,.0f} DT/m²")
                print(f"- Location: {prop['Location']}")
                print(f"- Type: {prop['Type']}")
                print(f"- URL: {prop['URL']}")
                print(f"- {most_compatible['why_compatible']}")
        
    else:
        print(f"\n🔍 Analysis Summary:")
        print(f"  - Properties Analyzed: {budget_analysis.get('total_properties_analyzed', 0)}")
        print(f"  - Matching Properties: {budget_analysis['market_statistics']['inventory_count']}")
        print(f"  - Budget Validation: {budget_analysis['budget_analysis']['budget_validation']}")
        print(f"  - Recommendation: {budget_analysis['budget_analysis']['recommendations']}")
        print(f"  - Confidence Score: {budget_analysis['budget_analysis']['confidence_score']:.1%}")
    
    print(f"\n📊 Visualizations disabled for now")
    print(f"\n🏁 Enhanced Budget Agent testing completed!")
    print(f"\n💡 The agent now supports both:")
    print(f"   1. Multi-agent architecture (process_client_input method)")
    print(f"   2. Traditional analysis (analyze_client_budget method)")