import streamlit as st
import sys
import os
import json
import pandas as pd
from datetime import datetime

# Add the agents directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

# Import the budget agent
try:
    from agents.budget import FullBudgetAgent
except ImportError:
    st.error("❌ Could not import FullBudgetAgent. Please check the import paths.")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="🏠 Architecture Budget Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid;
    }
    .user-message {
        background-color: #f0f8ff;
        border-color: #4169e1;
        color: #000080;
    }
    .agent-message {
        background-color: #f0fff0;
        border-color: #32cd32;
        color: #006400;
    }
    .metrics-container {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #ddd;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .element-container {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    with st.spinner("🚀 Initializing Budget Agent..."):
        try:
            st.session_state.agent = FullBudgetAgent(data_folder="cleaned_data")
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {str(e)}")
            st.session_state.initialized = False

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# Main title
st.markdown('<h1 class="main-header">🏠 Architecture Budget Assistant</h1>', unsafe_allow_html=True)

# Sidebar for agent status and options
with st.sidebar:
    st.header("🤖 Agent Status")
    
    if st.session_state.get('initialized', False):
        st.success("✅ Budget Agent Ready")
        st.info(f"📊 Properties Loaded: {len(st.session_state.agent.property_metadata):,}")
    else:
        st.error("❌ Agent Not Initialized")
    
    st.header("🔧 Options")
    
    # Mode selection
    mode = st.selectbox(
        "Select Mode:",
        ["💬 Chat Mode", "📊 Traditional Analysis", "📈 Market Report"],
        index=0
    )
    
    if st.button("🔄 Clear History", help="Clear conversation history"):
        st.session_state.conversation_history = []
        st.session_state.analysis_results = {}
        st.rerun()

# Chat Mode
if mode == "💬 Chat Mode":
    st.header("💬 Chat with Budget Agent")
    
    # Display conversation history
    for i, message in enumerate(st.session_state.conversation_history):
        if message['role'] == 'user':
            st.markdown(f'<div class="chat-message user-message"><strong>👤 You:</strong><br>{message["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message agent-message"><strong>🤖 Budget Agent:</strong><br>{message["content"]}</div>', 
                       unsafe_allow_html=True)
    
    # Example messages
    st.subheader("💡 Example Messages")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💰 Budget Question", help="Ask about budget estimation"):
            example_msg = "Je souhaite acheter une maison avec un budget de 300000 DT à Sousse"
            st.session_state.conversation_history.append({"role": "user", "content": example_msg})
            st.rerun()
    
    with col2:
        if st.button("🏠 Property Type", help="Ask about property types"):
            example_msg = "Quel type de propriété puis-je avoir avec 250000 DT?"
            st.session_state.conversation_history.append({"role": "user", "content": example_msg})
            st.rerun()
    
    with col3:
        if st.button("📍 Location Query", help="Ask about locations"):
            example_msg = "Quelles sont les meilleures zones pour investir à Tunis?"
            st.session_state.conversation_history.append({"role": "user", "content": example_msg})
            st.rerun()
    
    # Chat input
    user_input = st.text_area(
        "💬 Your Message:",
        placeholder="Ask me about budgets, property prices, market analysis...",
        height=100,
        key="chat_input"
    )
    
    if st.button("🚀 Send Message", type="primary") and user_input.strip():
        if st.session_state.get('initialized', False):
            # Add user message
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            
            with st.spinner("🤖 Agent is thinking..."):
                try:
                    # Process with budget agent
                    result = st.session_state.agent.process_client_input(user_input)
                    
                    # Format agent response
                    agent_response = f"""
**Budget Analysis Results:**
- Budget Extracted: {result['budget_analysis']['extracted_budget']}
- Reliability Score: {result['reliability_score']:.1%}
- Confidence Level: {result['confidence_level']}

**Recommendations:** {result['budget_analysis']['budget_range']}

**Next Steps:** {', '.join(result['next_actions'])}
                    """
                    
                    if result['targeted_questions']:
                        agent_response += f"\n\n**Questions for you:** {result['targeted_questions'][0]}"
                    
                    if result['inconsistencies_detected']:
                        agent_response += f"\n\n**Note:** {len(result['inconsistencies_detected'])} inconsistencies detected in your input."
                    
                    st.session_state.conversation_history.append({"role": "agent", "content": agent_response})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.conversation_history.append({"role": "agent", "content": error_msg})
            
            st.rerun()
        else:
            st.error("❌ Agent not initialized. Please refresh the page.")

# Traditional Analysis Mode
elif mode == "📊 Traditional Analysis":
    st.header("📊 Traditional Budget Analysis")
    
    with st.form("analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            city = st.selectbox(
                "🏙️ Target City:",
                ["Tunis", "Sousse", "Sfax", "Ariana", "Ben Arous", "Bizerte", "Kairouan", "Mahdia", "Monastir"]
            )
            
            budget = st.number_input(
                "💰 Budget (DT):",
                min_value=50000,
                max_value=5000000,
                value=300000,
                step=10000
            )
            
            preferences = st.text_input(
                "🏠 Property Preferences:",
                value="terrain villa appartement",
                help="Describe what type of property you're looking for"
            )
        
        with col2:
            min_size = st.number_input(
                "📐 Minimum Surface (m²):",
                min_value=50,
                max_value=2000,
                value=150,
                step=10
            )
            
            max_price = st.number_input(
                "💸 Maximum Price (DT):",
                min_value=budget,
                max_value=budget * 2,
                value=int(budget * 1.2),
                step=10000
            )
        
        submitted = st.form_submit_button("🔍 Analyze Market", type="primary")
    
    if submitted:
        client_profile = {
            "city": city,
            "budget": budget,
            "preferences": preferences,
            "min_size": min_size,
            "max_price": max_price
        }
        
        with st.spinner("📊 Analyzing market data..."):
            try:
                analysis = st.session_state.agent.analyze_client_budget(client_profile)
                st.session_state.analysis_results = analysis
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🏠 Properties Analyzed", 
                        f"{analysis.get('total_properties_analyzed', 0):,}"
                    )
                
                with col2:
                    st.metric(
                        "✅ Matching Properties", 
                        analysis['market_statistics']['inventory_count']
                    )
                
                with col3:
                    if analysis['market_statistics']['inventory_count'] > 0:
                        feasibility = analysis['market_statistics']['budget_feasibility']['feasibility_ratio']
                        st.metric("💯 Budget Feasibility", f"{feasibility:.1%}")
                    else:
                        st.metric("💯 Budget Feasibility", "N/A")
                
                with col4:
                    confidence = analysis['budget_analysis']['confidence_score']
                    st.metric("🎯 Confidence Score", f"{confidence:.1%}")
                
                # Detailed results
                if analysis['market_statistics']['inventory_count'] > 0:
                    market_stats = analysis['market_statistics']
                    budget_ai = analysis['budget_analysis']
                    
                    # Market Statistics
                    st.subheader("📈 Market Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**💰 Price Range:** {market_stats['price_stats']['min']:,.0f} - {market_stats['price_stats']['max']:,.0f} DT")
                        st.write(f"**📊 Average Price:** {market_stats['price_stats']['mean']:,.0f} DT")
                        st.write(f"**📊 Median Price:** {market_stats['price_stats']['median']:,.0f} DT")
                    
                    with col2:
                        st.write(f"**📏 Average Surface:** {market_stats['surface_stats']['mean']:.0f} m²")
                        st.write(f"**💵 Avg Price/m²:** {market_stats['price_per_m2_stats']['mean']:,.0f} DT/m²")
                    
                    # AI Recommendations
                    st.subheader("🎯 AI Recommendations")
                    
                    st.write(f"**✅ Budget Validation:** {budget_ai['budget_validation']}")
                    
                    if 'market_position' in budget_ai:
                        st.info(f"**📍 Market Position:** {budget_ai['market_position']}")
                    
                    st.write(f"**💡 Recommendations:** {budget_ai['recommendations']}")
                    
                    # Display the most compatible property with URL
                    if analysis.get('comparable_properties'):
                        most_compatible = st.session_state.agent.get_most_compatible_property(client_profile, analysis['comparable_properties'])
                        if most_compatible:
                            st.subheader("🏆 Most Compatible Property")
                            
                            prop = most_compatible['property_details']
                            
                            # Create two columns for property details
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**🏠 Title:** {prop['Title']}")
                                st.write(f"**💰 Price:** {prop['Price']:,.0f} DT")
                                st.write(f"**📐 Surface:** {prop['Surface']:.0f} m²")
                                st.write(f"**💵 Price per m²:** {prop['price_per_m2']:,.0f} DT/m²")
                            
                            with col2:
                                st.write(f"**📍 Location:** {prop['Location']}")
                                st.write(f"**🏗️ Type:** {prop['Type']}")
                                st.write(f"**🌟 Compatibility:** {most_compatible['compatibility_score']:.1%}")
                            
                            # Display URL as a clickable link
                            if prop['URL'] and prop['URL'] != 'No URL available':
                                st.markdown(f"**🔗 View Property:** [Click here to view property details]({prop['URL']})")
                            else:
                                st.write("**🔗 URL:** Not available")
                            
                            # Display compatibility explanation
                            st.info(f"**💡 Why this property matches:** {most_compatible['why_compatible']}")
                    
                    # Additional insights in expandable sections
                    if 'price_negotiation_tips' in budget_ai:
                        with st.expander("💬 Price Negotiation Tips"):
                            st.write(budget_ai['price_negotiation_tips'])
                    
                    if 'alternative_suggestions' in budget_ai:
                        with st.expander("🔄 Alternative Suggestions"):
                            st.write(budget_ai['alternative_suggestions'])
                    
                    if 'market_trends' in budget_ai:
                        with st.expander("📈 Market Trends"):
                            st.write(budget_ai['market_trends'])
                    
                    if 'risk_assessment' in budget_ai:
                        with st.expander("⚠️ Risk Assessment"):
                            st.write(budget_ai['risk_assessment'])
                
                else:
                    st.warning("⚠️ No properties found matching your criteria. Consider adjusting your parameters.")
                    st.write(f"**💡 Recommendation:** {analysis['budget_analysis']['recommendations']}")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

# Market Report Mode
elif mode == "📈 Market Report":
    st.header("📈 Market Report")
    
    # City selection for market report
    report_city = st.selectbox(
        "🏙️ Select City for Market Report:",
        ["All Cities", "Tunis", "Sousse", "Sfax", "Ariana", "Ben Arous", "Bizerte", "Kairouan", "Mahdia", "Monastir"]
    )
    
    if st.button("📊 Generate Market Report", type="primary"):
        st.info("📈 Market Report feature coming soon! Currently focused on budget analysis.")

# Footer
st.markdown("---")
st.markdown("🏠 **Architecture Budget Assistant** - Powered by AI | 🤖 Enhanced Budget Agent")
