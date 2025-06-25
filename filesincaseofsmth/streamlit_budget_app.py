import streamlit as st
from filesincaseofsmth.budget_agent import EnhancedBudgetAgent

st.set_page_config(page_title="Tunisian Real Estate Budget Assistant", layout="centered")

st.title("🏠 Tunisian Real Estate Budget Assistant")
st.write("Estimate and validate your property budget with AI-powered insights.")

# Sidebar for user input
st.sidebar.header("Client Profile")
city = st.sidebar.text_input("City", value="Tunis")
budget = st.sidebar.number_input("Budget (DT)", min_value=10000, max_value=2_000_000, value=200_000, step=1000)
property_type = st.sidebar.selectbox("Property Type", ["Appartement", "Villa", "Terrain", "Duplex", "Studio", "Any"], index=0)
min_size = st.sidebar.number_input("Minimum Surface Area (m²)", min_value=0, max_value=1000, value=80, step=10)
max_price = st.sidebar.number_input("Maximum Price (DT)", min_value=0, max_value=2_000_000, value=budget, step=1000)
preferences = st.sidebar.text_area("Other Preferences", value="proche centre ville, calme, sécurisé")

st.sidebar.markdown("---")
st.sidebar.write("Powered by EnhancedBudgetAgent")

# Main interface
st.header("Enter Your Project Details")
client_input = st.text_area("Describe your project, budget, and requirements (optional):", "")

if st.button("Analyze Budget"):
    with st.spinner("Analyzing your budget..."):
        agent = EnhancedBudgetAgent()
        # Use structured input if provided, else fallback to text
        if client_input.strip():
            result = agent.process_client_input(client_input)
            st.subheader("Extracted Budget Information")
            st.json(result["budget_analysis"])
            st.subheader("Targeted Questions")
            for q in result["targeted_questions"]:
                st.write("- " + q)
            st.subheader("Suggestions")
            for s in result["suggestions"]:
                st.write("- " + s)
            st.subheader("Reliability Score")
            st.write(f"{result['reliability_score']:.1%} ({result['confidence_level']})")
            st.subheader("Next Actions")
            st.write(result["next_actions"])
        else:
            client_profile = {
                "city": city,
                "budget": budget,
                "preferences": preferences,
                "min_size": min_size,
                "max_price": max_price,
                "property_type": property_type if property_type != "Any" else None
            }
            analysis = agent.analyze_client_budget(client_profile)
            st.subheader("Market Statistics")
            st.json(analysis["market_statistics"])
            st.subheader("AI Recommendations")
            st.json(analysis["budget_analysis"])
            
            # Show best compatible offer if available
            if analysis["comparable_properties"]:
                best_offer = analysis["comparable_properties"][0]
                st.subheader("🌟 Best Compatible Offer")
                # Show all available fields in a table
                st.table({k: v for k, v in best_offer.items()})
                # Show URL if available
                if best_offer.get('url'):
                    st.markdown(f"[View Property]({best_offer['url']})", unsafe_allow_html=True)
                else:
                    st.info("No URL available for this property.")
                st.markdown("---")
            else:
                st.warning("No compatible offers found for your criteria.")

            st.subheader("Top Comparable Properties")
            st.json(analysis["comparable_properties"])
            st.write(f"Total properties analyzed: {analysis['total_properties_analyzed']}")

    st.success("Analysis complete!")

st.info("Tip: For best results, fill in as many details as possible.")