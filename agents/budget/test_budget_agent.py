import pytest
import types
import os
import sys
import pandas as pd
import numpy as np
from agents.budget import budget

# --- Fixtures ---
@pytest.fixture(scope="module")
def agent():
    return budget.SimpleBudgetAgent()

# --- Tests for property loading ---
def test_properties_loaded(agent):
    assert isinstance(agent.sample_properties, list)
    # Should load at least one property if CSVs exist
    assert all(isinstance(p, dict) for p in agent.sample_properties)

# --- Tests for budget extraction ---
def test_extract_budget_info_basic(agent):
    text = "Je cherche un terrain à Tunis avec un budget de 300000 DT"
    info = agent.extract_budget_info(text)
    assert info['extracted_budget'] == 300000
    assert info['city'] == 'Tunis'
    assert info['property_type'] == 'terrain'
    assert info['confidence'] > 0

@pytest.mark.parametrize("text,expected", [
    ("Je veux un appartement à Sousse pour 150 mille DT", (150000, 'Sousse', 'appartement')),
    ("Maison à Kairouan entre 100000 et 200000 DT", (150000, 'Kairouan', 'maison')),
    ("Villa à Bizerte pour 500k DT", (500000, 'Bizerte', 'villa')),
])
def test_extract_budget_info_various(agent, text, expected):
    info = agent.extract_budget_info(text)
    assert info['extracted_budget'] == expected[0]
    assert info['city'] == expected[1]
    assert info['property_type'] == expected[2]

# --- Tests for property search ---
def test_search_properties_returns_list(agent):
    props = agent.search_properties(300000, city="Tunis", property_type="terrain")
    assert isinstance(props, list)
    for p in props:
        assert isinstance(p, dict)
        assert 'Price' in p
        assert 'City' in p
        assert 'Type' in p

# --- Tests for conversation memory ---
def test_update_conversation_memory(agent):
    before = agent.conversation_memory['interaction_count']
    agent.update_conversation_memory("test", {'extracted_budget': 123456, 'city': 'Sousse'})
    after = agent.conversation_memory['interaction_count']
    assert after == before + 1
    assert agent.conversation_memory['last_budget'] == 123456
    assert agent.conversation_memory['last_city'] == 'Sousse'

# --- Tests for contextual response ---
def test_get_contextual_response(agent):
    info = {'extracted_budget': 200000, 'city': 'Sousse', 'property_type': 'terrain'}
    resp = agent.get_contextual_response("test", info)
    assert isinstance(resp, str)
    assert "Budget" in resp or "budget" in resp

# --- Tests for process_message ---
def test_process_message(agent):
    result = agent.process_message("Je cherche un terrain à Tunis avec un budget de 300000 DT")
    assert isinstance(result, dict)
    assert 'agent_response' in result
    assert 'budget_analysis' in result
    assert 'should_search' in result

# --- Tests for property recommendation reason ---
def test_get_recommendation_reason(agent):
    # Use a sample property from loaded data if available
    if agent.sample_properties:
        prop = agent.sample_properties[0]
        reason = agent._get_recommendation_reason(prop, prop['Price'], prop.get('City', None))
        assert isinstance(reason, str)

# --- Tests for analyze_client_budget ---
def test_analyze_client_budget(agent):
    profile = {'budget': 300000, 'city': 'Tunis'}
    analysis = agent.analyze_client_budget(profile)
    assert 'comparable_properties' in analysis
    assert 'market_statistics' in analysis

# --- Tests for suggestions ---
def test_generate_contextual_suggestions(agent):
    info = {'extracted_budget': 300000, 'city': 'Tunis', 'property_type': 'terrain'}
    suggestions = agent._generate_contextual_suggestions(info, True)
    assert isinstance(suggestions, list)
    assert suggestions
