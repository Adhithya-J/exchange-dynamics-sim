from collections import deque
from main import Agent, MetricsCalculator, AffordabilityCalculator
import math

class TestMetricsCalculator:

    def test_gini_calculation(self):
        test_cases = [
            ([5, 5, 5, 5, 5,5, 5.0], 0.0),
            ([1, 2, 3, 4, 5],0.2666666667),
            ([1, 1, 1, 1, 10],0.5142857143),
            ([0, 0, 1],0.6666666667),
            ([10, 1, 5, 2, 4, 3],0.366667),
            ([42], 0.0),
            ([0.49, 0.59, 0.69, 0.79, 1.89, 2.55, 5.0, 10.0, 18.0, 60.0],0.7172999)
        ]
        for values, expected in test_cases: 
            assert math.isclose(MetricsCalculator._calculate_gini(values), expected, rel_tol=1e-6)

class TestAffordabilityCalculator:

    def test_affordability_check(self):
        config = {
            "AGENTS_INIT": {
                "GENEROSITY_RANGE":(0,1)
                ,"ACCEPTANCE_RANGE":(0,1)

            },
            "AFFORDABILITY" : {
                "RESOURCE_MIN":50
                ,"RESOURCE_MAX": 300
                ,"LOWER_LIMIT": 0.0
                ,"UPPER_LIMIT":1.0
            }
        }
        test_cases = [
            (20,0.0)
            ,(350,1.0)
            ,(175,0.5)
        ]
        afc = AffordabilityCalculator(config)
        for value, expected in test_cases:
            assert math.isclose(afc._affordability(value),expected)
    
    def test_effective_generosity(self):
        config = {
            "AGENTS_INIT": {
                "GENEROSITY_RANGE":(0,1)
                ,"ACCEPTANCE_RANGE":(0,1)

            },
            "AFFORDABILITY" : {
                "RESOURCE_MIN":50
                ,"RESOURCE_MAX": 300
                ,"LOWER_LIMIT": 0.0
                ,"UPPER_LIMIT":1.0
            }
        }

        test_cases = [
            ({"generosity_score": 0.0, "resources":50}, 0.0)
            ,({"generosity_score": 0.5, "resources":500}, 0.5)
            ,({"generosity_score": 1.0, "resources":500}, 1.0)
        ]
        calculator = AffordabilityCalculator(config)
        for value, expected in test_cases:
            assert math.isclose(calculator.effective_generosity(value),expected)

    def test_effective_acceptance(self):
        config = {
            "AGENTS_INIT": {
                "GENEROSITY_RANGE":(0,1)
                ,"ACCEPTANCE_RANGE":(0,1)

            },
            "AFFORDABILITY" : {
                "RESOURCE_MIN":50
                ,"RESOURCE_MAX": 300
                ,"LOWER_LIMIT": 0.0
                ,"UPPER_LIMIT":1.0
            }
        }

        test_cases = [
            ({"acceptance_score": 0.0, "resources":50}, 0.0)
            ,({"acceptance_score": 1.0, "resources":500}, 1.0)
            ,({"acceptance_score": 0.5, "resources":50}, 1.0)
        ]
        calculator = AffordabilityCalculator(config)
        for value, expected in test_cases:
            assert math.isclose(calculator.effective_acceptance(value),expected)



class TestAgent:
    def test_agent_initialization(self):
        agent = Agent(id ="test_id", 
                    generosity_score=0.8, 
                    acceptance_score=0.7, 
                    initial_resources=1000, 
                    memory_size=5)
        
        assert agent.id ==  "test_id"
        assert agent.generosity_score == 0.8 
        assert agent.acceptance_score == 0.7 
        assert agent.resources == 1000 
        assert agent.memory.maxlen == 5
    
    def test_to_dict(self):
        agent = Agent(id ="test_id", 
                    generosity_score=0.8, 
                    acceptance_score=0.7, 
                    initial_resources=1000, 
                    memory_size=5)
        agent_dict = agent.to_dict()
        assert agent_dict["id"] ==  "test_id"
        assert agent_dict["generosity_score"] == 0.8 
        assert agent_dict["acceptance_score"] == 0.7 
        assert agent_dict["resources"] == 1000 
        assert isinstance(agent_dict["memory"], deque)

def main():
    print("Tests started")
    test_metrics = TestMetricsCalculator()
    test_metrics.test_gini_calculation()
    test_affordability = TestAffordabilityCalculator()
    test_affordability.test_affordability_check()
    test_affordability.test_effective_generosity()
    test_affordability.test_effective_generosity()
    
    test_agent = TestAgent()
    test_agent.test_agent_initialization()
    test_agent.test_to_dict()

    print("Tests passed")