from collections import deque
from main import Agent


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
    test_agent = TestAgent()
    test_agent.test_agent_initialization()
    test_agent.test_to_dict()
    print("Tests passed")

if __name__=="__main__":
    main()