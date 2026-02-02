import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import shortuuid
from collections import deque

GIVING_FLOOR = 1
RECEIVING_CEIL = 1e4
# COST_OF_LIVING = 0.2 # assume food and other expenses? # let this be uniform for now
MIN_RESOURCES = 0.0 # floor for resources

random.seed(42)
np.random.seed(42)

# initialization


def initialize_agents(n=100) -> pd.DataFrame:
    df = pd.DataFrame(columns=["id","generosity_score","acceptance_score","resources"])
    
    generosity_vals = np.linspace(0,1,11) #range(11)
    acceptance_vals = np.linspace(0,1,11) #[1,]
    max_resources = 1_000

    agents = []
    for i in range(n):
        agent = {}
        agent["id"] = shortuuid.uuid()
        agent["generosity_score"] = random.choice(generosity_vals)
        agent["acceptance_score"] = random.choice(acceptance_vals)
        agent["resources"] = max_resources
        agent["memory"] = deque(maxlen=5)
        agents.append(agent)

    df = pd.DataFrame(agents)
    df = df.set_index("id")
    return df


def single_iteration(df) -> pd.DataFrame:
    df = df.copy()
    agent_ids = list(df.index)

    def _affordability(resources, low=50, high=300):
        if resources < low:
            return 0.0
        if resources > high:
            return 1.0
        return (resources - low) / (high - low)
    
    def _effective_generosity(row):
        return row["generosity_score"] * _affordability(row["resources"])

    def _effective_acceptance(row):
        return row["acceptance_score"] * (2 - _affordability(row["resources"]))



    def apply_living_cost(df):
        df["resources"] -= df["resources"] * 0.0001  #COST_OF_LIVING
        df["resources"] = df["resources"].clip(lower = MIN_RESOURCES)
        return df

    def find_random_receiver_id(df, sender_id):
        return random.choice(df.index[df.index != sender_id])
    
    def find_receiver_id(df, sender_id, memory_bonus=2.0):
        candidates = [i for i in df.index if i != sender_id]
        memory = df.loc[sender_id, "memory"]
        weights = []
        for c in candidates:
            w = 1.0
            if c in memory:
                w+= memory_bonus * (len(memory) - memory.index(c)) / len(memory)
            weights.append(w)
        return random.choices(candidates, weights=weights, k=1)[0]
    
    def _is_capable_of_giving(row) -> bool:
        return (row["resources"] > GIVING_FLOOR 
                and random.uniform(0,1) < _effective_generosity(row)
                                             #row["generosity_score"]
                )

    def _is_capable_of_receiving(row) -> bool:
        return (row["resources"] < RECEIVING_CEIL 
                and random.uniform(0,1) < max(1,_effective_acceptance(row)) 
                #row["acceptance_score"]
                )

    def generate_transfer_actions(df):
        actions = []
        for id in df.index:        
            if _is_capable_of_giving(df.loc[id]): 
                receiver = find_receiver_id(df,id) # find_random_receiver_id(df,id) #  #
                if _is_capable_of_receiving(df.loc[receiver,:]):
                    actions.append((id, receiver))
        return actions
        
    def perform_transfers(df, actions) -> pd.DataFrame:
        
        resources = df["resources"].copy()
        memory = df["memory"].copy()

        for giver, receiver in actions:
            resources[giver] -= 1
            resources[receiver] += 1
            memory[receiver].append(giver)

        df["resources"] = resources        
        return df

    transfer_actions = generate_transfer_actions(df)
    df = perform_transfers(df, transfer_actions)
    df = apply_living_cost(df)

    return df

def multiple_iterations(df, n=100):
    
    for i in range(n):
        df = single_iteration(df)

    return df


def main():
    df = initialize_agents(n = 10)
    print(df.head(10))
    print("--"*10)
    df_1 = multiple_iterations(df, n=5000)
    print(df_1.head(10))


if __name__ == "__main__":
    main()