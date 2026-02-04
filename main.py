import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import shortuuid
from collections import deque

config = {
    "ENV_INIT":{
            "N_AGENTS" : 10
            ,"N_ITERATIONS": 500
            ,"MAX_RESOURCES": 1_000
            ,"MIN_RESOURCES": 0.0 # floor for resources
            ,"SEED": 42
            ,"RANDOM_SAMPLING_RANGE": (0,1)
            ,"RESOURCE_TRANSFER_SIZE": 1.0
            }

    ,"AGENTS_INIT": {
        "GENEROSITY_RANGE": (0,1)
        ,"ACCEPTANCE_RANGE": (0,1)
        ,"GIVING_FLOOR": 1
        ,"RECEIVING_CEIL": 1e4
        ,"COST_OF_LIVING": 0.001 # assume food and other expenses? # let this be uniform for now

        }
    ,"AFFORDABILITY":
    {   "RESOURCE_MIN": 50
        ,"RESOURCE_MAX": 300
        ,"LOWER_LIMIT": 0.0
        ,"UPPER_LIMIT": 1.0
        

    }
    ,"MEMORY": {
        "MEMORY_SIZE": 5
        ,"DEFAULT_WEIGHT": 1.0
        ,"MEMORY_BONUS": 3.0
    }
    

}


random.seed(config["ENV_INIT"]["SEED"])
np.random.seed(config["ENV_INIT"]["SEED"])

# initialization


def initialize_agents(n=config["ENV_INIT"]["N_AGENTS"]) -> pd.DataFrame:
    df = pd.DataFrame(columns=["id","generosity_score","acceptance_score","resources"])
    
    generosity_vals = np.linspace(0,1,11) #range(11)
    acceptance_vals = np.linspace(0,1,11) #[1,]
    

    agents = []
    for i in range(n):
        agent = {}
        agent["id"] = shortuuid.uuid()
        agent["generosity_score"] = float(np.random.choice(generosity_vals, size=1))
        agent["acceptance_score"] = float(np.random.choice(acceptance_vals, size=1))
        agent["resources"] = config["ENV_INIT"]["MAX_RESOURCES"]
        agent["memory"] = deque(maxlen=config["MEMORY"]["MEMORY_SIZE"])
        agents.append(agent)

    df = pd.DataFrame(agents)
    df = df.set_index("id")
    return df


def single_iteration(df) -> pd.DataFrame:
    df = df.copy()
    agent_ids = list(df.index)

    def _affordability(resources, low=config["AFFORDABILITY"]["RESOURCE_MIN"], high=config["AFFORDABILITY"]["RESOURCE_MAX"]):
        if resources < low:
            return config["AFFORDABILITY"]["LOWER_LIMIT"]
        if resources > high:
            return config["AFFORDABILITY"]["UPPER_LIMIT"]
        return (resources - low) / (high - low)

    
    def _effective_generosity(row):
        return max(row["generosity_score"] * _affordability(row["resources"]) # to ensure it is within the specified range during initialization
                   ,config["AGENTS_INIT"]["GENEROSITY_RANGE"][0]
        )

    def _effective_acceptance(row):
        return min(row["acceptance_score"] * (2 - _affordability(row["resources"]))
                   ,config["AGENTS_INIT"]["ACCEPTANCE_RANGE"][1]
        )


    def apply_living_cost(df):
        df["resources"] -= 0 # df["resources"] * 0.001  #COST_OF_LIVING
        df["resources"] = df["resources"].clip(lower = config["ENV_INIT"]["MIN_RESOURCES"])
        return df

    def find_random_receiver_id(df, sender_id):
        return random.choice(df.index[df.index != sender_id])
    
    def find_receiver_id(df, sender_id, memory_bonus=config["MEMORY"]["MEMORY_BONUS"]):
        candidates = [i for i in df.index if i != sender_id]
        memory = df.loc[sender_id, "memory"]
        weights = []
        for c in candidates:
            w = config["MEMORY"]["DEFAULT_WEIGHT"]
            if c in memory:
                w+= memory_bonus * (len(memory) - memory.index(c)) / len(memory)
            weights.append(w)
        return random.choices(candidates, weights=weights, k=1)[0]
    
    def _is_capable_of_giving(row) -> bool:
        return (row["resources"] > config["AGENTS_INIT"]["GIVING_FLOOR"] 
                and random.uniform(*config["ENV_INIT"]["RANDOM_SAMPLING_RANGE"]) < _effective_generosity(row)
                )

    def _is_capable_of_receiving(row) -> bool:
        return (row["resources"] < config["AGENTS_INIT"]["RECEIVING_CEIL"] 
                and random.uniform(*config["ENV_INIT"]["RANDOM_SAMPLING_RANGE"]) < _effective_acceptance(row) 
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
        
        for giver, receiver in actions:
            df.at[giver, "resources"] -= config["ENV_INIT"]["RESOURCE_TRANSFER_SIZE"]
            df.at[receiver, "resources"] += config["ENV_INIT"]["RESOURCE_TRANSFER_SIZE"]
            df.at[receiver, "memory"].append(giver)

        return df

    transfer_actions = generate_transfer_actions(df)
    df = perform_transfers(df, transfer_actions)
    df = apply_living_cost(df)

    return df

def multiple_iterations(df, n=config["ENV_INIT"]["N_ITERATIONS"]):
    
    for i in range(n):
        df = single_iteration(df)

    return df


def main():
    df = initialize_agents()
    print(df.head(10))
    print("--"*10)
    df_1 = multiple_iterations(df)
    print(df_1.head(10))


if __name__ == "__main__":
    main()