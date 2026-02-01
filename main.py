import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import shortuuid

GIVING_FLOOR = 1
RECEIVING_CEIL = 1e4

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
        agents.append(agent)

    df = pd.DataFrame(agents)
    df = df.set_index("id")
    return df


def single_iteration(df) -> pd.DataFrame:
    df = df.copy()
    agent_ids = list(df.index)

    def find_random_receiver_id(sender_id, agent_ids):
        return random.choice(df.index[df.index != sender_id])
    
    
    def _is_capable_of_giving(row) -> bool:
        return (row["resources"] > GIVING_FLOOR and random.uniform(0,1) < row["generosity_score"])

    def _is_capable_of_receiving(row) -> bool:
        return (row["resources"] < RECEIVING_CEIL and random.uniform(0,1) < row["acceptance_score"])

    def generate_transfer_actions(df):
        actions = []
        for id in df.index:        
            if _is_capable_of_giving(df.loc[id]): 
                receiver = find_random_receiver_id(id, agent_ids)
                if _is_capable_of_receiving(df.loc[receiver,:]):
                    actions.append((id, receiver))
        return actions
        
    def perform_transfers(df, actions) -> pd.DataFrame:
        
        resources = df["resources"].copy()
        
        for giver, receiver in actions:
            resources[giver] -= 1
            resources[receiver] += 1

        df["resources"] = resources        
        return df

    transfer_actions = generate_transfer_actions(df)
    df = perform_transfers(df, transfer_actions)

    return df

def multiple_iterations(df, n=100):
    
    for i in range(n):
        df = single_iteration(df)

    return df


def main():
    df = initialize_agents(n = 10)
    print(df.head(10))
    print("--"*10)
    df_1 = multiple_iterations(df, n=10_000)
    print(df_1.head(10))


if __name__ == "__main__":
    main()