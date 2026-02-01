import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import shortuuid



# initialization
def initialize_agents(n=100) -> pd.DataFrame:
    df = pd.DataFrame(columns=["id","generosity_score","acceptance_score","resources"])
    
    generosity_vals = np.linspace(0,1,11) #range(11)
    acceptance_vals = [1,]
    max_resources = 1_000

    
    for i in range(n):
        agent = {}
        agent["id"] = shortuuid.uuid()
        agent["generosity_score"] = random.choice(generosity_vals)
        agent["acceptance_score"] = acceptance_vals[0]
        agent["resources"] = max_resources
        df = pd.concat([df, pd.DataFrame(agent,index=[i])],ignore_index=False)
    return df


def single_iteration(df) -> pd.DataFrame:
    df = df.copy()
    receivers_list = list(df["id"])

    def find_random_receiver_id(sender_id, receivers):
        receivers = receivers.copy()
        idx = receivers.index(sender_id)
        receivers.pop(idx)
        return random.choice(receivers)

    def give(df,giver_id) -> pd.DataFrame:
        new_df = df.copy()
        receiver_id = find_random_receiver_id(giver_id, receivers_list)
        receiver_index = df[df["id"]==receiver_id].index
        giver_index = df[df["id"]==giver_id].index
        new_df.loc[receiver_index,"resources"] +=1
        new_df.loc[giver_index,"resources"] -=1

        return new_df

    for index, row in df.iterrows():
        
        if row["resources"] <= 0: # handles dead agents
            continue

        tendency = random.uniform(0,1)
        
        if tendency <= row["generosity_score"]:
            df = give(df,giver_id=row["id"])


    return df

def multiple_iterations(df, n=100):
    df = df.copy()

    for i in range(n):
        df = single_iteration(df)

    return df


def main():
    df = initialize_agents(n = 10)
    print(df.head(10))
    print("--"*10)
    df_1 = multiple_iterations(df, n=2_000)
    print(df_1.head(10))


if __name__ == "__main__":
    main()