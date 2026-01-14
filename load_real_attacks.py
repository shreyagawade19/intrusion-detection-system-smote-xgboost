import pandas as pd
df = pd.read_csv("UNSW_NB15_training-set.csv")
row = df[df['attack_cat']=="Backdoor"].iloc[0]
print(row)

