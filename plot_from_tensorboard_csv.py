import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

if __name__ == "__main__":
    csv_files = ["./data/run-ppo_97956418-tag-rollout_ep_rew_mean.csv", "./data/run-ppo_306655929-tag-rollout_ep_rew_mean.csv", 
    "./data/run-ppo_3988534756-tag-rollout_ep_rew_mean.csv",]
    datasets = []
    for file in csv_files:
        data = pd.read_csv(file)
        datasets.append(data)
    data_frame = pd.concat(datasets, ignore_index = True, keys = "Step")
    data_frame.sort_values("Step", inplace=True)
    data_frame.reset_index(inplace=True)
    sns.lineplot(data=data_frame, x="Step", y="Value")
    plt.show()