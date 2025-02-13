import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def run(file):
    # extract
    command = f"cat {file} | grep -a '\[.*\]' | cut -d' ' -f 4,5 > temp.log"
    os.system(command)
    name = file.split('/')[-1].replace('.out', '')

    # load logs
    logs = {'step': [], 'loss': [], 'type': name}
    with open('temp.log', 'r') as f:
        for line in f:
            line = line.strip().split()
            try:
                step = int(line[0])
                loss = float(line[1])
                logs['step'] += [step]
                logs['loss'] += [loss]
            except:
                pass

    # transform into figs
    df = pd.DataFrame(logs)
    print(df)
    return df, name

def plot(df, name='NA'):
    # Create the plot
    plt.figure(figsize=(8, 5))
    sns.lineplot(x="step", y="loss", hue="type", data=df)

    # Labels and title
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.ylim(0.125, 0.4)
    plt.axhline(y=0.193, color='b', linestyle='-', linewidth=0.5) # 0.2788 
    plt.axhline(y=0.187, color='r', linestyle='-', linewidth=0.5) # 0.2973 
    plt.axhline(y=0.181, color='r', linestyle='-', linewidth=0.5) # 0.3058

    plt.title(f"Loss Over Steps - {name}")

    # Show the plot
    plt.savefig(f"log/{name}.pdf", format="pdf", bbox_inches="tight")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", action="append", default=[])
    args = parser.parse_args()

    files = args.files
    if len(files) == 1:
        df, name = run(args.files[0])
        plot(df, name)
    else:
        dfs, names = [], []
        for file in files:
            df, name = run(file)
            dfs.append(df)
            names.append(name)

        df_combined = pd.concat(dfs, ignore_index=True)
        plot(df_combined, 'All')
