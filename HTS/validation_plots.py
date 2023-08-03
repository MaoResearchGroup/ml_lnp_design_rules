import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy import stats
import seaborn as sns

def plot_scatter(df, pairs):
    # Set the style of the plot to white
    sns.set_style('white')

    # Create a color column, 'Group', based on index to differentiate first 32 and next 16 datapoints
    df['Group'] = np.where(df.index < 32, 'Historically High Performing', 'Historically Low Performing')
    
    fig, axes = plt.subplots(1, len(pairs), figsize=(15, 5))

    for i, (x, y) in enumerate(pairs):
        # Swap x and y for the 2nd and 3rd plots
        if i > 0:
            x, y = y, x

        # Calculate Spearman and Pearson correlation
        pearson_corr, _ = pearsonr(df[x], df[y])
        spearman_corr, _ = spearmanr(df[x], df[y])

        # Scatter plot with different colors for first 32 and next 16 datapoints
        scatter = sns.scatterplot(x=x, y=y, hue='Group', data=df, ax=axes[i], palette='Set1')
        
        # Set the title with y-axis label coming first
        axes[i].set_title(f'{y} RLU vs {x} RLU')

        # Add text for Spearman and Pearson correlation
        axes[i].text(df[x].max()*0.1, df[y].max()*0.9, f"Pearson's r = {round(pearson_corr, 3)}")
        axes[i].text(df[x].max()*0.1, df[y].max()*0.8, f"Spearman's R = {round(spearman_corr, 3)}")
        
        # Plot correlation line
        sns.lineplot(x=np.unique(df[x]), y=np.poly1d(np.polyfit(df[x], df[y], 1))(np.unique(df[x])), color='red', ax=axes[i])
        
        # Plot perfect correlation line
        axes[i].plot([0, df[x].max()], [0, df[y].max()], color='grey', linestyle='--', lw=2)
        
        # Set x and y labels with 'RLU'
        axes[i].set_xlabel(f'{x} RLU')
        axes[i].set_ylabel(f'{y} RLU')

        # Handle legends
        scatter.legend_.remove()

    # Add a single legend outside of the subplots
    handles, labels = scatter.get_legend_handles_labels()
    fig.legend(handles, labels, title='Group', loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    
    plt.savefig('SpearmanPlot.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_scatter_ranks(df, pairs):
    sns.set_style('white')

    # Create a color column, 'Group', based on index
    df['Group'] = np.where(df.index < 32, 'Historically High Performing', 'Historically Low Performing')

    df_rank = df.rank(ascending=False)

    fig, axes = plt.subplots(1, len(pairs), figsize=(5*len(pairs), 5))

    for i, (x, y) in enumerate(pairs):
        if i > 0:
            x, y = y, x

        pearson_corr, _ = pearsonr(df_rank[x], df_rank[y])
        spearman_corr, _ = spearmanr(df_rank[x], df_rank[y])

        scatter = sns.scatterplot(x=y, y=x, hue='Group', data=df_rank, ax=axes[i], palette=['blue', 'red'])  # Change the palette colors

        axes[i].set_title(f'{x} Rank vs {y} Rank')
        axes[i].text(0.05, 0.95, f"Pearson's r = {round(pearson_corr, 3)}", transform=axes[i].transAxes)
        axes[i].text(0.05, 0.90, f"Spearman's R = {round(spearman_corr, 3)}", transform=axes[i].transAxes)

        sns.lineplot(x=np.unique(df_rank[y]), y=np.poly1d(np.polyfit(df_rank[y], df_rank[x], 1))(np.unique(df_rank[y])),
                     color='red', ax=axes[i])

        axes[i].plot([df_rank[y].max(), df_rank[y].min()], [df_rank[x].max(), df_rank[x].min()], color='grey',
                     linestyle='--', lw=2)

        axes[i].set_xlabel(f'{y} Rank')
        axes[i].set_ylabel(f'{x} Rank')
        axes[i].invert_xaxis()
        axes[i].invert_yaxis()

        scatter.legend_.remove()

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) 
               for color in ['red', 'blue']]
    labels = ['Historically High Performing', 'Historically Low Performing']

    fig.legend(handles, labels, title='Group', loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()

    plt.savefig('SpearmanPlotRanks.png', dpi=300, bbox_inches='tight')
    plt.show()




def evaluate_pred_quality(exp, pred):
    #evaluate accuracy
    acc = mean_absolute_error(exp, pred)
    spearmans_rank = stats.spearmanr(exp, pred)
    pearsons_r = stats.pearsonr(exp, pred)

    return acc, spearmans_rank[0], pearsons_r[0]


################ INPUT PARAMETERS ############################################
plot_save_path = "HTS/"

def main():
  ################ Retreive Data ##############################################

    df = pd.read_csv(f'HTS/230622_flow.csv' )


    df["Manual_Rank"] = df["Manual"].rank(ascending = False) #Add ranking
    df["Auto_Rank"] = df["Auto"].rank(ascending = False) #Add ranking
    print(df)
    acc, spearman, pearson = evaluate_pred_quality(df.loc[df['Type'] == "Experimental", "Experimental_RLU"], df.loc[df['Type'] == "Experimental", "Predicted_RLU"])
    # Specify pairs to plot
    pairs = [('Manual', 'Auto')]

    # Generate scatter plots
    plot_scatter(df, pairs)
    

if __name__ == "__main__":
    main()
