import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


def get_data(parameter_file):
    df = pd.read_csv(parameter_file)
    return df


def plot_timeseries(df, ITMs_list, directory, Date_Now):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    color = iter(cm.viridis(np.linspace(0, 1, len(ITMs_list))))
    fig, ax = plt.subplots(1, 1)
    for ITMs in ITMs_list:
        df_ITM = df.loc[df['ITMs'] == ITMs]
        df_ITM['necrotic norm'] = df_ITM['necrotic']*100/2500.
        c = next(color)
        necrotic = df_ITM.groupby('timestep').mean()['necrotic norm']
        necrotic_std = df_ITM.groupby('timestep').std()['necrotic norm']

        ax.errorbar(x=range(len(necrotic)), y=necrotic, yerr=necrotic_std, label='', color=c, alpha=0.05)
        ax.plot(range(len(necrotic)), necrotic, label=str(ITMs) + ' ITMs', color=c)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_xlabel('Time Steps', fontsize=25)
    ax.set_ylabel('\% Necrosis', fontsize=25)

    ax.legend(fontsize=16)
    plt.savefig(directory + Date_Now + '_CA_timesteps_mean',  dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    project_dir = 'C:/Users/user/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/' \
                  'Neutrophils Game/results/Cellular Automata/Global ITMs/Time Steps/'
    parameter_file = 'C:/Users/user/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/' \
                  'Neutrophils Game/results/Cellular Automata/Global ITMs/Time Steps/2019-03-21_18-25_res.csv'

    ITMs_list = [0, 1, 10, 100, 500]
    Date_Now = pd.datetime.now().strftime("%Y-%m-%d_%H-%M")
    df = get_data(parameter_file)
    df_final = plot_timeseries(df, ITMs_list, project_dir, Date_Now)
