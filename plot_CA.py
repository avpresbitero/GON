import os
import pandas as pd
import matplotlib.pyplot as plt


def read_data(directory, params):
    results_dic = dict()
    results_dic['mean'] = []
    results_dic['std'] = []
    results_dic['ITMs'] = []
    for ITMs in params['ITMs'].keys():
        for filename in os.listdir(directory):
            if filename.endswith('_' + str(ITMs) + ".csv"):
                df_ITMs = pd.read_csv(directory + filename)
                std = df_ITMs['necrotic fraction'].std()
                mean = df_ITMs['necrotic fraction'].mean()
                results_dic['mean'].append(mean)
                results_dic['std'].append(std)
                results_dic['ITMs'].append(ITMs)

    results_df = pd.DataFrame(results_dic)
    return results_df


def loop_experiments(params, directory):
    dic_experiments = {}
    for experiment in params['experiments']:
        for stability in params['stability']:
            results_df = read_data(directory + experiment + '/' + stability + '/', params)
            dic_experiments[experiment, stability] = results_df
    return dic_experiments


def plot(dic_experiments, directory):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    Date_Now = pd.datetime.now().strftime("%Y-%m-%d_%H-%M")
    count = 0
    markers = ['o', '+', 'x', '^', ',']
    linestyles = ['-', '--', '-.', ':', '-']

    fig, ax = plt.subplots(1, 1)
    ms = 15
    for experiment, stability in dic_experiments.keys():
        results_df = dic_experiments[experiment, stability]
        results_df['mean'] += params['baseline']
        results_df.loc[results_df['mean']>1, 'mean'] = 1
        ax.errorbar(results_df['ITMs'], results_df['mean'] * 100, results_df['std']*100, markersize=ms,
                    label=experiment , marker=markers[count], linestyle=linestyles[count])
        count += 1
    ax.plot(params['ITMs'].keys(), [i*100 for i in params['ITMs'].values()], markersize=ms, label='Data', marker=markers[count],
            linestyle=linestyles[count])
    plt.xticks(fontsize=20)  # work on current fig
    plt.yticks(fontsize=20)  # work on current fig
    ax.set_xlabel('$ITMs_{initial}$', fontsize=25)
    ax.set_ylabel(r'\% Necrosis', fontsize=25)
    ax.legend(fontsize=20)
    plt.savefig(directory + Date_Now + '_CA_runs_mean',  dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    project_dir = 'C:/Users/user/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/Neutrophils ' \
                  'Game/results/Cellular Automata/'
    params = dict()
    params['baseline'] = 0.164
    params['ITMs'] = {500: 1, 100: 0.865, 10: 0.75 , 1: 0.55, 0: 0.164}
    params['experiments'] = ['Global ITMs', 'Local ITMs']
    params['stability'] = ['No ITMs']
    dic_experiments = loop_experiments(params, project_dir)
    plot(dic_experiments, project_dir)