import numpy as np
import matplotlib.pyplot as plt


def generate_parameter_space(b_apoptosis, c_necrosis, alpha, ITMs, m, n, params):
    necrotic_population = -((np.log((b_apoptosis + c_necrosis)/(alpha * (m - n)))
                            + alpha * n * params['N']
                            - alpha * ITMs ** params['k'])/(alpha * m * params['N']
                                             - alpha * n * params['N']))

    return necrotic_population


def get_params(experiment):
    if experiment == 1:
        params = {'ITMs': np.linspace(0, 1e8, 11),
                  'k' : 0.0929,
                  'N': 2500,
                  'm': 0.0015,
                  'n': np.linspace(0, 0.001, 500),
                  'alpha': 1.0,
                  'b_apoptosis': 0.0001,
                  'c_necrosis': np.linspace(0, 0.001, 500)
                  }
    elif experiment == 2:
        params = {'ITMs': np.linspace(0, 1e8, 11),
                  'k': 0.0929,
                  'N': 2500,
                  'm': 0.0008,
                  'n': 0.0004,
                  'alpha': np.linspace(0, 1, 500),
                  'b_apoptosis': np.linspace(0, 0.0002, 500),
                  'c_necrosis': 0.0001
                  }

    elif experiment == 3:
        params = {'ITMs': np.linspace(0, 1e8, 11),
                  'k': 0.0929,
                  'N': 2500,
                  'm': 0.0008,
                  'n': 0.0004,
                  'alpha': 1.0,
                  'b_apoptosis': np.linspace(0, 0.0005, 500),
                  'c_necrosis': np.linspace(0, 0.0005, 500)
                  }

    return params


def create_lattice(n, m):
    """
    :param n: rows
    :param m: columns
    :return: n x m zero matrix
    """
    return np.zeros((n, m))


def build_lattice(experiment, params, ITMs):
    lattice = create_lattice(500, 500)
    if experiment == 1:
        for i in range(len(params['c_necrosis'])):
            c_necrosis = params['c_necrosis'][i]
            for j in range(len(params['n'])):
                n = params['n'][j]
                necrotic_population = generate_parameter_space(params['b_apoptosis'], c_necrosis, params['alpha'], ITMs, params['m'], n, params)
                lattice[i, j] = 1.0 - necrotic_population
    elif experiment == 2:
        for i in range(len(params['alpha'])):
            alpha = params['alpha'][i]
            for j in range(len(params['b_apoptosis'])):
                b_apoptosis = params['b_apoptosis'][j]
                if alpha != 0:
                    necrotic_population = generate_parameter_space(b_apoptosis, params['c_necrosis'],
                                                                   alpha, ITMs, params['m'], params['n'], params)
                    lattice[i, j] = 1.0 - necrotic_population
                else:
                    lattice[i, j] = 0
    elif experiment == 3:
        for i in range(len(params['b_apoptosis'])):
            b_apoptosis = params['b_apoptosis'][i]
            for j in range(len(params['c_necrosis'])):
                c_necrosis = params['c_necrosis'][j]
                necrotic_population = generate_parameter_space(b_apoptosis, c_necrosis, params['alpha'], ITMs,
                                                               params['m'], params['n'], params)
                lattice[i, j] = 1.0 - necrotic_population
    return lattice


def plot_lattice(lattice, ITMs, directory, experiment, params):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    lattice = np.ma.masked_where(lattice < 0, lattice)
    lattice = np.ma.masked_where(lattice > 1, lattice)
    cmap = plt.get_cmap('Spectral')
    cmap.set_bad(color='white')
    fs = 30
    if experiment == 1:
        plt.imshow(lattice, cmap=cmap, origin='lower', aspect='auto', extent=[0, params['c_necrosis'][-1], 0, params['n'][-1]])
        plt.xlabel('$n$', fontsize=fs)
        plt.ylabel('$c_{necrosis}$', fontsize=fs)
    elif experiment == 2:
        plt.imshow(lattice, cmap=cmap, origin='lower',  aspect='auto', extent=[0, params['b_apoptosis'][-1], 0, params['alpha'][-1]])
        plt.xlabel('$b_{Apoptotis}$', fontsize=fs)
        plt.ylabel(r'$\alpha$', fontsize=fs)
    elif experiment == 3:
        plt.imshow(lattice, cmap=cmap, origin='lower', aspect='auto', extent=[0, params['c_necrosis'][-1], 0, params['b_apoptosis'][-1]])
        plt.xlabel('$c_{necrosis}$', fontsize=fs)
        plt.ylabel('$b_{apoptosis}$', fontsize=fs)
        plt.clim(0, 1)
        plt.colorbar()

    plt.xticks(fontsize=20)  # work on current fig
    plt.yticks(fontsize=20)  # work on current fig
    plt.title('$ITMs_{initial}=' + str(ITMs) + '$', fontsize=25)
    plt.savefig(directory + str(experiment) + '_' + str(ITMs) + '_lattice.png', dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    project_dir = 'C:/Users/user/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/Neutrophils ' \
                  'Game/results/Mean Field/Sensitivity Analysis/'

    experiment = 2
    ITMs = 500
    params = get_params(experiment)
    lattice = build_lattice(experiment, params, ITMs)
    plot_lattice(lattice, ITMs, project_dir, experiment, params)