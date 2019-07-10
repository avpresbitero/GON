import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from matplotlib.colors import from_levels_and_colors
from joblib import Parallel, delayed
from timeit import default_timer as timer

"""
General steps for systemmic inflammation:


1. Start with lattice size NxM with initial activated neutrophils population and initial concentration of ITMs.
2. Algorithm randomly chooses an activated neutrophil. Activated neutrophil plays the game with its neighbors. Note,
however, that the payoff matrix only works for necrotic and apoptotic neutrophils. Hence, without these two,
a game is not played.

A global payoff still is computed by calculating for the remaining number of ITMs if the chosen activated neutrophil
goes into apoptotis or necrosis. In other words, when an activated neutrophil is paired with another activated
neutrophil, the global payoff only depends on the neutralizing capacity of going into necrosis/apoptosis. If going into
necrosis neutralizes more ITMs than going into apoptosis, then it is likely that the activated neutrophil goes into
necrosis as well. The same goes if apoptosis poses more benefits.

3. Sum of local and global payoffs are calculated for each strategy. The activated neutrophil then adapts the strategy
corresponding to the higher payoff.

3. ITM concentration is recalculated per time step, where one timestep pertains to a single activated neutrophil choosing
a strategy.

"""

__author__ = "Alva Presbitero"


def set_ITMs_remaining(new_ITMs_remaining):
    global ITMs_remaining
    ITMs_remaining = new_ITMs_remaining


def create_lattice(n, m):
    """
    :param n: rows
    :param m: columns
    :return: n x m zero matrix
    """
    return np.zeros((n, m))


def generate_random_pairs(n, m, index):
    """
    :return: list of random pairs within the lattice
    """
    n_list = list(range(0, n))
    m_list = list(range(0, m))

    pairs = []
    for r in itertools.product(n_list, m_list):
        pairs.append((r[0], r[1]))
    np.random.seed(index)
    np.random.shuffle(pairs)
    return pairs


def fill_lattice(lattice, params, index):
    """
    Fills lattice with activated neutrophils.
    :param lattice: empty lattice
    :param params:
    :return: lattice with activated neutrophils
    """
    np.random.seed(index)
    n, m = np.shape(lattice)
    count = 0
    pairs = generate_random_pairs(n, m, index)
    while count < params['active_neutrophils']:
        x, y = pairs[count]
        if lattice[x, y] == params['_empty']:
            lattice[x, y] = params['_activated']
            count += 1
    return lattice


def get_activated_neighbors(lattice, i, j, params):
    activated_neighbors = 0
    x = params['x']
    y = params['y']

    if lattice[i, (j - 1) % y] == params['_activated'] or lattice[i, (j - 1) % y] == params['_empty']:
        activated_neighbors += 1
    if lattice[(i - 1) % x, j] == params['_activated'] or lattice[(i - 1) % x, j] == params['_empty']:
        activated_neighbors += 1
    if lattice[(i + 1) % x, j] == params['_activated'] or lattice[(i + 1) % x, j] == params['_empty']:
        activated_neighbors += 1
    if lattice[i, (j + 1) % y] == params['_activated'] or lattice[i, (j + 1) % y] == params['_empty']:
        activated_neighbors += 1

    if lattice[(i - 1) % x, (j - 1) % y] == params['_activated'] or lattice[i, (j - 1) % y] == params['_empty']:
        activated_neighbors += 1
    if lattice[(i + 1) % x, (j - 1) % y] == params['_activated'] or lattice[i, (j - 1) % y] == params['_empty']:
        activated_neighbors += 1
    if lattice[(i + 1) % x, (j + 1) % y] == params['_activated'] or lattice[i, (j - 1) % y] == params['_empty']:
        activated_neighbors += 1
    if lattice[(i - 1) % x, (j + 1) % y] == params['_activated'] or lattice[i, (j - 1) % y] == params['_empty']:
        activated_neighbors += 1
    return activated_neighbors


def check_if_all_activated(lattice, i, j, params):
    activated_neighbors = get_activated_neighbors(lattice, i, j, params)
    if activated_neighbors == params['neighbors']:
        return True
    return False


def get_local_pay_off(lattice, i, j, params):
    """
    :param lattice:
    :param i: current x position
    :param j: current y position
    :param params:
    :return: local payoff, which depends on the Moore neighborhood, note that this method only takes necrotic
    and apoptotic neutrophils into account. For instance, an activated neutrophil with ALL neighbors activated would
    take global payoff into account.
    """
    necrotic_payoff = get_local_payoff_necrotic(lattice, i, j, params)
    apoptotic_payoff = get_local_payoff_apoptotic(lattice, i, j, params)

    if necrotic_payoff >= apoptotic_payoff:
        return necrotic_payoff, params['_necrotic']
    else:
        # under the assumption that the body goes into apoptosis when the payoff to go into either pathway is the same
        return apoptotic_payoff, params['_apoptotic']


def get_local_payoff_necrotic(lattice, i, j, params):
    pay_off = 0.0
    x = params['x']
    y = params['y']

    if lattice[i, (j - 1) % y] == params['_necrotic']:
        pay_off += params['A']
    if lattice[(i - 1) % x, j] == params['_necrotic']:
        pay_off += params['A']
    if lattice[(i + 1) % x, j] == params['_necrotic']:
        pay_off += params['A']
    if lattice[i, (j + 1) % y] == params['_necrotic']:
        pay_off += params['A']

    if lattice[(i - 1) % x, (j - 1) % y] == params['_necrotic']:
        pay_off += params['A']
    if lattice[(i + 1) % x, (j - 1) % y] == params['_necrotic']:
        pay_off += params['A']
    if lattice[(i + 1) % x, (j + 1) % y] == params['_necrotic']:
        pay_off += params['A']
    if lattice[(i - 1) % x, (j + 1) % y] == params['_necrotic']:
        pay_off += params['A']

    if lattice[i, (j - 1) % y] == params['_apoptotic']:
        pay_off += params['B']
    if lattice[(i - 1) % x, j] == params['_apoptotic']:
        pay_off += params['B']
    if lattice[(i + 1) % x, j] == params['_apoptotic']:
        pay_off += params['B']
    if lattice[i, (j + 1) % y] == params['_apoptotic']:
        pay_off += params['B']

    if lattice[(i - 1) % x, (j - 1) % y] == params['_apoptotic']:
        pay_off += params['B']
    if lattice[(i + 1) % x, (j - 1) % y] == params['_apoptotic']:
        pay_off += params['B']
    if lattice[(i + 1) % x, (j + 1) % y] == params['_apoptotic']:
        pay_off += params['B']
    if lattice[(i - 1) % x, (j + 1) % y] == params['_apoptotic']:
        pay_off += params['B']
    return pay_off


def get_local_payoff_apoptotic(lattice, i, j, params):
    pay_off = 0
    x = params['x']
    y = params['y']
    if lattice[i, (j - 1) % y] == params['_necrotic']:
        pay_off += params['C']
    if lattice[(i - 1) % x, j] == params['_necrotic']:
        pay_off += params['C']
    if lattice[(i + 1) % x, j] == params['_necrotic']:
        pay_off += params['C']
    if lattice[i, (j + 1) % y] == params['_necrotic']:
        pay_off += params['C']

    if lattice[(i - 1) % x, (j - 1) % y] == params['_necrotic']:
        pay_off += params['C']
    if lattice[(i + 1) % x, (j - 1) % y] == params['_necrotic']:
        pay_off += params['C']
    if lattice[(i + 1) % x, (j + 1) % y] == params['_necrotic']:
        pay_off += params['C']
    if lattice[(i - 1) % x, (j + 1) % y] == params['_necrotic']:
        pay_off += params['C']

    if lattice[i, (j - 1) % y] == params['_apoptotic']:
        pay_off += params['D']
    if lattice[(i - 1) % x, j] == params['_apoptotic']:
        pay_off += params['D']
    if lattice[(i + 1) % x, j] == params['_apoptotic']:
        pay_off += params['D']
    if lattice[i, (j + 1) % y] == params['_apoptotic']:
        pay_off += params['D']

    if lattice[(i - 1) % x, (j - 1) % y] == params['_apoptotic']:
        pay_off += params['D']
    if lattice[(i + 1) % x, (j - 1) % y] == params['_apoptotic']:
        pay_off += params['D']
    if lattice[(i + 1) % x, (j + 1) % y] == params['_apoptotic']:
        pay_off += params['D']
    if lattice[(i - 1) % x, (j + 1) % y] == params['_apoptotic']:
        pay_off += params['D']
    return pay_off


def get_global_apoptotic_cost(lattice, params):
    count_apoptotic, count_necrotic = count_neutrophils(lattice, params)
    global_cost_apoptotic = params['alpha'] * (params['ITMs'] ** params['k'] - count_necrotic * params['m'] \
                              - (count_apoptotic + 1) * params['n'])
    if global_cost_apoptotic >= 0:
        return global_cost_apoptotic
    return 0.0


def get_global_necrotic_cost(lattice, params):
    count_apoptotic, count_necrotic = count_neutrophils(lattice, params)
    global_cost_necrotic = params['alpha'] * (params['ITMs'] ** params['k'] - (count_necrotic + 1) * params['m'] \
                             - count_apoptotic * params['n'])
    if global_cost_necrotic >= 0:
        return global_cost_necrotic
    return 0.0


def get_global_cost(lattice, params):
    global_cost_apoptotic = get_global_apoptotic_cost(lattice, params)
    global_cost_necrotic = get_global_necrotic_cost(lattice, params)
    if global_cost_necrotic <= global_cost_apoptotic and global_cost_necrotic >= 0:
        return global_cost_necrotic, params['_necrotic']
    elif global_cost_necrotic > global_cost_apoptotic and global_cost_apoptotic >= 0:
        return global_cost_apoptotic, params['_apoptotic']
    else:
        # always goes into apoptosis when there is no information
        return 0, params['_apoptotic']


def count_neutrophils(lattice, params):
    unique, counts = np.unique(lattice, return_counts=True)
    component_dictionary = dict(zip(unique, counts))
    if params['_apoptotic'] in component_dictionary.keys():
        count_apoptotic = component_dictionary[params['_apoptotic']]
    if params['_apoptotic'] not in component_dictionary.keys():
        count_apoptotic = 0
    if params['_necrotic'] in component_dictionary.keys():
        count_necrotic = component_dictionary[params['_necrotic']]
    if params['_necrotic'] not in component_dictionary.keys():
        count_necrotic = 0
    return count_apoptotic, count_necrotic


def compute_global_cost(lattice, params):
    count_apoptotic, count_necrotic = count_neutrophils(lattice, params)
    global_cost = params['alpha'] * (params['ITMs'] ** params['k'] - count_necrotic * params['m'] - count_apoptotic * params['n'])
    if global_cost >= 0:
        params['b_necrosis'] = global_cost
        return global_cost
    params['b_necrosis'] = 0
    return 0


def get_payoff_matrix_(params):
    params['A'] = - params['c_necrosis'] + 2.0 * params['b_necrosis']
    params['B'] = - params['c_necrosis'] + params['b_apoptosis'] + params['b_necrosis']
    params['C'] = params['b_apoptosis'] + params['b_necrosis']
    params['D'] = 2.0 * params['b_apoptosis']
    return params


def get_payoff_matrix(params):
    params['A'] = - params['c_necrosis']
    params['B'] = - params['c_necrosis'] + params['b_apoptosis']
    params['C'] = params['b_apoptosis']
    params['D'] = 2.0 * params['b_apoptosis']
    return params


def print_results(lattice, params, destination_folder, first_line):
    count_apoptotic, count_necrotic = count_neutrophils(lattice, params)
    if first_line == 0:
        with open(destination_folder + 'CA_results.csv', 'w') as fd:
            fd.write('b_apoptosis,'
                     'c_necrosis,'
                     'm,'
                     'n,'
                     'alpha,'
                     'apoptotic,'
                     'necrotic' + "\n")
        first_line = 1
    else:
        with open(destination_folder + 'data_param_space.csv', 'a') as fd:
            fd.write(str(params['b_apoptosis']) + ','
                     + str(params['c_necrosis']) + ','
                     + str(params['m']) + ','
                     + str(params['n']) + ','
                     + str(params['alpha']) + ','
                     + str(count_apoptotic) + ','
                     + str(count_necrotic) + "\n")
    return first_line


def compute_cost_remaining_ITMs(params, lattice):
    count_apoptotic, count_necrotic = count_neutrophils(lattice, params)
    cost_remaining_ITMs = params['alpha'] * (params['ITMs'] ** params['k'] - count_necrotic * params['m']
                                              - count_apoptotic * params['n'])
    return cost_remaining_ITMs


def process_lattice(params, index, directory):
    lattice = create_lattice(params['x'], params['y'])
    lattice = fill_lattice(lattice, params, index)
    xy_pairs = generate_random_pairs(params['x'], params['y'], index)
    count_step = 0
    cmap, norm = from_levels_and_colors(
        [params['_necrotic'], params['_apoptotic'], params['_empty'], params['_activated'], 2],
        ['#d62728', '#1f77b4', 'k', '#2ca02c'])
    params['ITMs'] = params['ITMs_list'][index]

    if not os.path.exists(directory + str(params['ITMs_list'][index]) + '/'):
        os.makedirs(directory + str(params['ITMs_list'][index]) + '/')

    for i, j in xy_pairs:
        if lattice[i, j] == params['_activated']:
            if count_step % 100 == 0:
                plt.imshow(lattice, cmap=cmap, norm=norm)
                plt.text(5, 5, 't=' + str(count_step), bbox={'facecolor': 'white', 'pad': 10})
                plt.savefig(directory + str(params['ITMs_list'][index]) + '/' + 'ITM_' + str(params['ITMs_list'][index]) + '_' + str(count_step) + '_lattice.png')
            params = get_payoff_matrix(params)
            all_activated = check_if_all_activated(lattice, i, j, params)

            if all_activated:
                pay_off, strategy = get_global_cost(lattice, params)
            else:
                global_necrotic_cost = get_global_necrotic_cost(lattice, params)
                global_apoptotic_cost = get_global_apoptotic_cost(lattice, params)
                necrotic_local_pay_off = get_local_payoff_necrotic(lattice, i, j, params) - params['mult'] \
                                         * np.exp(global_necrotic_cost)
                apoptotic_local_pay_off = get_local_payoff_apoptotic(lattice, i, j, params) - params['mult'] \
                                          * np.exp(global_apoptotic_cost)

                if params['experiment'] == 'No ITMs':
                    # print (params['experiment'])
                    if ITMs_remaining > 0:
                        if necrotic_local_pay_off >= apoptotic_local_pay_off:
                           strategy = params['_necrotic']
                        else:
                           strategy = params['_apoptotic']
                    else:
                       strategy = params['_activated']
                elif params['experiment'] == 'No Activated':
                    # print(params['experiment'])
                    if necrotic_local_pay_off >= apoptotic_local_pay_off:
                        strategy = params['_necrotic']
                    else:
                        strategy = params['_apoptotic']
            lattice[i, j] = strategy
            set_ITMs_remaining(compute_global_cost(lattice, params))
            count_step += 1

    plt.imshow(lattice, cmap=cmap, norm=norm)
    plt.text(5, 5, 't=' + str(count_step), bbox={'facecolor': 'white', 'pad': 10})
    plt.savefig(directory + str(params['ITMs_list'][index]) + '/' + 'ITM_' + str(params['ITMs_list'][index]) + '_' + str(count_step) + '_lattice.png')
    return lattice


def sweep_params(params, index, directory):
    start = timer()
    print(str(index) + ' started ')

    lattice = process_lattice(params, index, directory)
    apoptotic, necrotic = count_neutrophils(lattice, params)

    results = {'index': index,
               'apoptotic': apoptotic,
               'necrotic': necrotic,
               'necrotic fraction': necrotic/(apoptotic + necrotic)}

    end = timer()
    print(str(index) + ' finished, execution time: ' + str(end - start), ' necrosis:', results['necrotic fraction'])
    return results


def do_parallel(params, directory):
    results = Parallel(n_jobs=1)(delayed(sweep_params)(params,
                                                        index,
                                                       directory)
                                        for index in range(len(params['ITMs_list'])))



if __name__ == "__main__":
    project_dir = 'C:/Users/user/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/Neutrophils ' \
                  'Game/results/Cellular Automata/Global ITMs/Lattice/'
    data_param_space_file = 'C:/Users/user/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/' \
                            'Neutrophils Game/results/Mean Field/Data_Space/data_param_space.csv'
    # data_param_space_file = 'C:/Users/Alva/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/' \
    #                         'Neutrophils Game/results/Data_Space/test_data.csv'

    params = {}

    params['x'] = 50
    params['y'] = 50

    params['k'] = 0.0978

    params['active_neutrophils'] = 2500
    params['ITMs'] = 100.
    params['ITMs_list'] = [0, 1, 10, 100, 500]

    params['neighbors'] = 8

    params['_empty'] = 0
    params['_activated'] = 1
    params['_apoptotic'] = -1
    params['_necrotic'] = -2
    params['mult'] = 8.

    # params['b_apoptosis'] = 1.32E-05
    # params['c_necrosis'] = 0.00024
    # params['m'] = 0.001
    # params['n'] = 0.000271152
    # params['alpha'] = 0.68

    params['b_apoptosis'] = 0.00032828
    params['c_necrosis'] =1.00E-05
    params['m'] = 0.00091
    params['n'] = 0.000181152
    params['alpha'] = 0.98

    params['experiment'] = 'No Activated'


    df = pd.read_csv(data_param_space_file)
    do_parallel(params, project_dir)
