import os
import numpy as np
import pandas as pd
import itertools

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
    xy_pairs = get_moore_neighborhood(params, i, j)
    for x, y in xy_pairs:
        if lattice[x, y] in [params['_activated'], params['_empty']]:
            activated_neighbors += 1
    return activated_neighbors


def check_if_all_activated(lattice, i, j, params):
    activated_neighbors = get_activated_neighbors(lattice, i, j, params)
    xy_pairs = get_moore_neighborhood(params, i, j)
    # print (activated_neighbors, 'LEN',  len(xy_pairs))
    if activated_neighbors == len(xy_pairs):
        return True
    return False


def check_if_all_activated_(lattice, i, j, params):
    activated_neighbors = get_activated_neighbors(lattice, i, j, params)
    if activated_neighbors == params['neighbors']:
        return True
    return False


def get_moore_neighborhood(params, i, j):
    moore_len = (params['moore'] * 2) + 1
    moore_half = int(((params['moore'] * 2) + 1)/2)
    start_x = (i - moore_half) % params['x']
    start_y = (j - moore_half) % params['y']

    moore_neighborhood_pairs = []

    for x in range(moore_len):
        for y in range(moore_len):
            x_in_pair = (start_x + x) % params['x']
            y_in_pair = (start_y + y) % params['y']
            if not (x_in_pair == i and y_in_pair == j):
                moore_neighborhood_pairs.append((x_in_pair, y_in_pair))
    return moore_neighborhood_pairs


def get_local_payoff_necrotic(lattice, i, j, params):
    pay_off = 0.0
    xy_pairs = get_moore_neighborhood(params, i, j)
    # print (i, j, xy_pairs)
    for x, y in xy_pairs:
        if lattice[x, y] == params['_necrotic']:
            pay_off += params['A']
        if lattice[x, y] == params['_apoptotic']:
            pay_off += params['B']
    return pay_off


def get_local_payoff(lattice, i, j, params):
    rolled_lattice = np.roll(lattice, int(params['x']/2) - i, axis=0)
    rolled_lattice = np.roll(rolled_lattice, int(params['y']/2) - j, axis=1)

    moore_len = (params['moore'] * 2) + 1
    moore_half = int(((params['moore'] * 2) + 1) / 2)
    start_x = (int(params['x']/2) - moore_half) % params['x']
    start_y = (int(params['y']/2) - moore_half) % params['y']
    end_x = start_x + moore_len
    end_y = start_y + moore_len
    sub_lattice = rolled_lattice[start_x: end_x, start_y : end_y]
    count_apoptotic, count_necrotic = count_neutrophils(sub_lattice, params)

    necrotic_pay_off = params['A'] * count_necrotic + params['B'] * count_apoptotic
    apoptotic_pay_off = params['C'] * count_necrotic + params['D'] * count_apoptotic
    return apoptotic_pay_off, necrotic_pay_off


def get_local_payoff_apoptotic(lattice, i, j, params):
    pay_off = 0.0
    xy_pairs = get_moore_neighborhood(params, i, j)
    for x, y in xy_pairs:
        if lattice[x, y] == params['_necrotic']:
            pay_off += params['C']
        if lattice[x, y] == params['_apoptotic']:
            pay_off += params['D']
    return pay_off


def get_local_ITMs_cost_all_activated(params, lattice, i, j):
    initial_local_ITMs = get_initial_local_ITMs(params)
    apoptotic_neighbors, necrotic_neigbors, activated_neigbors = count_neighbors(params, lattice, i, j)
    initial_total_local_ITMs = initial_local_ITMs * (((params['moore'] * 2) + 1)**2)
    necrotic_global_local_cost = params['alpha'] * (
                initial_total_local_ITMs ** params['k'] - (necrotic_neigbors + 1) * params['m'] - apoptotic_neighbors * params['n'])
    apoptotic_global_local_cost = params['alpha'] * (
            initial_total_local_ITMs ** params['k'] - necrotic_neigbors * params['m'] - (apoptotic_neighbors + 1) *
            params['n'])

    if necrotic_global_local_cost > apoptotic_global_local_cost:
        return necrotic_global_local_cost, params['_necrotic']
    return apoptotic_global_local_cost, params['_apoptotic']


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
        return global_cost_necrotic #* params['mult']
    return 0.0



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


def count_neighbors(params, lattice, i, j):
    apoptotic_neighbors = 0
    necrotic_neigbors = 0
    activated_neigbors = 0
    
    xy_pairs = get_moore_neighborhood(params, i, j)

    for x, y in xy_pairs:
        if lattice[x, y] == params['_apoptotic']:
            apoptotic_neighbors += 1
        if lattice[x, y] == params['_necrotic']:
            necrotic_neigbors += 1
        if lattice[x, y] == params['_activated']:
            activated_neigbors += 1
    return apoptotic_neighbors, necrotic_neigbors, activated_neigbors


def get_initial_local_ITMs(params):
    return params['ITMs']/(params['x']*params['y'])


def compute_local_b_necrosis(lattice, params, i, j):
    initial_local_ITMs = get_initial_local_ITMs(params)
    count_apoptotic, count_necrotic, count_activated = count_neighbors(params, lattice, i, j)
    initial_total_local_ITMs = initial_local_ITMs * (len(get_moore_neighborhood(params, i, j))+1)
    global_local_cost = params['alpha'] * (initial_total_local_ITMs ** params['k'] - count_necrotic * params['m'] - count_apoptotic * params['n'])
    if global_local_cost >= 0:
        params['b_necrosis'] = global_local_cost
    else:
        params['b_necrosis'] = 0
    return params


def compute_cost_remaining_ITMs(lattice, params, i, j):
    initial_local_ITMs = get_initial_local_ITMs(params)
    count_apoptotic, count_necrotic, count_activated = count_neighbors(params, lattice, i, j)
    initial_total_local_ITMs = initial_local_ITMs * (len(get_moore_neighborhood(params, i, j))+1)
    global_local_cost_apoptosis = params['alpha'] * (initial_total_local_ITMs ** params['k'] - count_necrotic * params['m'] - (count_apoptotic +1) * params['n'])
    global_local_cost_necrosis = params['alpha'] * (
                initial_total_local_ITMs ** params['k'] - (count_necrotic + 1) * params['m'] - count_apoptotic * params['n'])
    return global_local_cost_apoptosis, global_local_cost_necrosis



def get_payoff_matrix(params):
    params['A'] = - params['c_necrosis'] + 2.0 * params['b_necrosis']
    params['B'] = - params['c_necrosis'] + params['b_apoptosis'] + params['b_necrosis']
    params['C'] = params['b_apoptosis'] + params['b_necrosis']
    params['D'] = 2.0 * params['b_apoptosis']
    return params


def get_payoff_matrix_(params):
    params['A'] = - params['c_necrosis']
    params['B'] = - params['c_necrosis'] + params['b_apoptosis']
    params['C'] = params['b_apoptosis']
    params['D'] = 2.0 * params['b_apoptosis']
    return params


def process_lattice(params, index):
    lattice = create_lattice(params['x'], params['y'])
    lattice = fill_lattice(lattice, params, index)
    xy_pairs = generate_random_pairs(params['x'], params['y'], index)

    for i, j in xy_pairs:
        if lattice[i,j] == params['_activated']:
            params = compute_local_b_necrosis(lattice, params, i, j)
            params = get_payoff_matrix(params)
            all_activated = check_if_all_activated(lattice, i, j, params)
            if all_activated:
                pay_off, strategy = get_local_ITMs_cost_all_activated(params, lattice, i, j)
            else:
                len_moore = ((params['moore'] * 2) + 1) ** 2
                apoptotic_local_pay_off, necrotic_local_pay_off = get_local_payoff(lattice, i, j, params)
                apoptotic_local_pay_off = apoptotic_local_pay_off/len_moore
                necrotic_local_pay_off = necrotic_local_pay_off/len_moore
                if params['experiment'] == 'No ITMs':
                    if params['b_necrosis'] > 0:
                        if necrotic_local_pay_off >= apoptotic_local_pay_off:
                           strategy = params['_necrotic']
                        else:
                           strategy = params['_apoptotic']
                    else:
                       strategy = params['_activated']
                elif params['experiment'] == 'No Activated':
                    if necrotic_local_pay_off >= apoptotic_local_pay_off:
                        strategy = params['_necrotic']
                    else:
                        strategy = params['_apoptotic']
            lattice[i, j] = strategy
    return lattice


def sweep_params(params, index, b_apoptosis, c_necrosis, m, n, alpha):
    start = timer()
    print(str(index) + ' started ')

    params['b_apoptosis'] = b_apoptosis
    params['c_necrosis'] = c_necrosis
    params['m'] = m
    params['n'] = n
    params['alpha'] = alpha

    lattice = process_lattice(params, index)
    apoptotic, necrotic = count_neutrophils(lattice, params)

    results = {'index': index,
               'apoptotic': apoptotic,
               'necrotic': necrotic,
               'necrotic fraction': necrotic/(apoptotic + necrotic)}

    end = timer()
    print(str(index) + ' finished, execution time: ' + str(end - start), ' necrosis:', results['necrotic fraction'])
    return results


def do_parallel(df, params):
    results = Parallel(n_jobs=1)(delayed(sweep_params)(params,
                                                        index,
                                                        row['b_apoptosis'],
                                                        row['c_necrosis'],
                                                        row['m'],
                                                        row['n'],
                                                        row['alpha'])
                                        for index, row in df.iterrows() if index % 30 == 0)

    return results


if __name__ == "__main__":
    project_dir = 'C:/Users/user/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/Neutrophils ' \
                  'Game/results/Cellular Automata/Local ITMs/Necrosis Payoff/'
    data_param_space_file = 'C:/Users/user/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/' \
                            'Neutrophils Game/results/Mean Field/Data_Space/data_param_space.csv'

    params = {}

    params['x'] = 50
    params['y'] = 50

    params['k'] = 0.0978

    params['active_neutrophils'] = 2500
    params['ITMs'] = 500.

    params['_empty'] = 0
    params['_activated'] = 1
    params['_apoptotic'] = -1
    params['_necrotic'] = -2
    params['moore'] = 4
    params['neighbors'] = 8
    params['mult'] = 8.

    params['experiment'] = 'No Activated'

    ITMs_remaining = params['ITMs']

    df = pd.read_csv(data_param_space_file)
    results = do_parallel(df, params)
    res_df = pd.DataFrame(results)
    res_df = res_df.set_index('index')

    if not os.path.exists(project_dir + str(params['moore']) + '/' + params['experiment']):
        os.makedirs(project_dir + str(params['moore']) + '/' + params['experiment'])

    res_df = df.join(res_df)
    Date_Now = pd.datetime.now().strftime("%Y-%m-%d_%H-%M")

    res_df.to_csv(project_dir + str(params['moore']) + '/' + params['experiment'] + '/' + Date_Now + '_res_'
                  + str(int(params['ITMs'])) + '.csv')