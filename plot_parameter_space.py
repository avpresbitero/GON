import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import os


def get_c_greater_b(parameter_space, project_dir):
    df = pd.read_csv(parameter_space)
    df_trimmed = df.loc[df['c_necrosis'] > df['b_apoptosis']]
    df_trimmed.to_csv(project_dir + 'c_greater_b.csv')


def get_c_less_b(parameter_space, project_dir):
    df = pd.read_csv(parameter_space)
    df_trimmed = df.loc[df['c_necrosis'] < df['b_apoptosis']]
    df_trimmed.to_csv(project_dir + 'c_less_b.csv')


def get_c_equal_b(parameter_space, project_dir):
    df = pd.read_csv(parameter_space)
    df_trimmed = df.loc[df['c_necrosis'] == df['b_apoptosis']]
    df_trimmed.to_csv(project_dir + 'c_equal_b.csv')


def get_data_to_plot(parameter_space, project_dir):
    df = pd.read_csv(parameter_space)
    for combination in params['combinations'].keys():
        x = df[params['combinations'][combination][0]].tolist()
        y = df[params['combinations'][combination][1]].tolist()
        z = df[params['combinations'][combination][2]].tolist()
        generate_html_plot(x, y, z, combination, project_dir)


def get_trace_dic(parameter_space, color, trace_dic):
    df = pd.read_csv(parameter_space)
    for combination in params['combinations'].keys():
        x = df[params['combinations'][combination][0]].tolist()
        y = df[params['combinations'][combination][1]].tolist()
        z = df[params['combinations'][combination][2]].tolist()
        trace = get_trace(x, y, z, color)
        if combination not in trace_dic.keys():
            trace_dic[combination] = []
        trace_dic[combination].append(trace)
    return trace_dic


def generate_html_plot(x, y, z, combination, project_dir):
    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=12,
            color=z,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )

    data = [trace1]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='3d-scatter-colorscale')
    plotly.offline.plot(data, filename=project_dir + combination +'.html')


def get_trace(x, y, z, color):
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=12,
            color= color, # set color to an array/list of desired values
            # colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )
    return trace


def generate_html_plots(trace_dic, project_dir):
    for combination in params['combinations'].keys():
        data = trace_dic[combination]
        plotly.offline.plot(data, filename=project_dir + combination +'.html')


def run_c_b_checker(project_dir):
    colors = ['red', 'blue', 'green']
    count = 0
    trace_dic = dict()
    for parameter_space in params['files']:
        color = colors[count]
        trace_dic = get_trace_dic(project_dir + '/' + parameter_space, color, trace_dic)
        count += 1
    generate_html_plots(trace_dic, project_dir)


if __name__ == "__main__":
    project_dir = 'C:/Users/user/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/Neutrophils ' \
                  'Game/results/Mean Field/'
    data_param_space_file = 'C:/Users/user/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/' \
                            'Neutrophils Game/results/Mean Field/c_less_b.csv'
    data_param_space_file = 'C:/Users/user/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/' \
                            'Neutrophils Game/results/Mean Field/c_greater_b.csv'
    data_param_space_file = 'C:/Users/user/Google Drive/Alva Modeling Immune System/Innate Immunity/Journal Papers/' \
                            'Neutrophils Game/results/Mean Field/c_equal_b.csv'

    params = dict()
    params['combinations'] = {
        'cba': ['c_necrosis', 'b_apoptosis', 'alpha'],
        'cma': ['c_necrosis', 'm', 'alpha'],
        'bma': ['b_apoptosis', 'm', 'alpha'],
        'cbm': ['c_necrosis', 'b_apoptosis', 'm'],
    }
    params['explore'] = ['less', 'greater', 'equal']
    params['files'] = ['c_less_b.csv', 'c_greater_b.csv', 'c_equal_b.csv']

    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    # get_data_to_plot(data_param_space_file, project_dir)
    # get_c_greater_b(data_param_space_file, project_dir)
    # get_c_less_b(data_param_space_file, project_dir)
    # get_c_equal_b(data_param_space_file, project_dir)
    run_c_b_checker(project_dir)