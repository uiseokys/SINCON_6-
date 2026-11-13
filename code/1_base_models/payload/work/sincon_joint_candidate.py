import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import torch
from sincon_daily_network import prepare_data
from sincon_district_mobility import station_districts
from sincon_graph_network import adjacency_matrix, graph_features, predict
from sincon_station_state import KEY, WINDOWS
from sincon_tree_covariates import attach_mobility

def rmsle_logs(actual, forecast):
    return float(np.sqrt(np.mean(np.square(actual - np.maximum(forecast, 0)))))

def assemble(reference, neural, mobility, graph, keys):
    tree_blend = 0.5 * (neural + mobility)
    difference = keys[['date', 'station_id']].assign(error=graph - tree_blend)
    day_level = difference.groupby(['date', 'station_id'])['error'].transform('mean').to_numpy()
    return {'ensemble': reference, 'mobility_tree': tree_blend, 'joint': tree_blend + 0.25 * (graph - tree_blend - day_level), 'joint_half_weight': (tree_blend + 0.25 * (graph - tree_blend - day_level) + reference) / 2}

def setup(team_dir, external_zip, scoreboard, directory, address, threads):
    data, archive, daily = prepare_data(team_dir, external_zip, scoreboard)
    from ensemble import FEATURES
    from improved import PARAMS, fit_predict
    PARAMS['num_threads'] = threads
    data, mobility = attach_mobility(data, directory, station_districts(address, team_dir))
    stations = pd.read_csv(team_dir / 'data/metaData.csv').set_index('station_id').sort_index()
    adjacency = adjacency_matrix(stations)
    data, daily = graph_features(data, daily, stations)
    return (data, archive, daily, FEATURES + mobility, stations, adjacency, fit_predict)
