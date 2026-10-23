import os

import numpy as np

dataset_path_photons = os.environ.get(
    "CALOCHILL_DATASET",
    "./dataset/hdf5_HGx9new/all_interactions_pions_regular_ECAL+HCAL_10-90GeV_{}.slcio.root_with_time.hdf5",
)  # same default as configs/configs_data.yaml
dataset_path_pions = os.environ.get(
    "CALOCHILL_DATASET",
    "./dataset/hdf5_HGx9new/all_interactions_pions_regular_ECAL+HCAL_10-90GeV_{}.slcio.root_with_time.hdf5",
)  # same default as configs/configs_data.yaml

cell_size_ecal = 5
cell_size_hcal = 30
Ymin, Ymin_hcal, Ymax = 0, 30, 78
maximum_time = 200  # ns

Xmin_pions, Xmax_pions = -450, 450
Zmin_pions, Zmax_pions = -450, 450
max_points_per_layer_pions = 413
max_points_pions = 5376
log_mean_pions = -1.8573549
log_var_pions = 1.7792583
log_mean_time_pions = 1.7896006
log_var_time_pions = 2.0470169

Xmin_photons, Xmax_photons = -450, 450
Zmin_photons, Zmax_photons = -450, 450
max_points_per_layer_photons = 413
max_points_photons = 5376
log_mean_photons = -1.8573549
log_var_photons = 1.7792583
log_mean_time_photons = 1.7896006
log_var_time_photons = 2.0470169

x_bins = np.arange(Xmin_pions, Xmax_pions + cell_size_ecal, cell_size_ecal)
z_bins = np.arange(Zmin_pions, Zmax_pions + cell_size_ecal, cell_size_ecal)
x_bins_hcal = np.arange(Xmin_pions, Xmax_pions + cell_size_hcal, cell_size_hcal)
z_bins_hcal = np.arange(Zmin_pions, Zmax_pions + cell_size_hcal, cell_size_hcal)
y_bins = np.arange(Ymin, Ymax + 1, 1)
y_bins_ecal = np.arange(Ymin, Ymin_hcal + 1, 1)
y_bins_hcal = np.arange(Ymin_hcal, Ymax + 1, 1)

f_names = [
    "e_radial",
    "e_radial_ecal",
    "e_radial_hcal",
    "occ_list",
    "occ_list_025",
    "e_sum_list",
    "e_sum_ecal_list",
    "e_sum_hcal_list",
    "hits_list",
    "hits_ecal_list",
    "hits_hcal_list",
    "e_layers_list",
    "e_layers_ecal_list",
    "e_layers_hcal_list",
    "X",
    "Y",
    "Z",
    "numb_active_hits_list",
    "numb_active_hits_ecal",
    "numb_active_hits_hcal",
    "start_layer_list",
    "cog_x",
    "cog_y",
    "cog_z",
]


f_names_r = [
    "e_radial_r",
    "e_radial_ecal_r",
    "e_radial_hcal_r",
    "occ_list_r",
    "occ_list_025_r",
    "e_sum_list_r",
    "e_sum_ecal_list_r",
    "e_sum_hcal_list_r",
    "hits_list_r",
    "hits_ecal_list_r",
    "hits_hcal_list_r",
    "e_layers_list_r",
    "e_layers_ecal_list_r",
    "e_layers_hcal_list_r",
    "X_r",
    "Y_r",
    "Z_r",
    "numb_active_hits_list_r",
    "numb_active_hits_ecal_r",
    "numb_active_hits_hcal_r",
    "start_layer_list_r",
    "cog_x",
    "cog_y",
    "cog_z",
]


time_names = [
    "time_radial",
    "time_radial_ecal",
    "time_radial_hcal",
    "time_sum_list",
    "time_sum_ecal_list",
    "time_sum_hcal_list",
    "time_hits_list",
    "time_hits_ecal_list",
    "time_hits_hcal_list",
    "time_layers_list",
    "numb_active_time_hits_list",
    "numb_active_time_hits_ecal",
    "numb_active_time_hits_hcal",
    "start_layer_time_list",
    "velocity",
    "velocity_per_layer",
    "n_unphysical_hits_per_shw",
    "n_unphysical_hits_per_layer",
]
