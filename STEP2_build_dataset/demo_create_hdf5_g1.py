from dataset_utils_g1 import *

metadata_root = '/media/wsw/SSD1T1/data/metadata'

action_gap = 1
num_points_to_sample = 10000

output_hdf5_file = '/media/wsw/SSD1T1/data/g1_{}actiongap_{}points.hdf5'.format(action_gap, num_points_to_sample)
# output_hdf5_file = '[PATH_TO_SAVE_FOLDER]/hand_packaging_wild_1-20_{}actiongap_{}points.hdf5'.format(action_gap, num_points_to_sample)

# process_hdf5(output_hdf5_file, dataset_folders, action_gap, num_points_to_sample)
process_hdf5(output_hdf5_file, metadata_root, action_gap, num_points_to_sample)