import numpy as np
import InformationAnalysis as calc_info

def make_freq_table(filtered_rates, info_for_analysis):
    num_layers = info_for_analysis[0]
    num_exci = info_for_analysis[1]
    num_inhi = info_for_analysis[2]
    num_stimuli = info_for_analysis[3]

    num_categories = 4
    category_matrix = np.zeros((num_stimuli, num_categories))
    response_matrix = np.asarray(filtered_rates)

    category_matrix[0,:] = np.array([0, 1, 0, 1])
    category_matrix[1,:] = np.array([0, 1, 1, 1])
    category_matrix[2,:] = np.array([0, 1, 0, 0])
    category_matrix[3,:] = np.array([0, 1, 1, 0])
    category_matrix[4,:] = np.array([1, 1, 0, 1])
    category_matrix[5,:] = np.array([1, 1, 1, 1])
    category_matrix[6,:] = np.array([1, 1, 0, 0])
    category_matrix[7,:] = np.array([1, 1, 1, 0])
    category_matrix[8,:] = np.array([0, 0, 0, 1])
    category_matrix[9,:] = np.array([0, 0, 1, 1])
    category_matrix[10,:] = np.array([0, 0, 0, 0])
    category_matrix[11,:] = np.array([0, 0, 1, 0])
    category_matrix[12,:] = np.array([1, 0, 0, 1])
    category_matrix[13,:] = np.array([1, 0, 1, 1])
    category_matrix[14,:] = np.array([1, 0, 0, 0])
    category_matrix[15,:] = np.array([1, 0, 1, 0])
    
    return response_matrix, category_matrix


def make_freq_table_layers(filtered_rates_all, info_network_size):    
    freq_table_layer = list()
    for layer in range(len(filtered_rates_all)):
        response_matrix_tmp, category_matrix_tmp = make_freq_table(filtered_rates_all[layer], info_network_size)
        freq_table_tmp = calc_info.frequency_table(response_matrix_tmp, category_matrix_tmp, [-1, 0, 1])

        freq_table_layer.append(freq_table_tmp)
    return freq_table_layer

def make_information_layers(freq_table_layer, filtered_rates_all, info_network_size):
    information_all_layers = list()
    for layer in range(len(freq_table_layer)):
        _ , category_matrix = make_freq_table(filtered_rates_all[layer], info_network_size)
        information_tmp = calc_info.single_cell_information(freq_table_layer[layer], category_matrix)
        information_all_layers.append(information_tmp)
    
    return information_all_layers



    