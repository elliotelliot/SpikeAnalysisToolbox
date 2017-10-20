

import numpy as np

# num_attribute_values same for all categories?

"""
Function to create a frequency table based upon responses and categories

Args:
    response_matrix: numpy matrix of responses, shape (num_stimuli, num_neurons)
    categories: numpy matrix of category membership (num_stimuli, num_attributes), contains index of the value for that attribute
    responses: list of possible responses in response_matrix

Returns:
    freq_table: list of numpy arrays indicating frequency of responses for categories/neurons
                shape: (num_attributes, num_attribute_values, num_neurons, num_responses)
                freq_table[attribute_id][attribute_value_id, neuron_id, response_type] 
                = number of trials that had: attribute with id attribute_id had value attribute_value_id and neuron_id 
                responded with response response_type
                freq_table[x][:, :, :] sums up to num_neurons * num_stimuli
                so for each attribute we have a separate freq_table looking at all trials each just divided into that attribute
"""
def frequency_table(response_matrix, categories, responses):
    # First test the dimensions of the provided data
    if (response_matrix.shape[0] != categories.shape[0]):
        raise ValueError("The dimensions of the matrices provided do not match as expected")
    # Now extract the number of neurons, stimuli and categories
    num_stimuli = response_matrix.shape[0]
    num_neurons = response_matrix.shape[1]
    num_categories = categories.shape[1]
    num_responses = len(responses)

    # We must also extract the number of elements in each category
    num_elements_per_category = list()
    for cat_index in range(num_categories):
        num_elements_per_category.append(int(np.max(categories[:, cat_index])) + 1) # number of different attribute values for each attribute

    # Using these dimensions, create a frequency table
    freq_table = list()
    for cat_index in range(num_categories): # iterate threw all attributes
        freq_table.append(np.zeros((num_elements_per_category[cat_index], num_neurons, num_responses)))

    # Populate the freq table:
    for stim_index in range(num_stimuli):
        for cat_index in range(num_categories): # iterate threw attributes
            elem_id = int(categories[stim_index, cat_index]) # get attribute value
            for response_index in range(num_responses):
                freq_table[cat_index][elem_id, :, response_index] += (response_matrix[stim_index, :] == responses[response_index]) # +1 for all neurons that responded like that
                # uses each stimulus response seperatly for each of that stimulus'es attribute.
    return freq_table

"""
Function to convert a frequency table into single cell information analysis

Args:
    freq_table: a list of numpy matrices, shape: num_attributes, [num_attribute_values, num_neurons, num_responses]
    categories: numpy matrix of category membership (num_stimuli, num_attributes)

Returns:
    single_cell_info: A list of numpy arrays of the amount of information per single cell, shape: num_categories, [num_elements_per_category, num_neurons]
"""
def single_cell_information(freq_table, categories):
    num_categories = len(freq_table)
    num_neurons = freq_table[0].shape[1]
    num_responses = freq_table[0].shape[2]
    num_stimuli = categories.shape[0]
    
    single_cell_info = list()
    for cat_index in range(num_categories): # num_attributes
        num_elements_in_category = freq_table[cat_index].shape[0] # num attribute values
        single_cell_info.append(np.zeros((num_elements_in_category, num_neurons)))
        # This division ensures that the number of presentations of each category does not skew the probabilities
        # !!! correcting for unequal stimulus presentation
        freq_table[cat_index] /= np.sum(freq_table[cat_index], axis=2)[:, :, np.newaxis]
    # Single cell information is independently calculated for each neuron:
    # Probability of Response is the same for all categories and can be calculated separately:
    # P_R is of shape (num_neurons, num_responses) (average over num_attribute_values)
    P_R = list()
    for cat_index in range(num_categories):
        P_R.append(np.sum(freq_table[cat_index], axis=0))
        for n in range(num_neurons):
            if (np.sum(P_R[cat_index][n, :]) > 0.0): # when does this ever happen????? <<<<
                P_R[cat_index][n, :] /= np.sum(P_R[cat_index][n, :])

    # Calculating P(R|S) of shape(num_categories, num_elements_in_category, num_neurons, num_responses)
    P_R_S = list()
    for cat_index in range(num_categories):
        num_elements_in_category = freq_table[cat_index].shape[0]
        P_R_S.append(np.copy(freq_table[cat_index]))
        # P_R_S[-1] /= np.sum(P_R_S[-1], axis=2)[:,:,np.newaxis]
        #for elem_index in range(num_elements_in_category):
        #    for n in range(num_neurons):
        #        sum_of_number_responses = np.sum(P_R_S[-1][elem_index, n, :]) 
        #        # Ensuring that we do not divide by zero
        #        if (sum_of_number_responses > 0.0):
        #            P_R_S[-1][elem_index, n, :] /=  sum_of_number_responses 

    ## Calculating P(S) of shape (num_categories, num_elements_in_category)
    #P_S = list()
    #for cat_index in range(num_categories):
    #    num_elements_in_category = freq_table[cat_index].shape[0]
    #    P_S.append(np.zeros((num_elements_in_category)))
    #    for stim_index in range(num_stimuli):
    #        P_S[-1][categories[stim_index, cat_index]] += 1.0
    #    if (np.sum(P_S[-1]) > 0.0):
    #        P_S[-1] /= np.sum(P_S[-1])
    #
    ## Finally calculating P(S|R)
    #P_S_R = list()
    #for cat_index in range(num_categories):
    #    num_elements_in_category = freq_table[cat_index].shape[0]
    #    P_S_R.append(np.zeros((num_elements_in_category, num_neurons, num_responses)))
    #    for elem_index in range(num_elements_in_category):
    #        for n in range(num_neurons):
    #            for r in range(num_responses): 
    #                if (P_R[n, r] > 0.0):
    #                    P_S_R[-1][elem_index, n, r] = ((P_R_S[cat_index][elem_index, n, r]*P_S[cat_index][elem_index]) / (P_R[n, r]))

    # Now eventually calculating the single cell information:
    for cat_index in range(num_categories):
        num_elements_in_category = freq_table[cat_index].shape[0]
        for elem_index in range(num_elements_in_category):
            for n in range(num_neurons):
                for r in range(num_responses):
                    # Ensure no divisions by zero or log of zero
                    if (P_R_S[cat_index][elem_index, n, r] != 0.0):
                        single_cell_info[cat_index][elem_index, n] += P_R_S[cat_index][elem_index, n, r]*np.log2(P_R_S[cat_index][elem_index, n, r] / P_R[cat_index][n, r])
            
    # Finally Return
    return single_cell_info

"""
"""
def naive_bayes_accuracy():
    pass
