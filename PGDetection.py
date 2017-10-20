import numpy as np
from numba import jit


"""
Extraction of two-neuron spike patterns based upon the synapses between them

Args:
    ids: An numpy vector of neuron ids
    times: A numpy vector of the times at which the accompanying ids fired
    pre: the presynaptic neuron ids of the synapse group in which we are interested
    post: the postsynaptic neuron ids of chosen synapse group
    delays: the number of timesteps of delay for the corresponding synapses
    margin: the margin (in seconds) for a pre-post spike pair (as a float)
    timestep: the timestep at which the simulation was run (as a float)

Returns:
    PG_ids: The synapse ids for spike pairs that were found
    PG_times_pre: The times at which the presynaptic neuron of the corresponding pairs fired.
    PG_times_post: The times at which the presynaptic neuron of the corresponding pairs fired.
"""
@jit(nopython=True)
def synapse_wise(ids, times, pre, post, delays, margin, timestep):
    # First getting the number of synapses that we must consider:
    num_synapses = len(pre)
    syn_PG_occurrences = np.zeros((num_synapses))
    #syn_PG_times = [[] for n in range(num_synapses)]

    for syn_id in range(num_synapses):
        #print(syn_id)
        for preloc in range(len(ids)):
            preidx = ids[preloc]
            if (preidx == pre[syn_id]):
                # Find post synapses
                for postloc in range(len(ids)):
                    postidx = ids[postloc]
                    if (np.abs((times[preloc] + delays[syn_id]*timestep) - times[postloc]) < margin):
                        syn_PG_occurrences[syn_id] += 1
                        #syn_PG_times.append(times[preloc])
        ## Get the pre spikes
        #loc_pre_spikes = np.where(ids == pre[syn_id])[0]
        #num_pre_spikes = len(loc_pre_spikes)

        ## Check how often the post neuron fired
        #loc_post_spikes = np.where(ids == post[syn_id])[0]
        #num_post_spikes = len(loc_post_spikes)
        #
        ## For each pre-spike, check if there is a PG
        #for pre_spike_id in loc_pre_spikes:
        #    pre_spike_time = times[pre_spike_id]

        #    # Check if any of the post spikes are during margin
        #    post_mask = np.abs((pre_spike_time + delays[syn_id]*timestep) - times[loc_post_spikes]) < margin

        #    for PG in np.sum(post_mask):
        #        syn_PG_occurrences[syn_id] += 1
        #        syn_PG_times[syn_id].append(pre_spike_time)

    return(syn_PG_occurrences)
    #return (syn_PG_occurrences, syn_PG_times)
#
#    
#    
#    
#    # We must also only consider ids which are present in the list of pre/post
#    for idx in idset:
#        if ((np.sum(pre == idx) == 0) or (np.sum(post == idx) == 0)):
#            idset.remove(idx)
#
#    # Now our list of ids (in idset) are the only ones we should be interested in.
#    # For each of the spikes we have been given
#    for idval in ids:
#        # Check if this id is one that we are interested in:
#        if idval not in idset:
#            continue
#        # If the id is appropriate, get the list of synapses for which it is pre
#        premask = [pre == idval]
#        potentialsynapses = np.where(premask)
#        # Running through each possible synapse, check the post synapse id
#        for synid in potentialsynapses:
#            if post[synid] not in idset:
#                continue
#        
#    # For each spike time/id pair, check if the id is relevant
#    # If the id matches, find synapses with this pre-id and get a list of postids and delays
#    # Check if any future post-synaptic neuron spikes exist in the correct delay +- margin timescale
#    # If yes, store synaptic id, pre and post spike times
        
