# SpikeAnalysisToolbox
A Toolbox for analysing spiking neural networks. Primarily for internal use in the Computational Neuroscience Lab at Uni of Oxford (oftnai.org).

## Network Information Dictionaries
Lots of functions need information about either the network or the timing of stimuli. This is given in dictionaries with fields named as shown bellow.
It would be good to refactor this.

Whenever a function has a parameter network_info or network_architecture
```python
network_architecture = dict(
    num_exc_neurons_per_layer = 64*64,
    num_inh_neurons_per_layer = 32*32,
    num_layers = 4,
)
```

Time information for the stimuli.
```python
info_times = dict(
    length_of_stimulus = < time in seconds for each stimulus presentation, e.g. 2.0 >,
    num_stimuli = < number of stimulus presentations, if a stimulus is presented 10 times it counts as 10 presentations > ,
    time_start = < start time after which to calculate the firing rates, e.g. 1.5 >,
    time_end = < end time. The firing rates are estimated in this window since the network needs some time to settle >
)
```

## Information Scores
### Based on Information Theory
There is a range of different information scores implemented in `information_scores.py`. These require labeling to which object a stimulus presentation belongs.
An object for example is "Left Edge anywhere on the retina" and this object is present in a number of stimulus presentations.

All information theory based scores require the object labeling to occur in the following form:
The functions usually have a parameter `objects`. This is a nested list of integers. e.g.:
`[[1, 3, 5, 7], [0, 2, 4, 6]]`
=> There are 2 objects. The first one is present in stimulus presentations 1, 3, 5 and 7. The second object is present in stimulus presentations 0, 2, 4 and 6.


### Decoder Performance
`class SingleCellDecoder`
The above approach only works if a stimulus does not contain multiple objects. If it does there is this decoder approach:
For each cell we find a threshold and call the cell 'ON' if the firing rate is above the threshold and 'OFF' if it is bellow.
Then we measure how many stimulus presentations are labeled correctly by the cell.

This requires a label matrix that is a numpy array of shape [n_objects, n_stimulus_presentations]

## Loading Label from Files
I assume a certain naming convention for my stimuli:
A stimulus has for example the name `1wcl` where each position in this 4 character word corresponds to an attribute of the
stimulus. In this case: location=1, color=white, shape=circle, position_relative_to_location=left.

If you use this naming convention you can load label structures as described above with bash style wildcards using functions from `data_loading.py`
`load_testing_stimuli_indices_from_wildcarts(path_to_file_list, ['1***', '2***'])`
Will return a list of 2 dictionaries, where the first one contains in it's field 'indices' all stimulus prestation indices where something was shown at location 1 with the other three attributes being arbitrary.

This requires a textfile with the name 'testing_list.txt' to be located in the folder 'path_to_file_list/'

My version of the LabVisonIntro module copies this automatically into the experiments output folder: https://github.com/rauwuckl/LabVisionIntro