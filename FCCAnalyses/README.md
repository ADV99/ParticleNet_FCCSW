# Preparation of dataset
## Generated samples
The samples used are stored in the directory `/eos/experiment/fcc/ee/generation/DelphesEvents/pre_fall2022_training/IDEA/` .
The events were simulated using Delphes. 
The processes considered are $e^+ e^- \to Z(\to \nu \nu) H(\to aa)$ with $a = u,d,b,c,s,g$.
For the processes $a =  u,d,b,c,s$ samples of $\sim 10^6$ events were produced (i.e. $2 \times 10^6$ jets per sample), for $a = g$ $\sim 2 \times 10^6$ (i.e. $2 \times 10^6$ jets per sample).
Beamspot of 20 um size on Y-axis and 600 um on Z-axis was set.


## Description and usage
During the training we want ParticleNet to learn to identify a jet from its properties. This means that each entry of the training dataset should contain the properties of one jet (and of its constituents) which are significant for the discrimination only; furthermore, these properties should be organized in a format accessible to ParticleNet: *arrays*.

However, the samples generated
* have a per-event structure, 
* each event contains way more information than the needed one for the training,
* each event is saved in edm4hep format 

So, before performing the training three actions are required:
1. read the generated samples in edm4hep format(through fccanalysis),
2. extract/compute the features of interest (through fccanalysis),
3. produce the ntuples (one per class) containing the interesting features.

In our case, the first two actions are executed by `analysis_constituents_stage1_cluster.py` and the third by `MakeNtuple_constituents2.cpp` .
Since we are interested in the final ntuple, these two codes are executed jointly by `produceTrainingTrees_mp.py`, optimizing the times through the usage of multiprocessing.
So the production of the training dataset from the generated samples is performed in two steps, which in the folloing will be referred to as _Stage1_ and _Stage_ntuple_.
Even though the joint action of the two steps, an intermediate file is produced by _Stage1_, which will be saved in the ouptut directory (_OUTDIR_) with a recognizable name.

For what concerns time: 
* _Stage1_ takes $\sim 3-4$ minutes per $10^6$ events (run on 8 cpus);
* _Stage_ntuple_ takes $\sim 5$ minutes per $10^6$ events (run on 1 cpu).

For what concerns memory usage:
* intermediate files weight $\sim 4$ Gb per $10^6$ events; 
* final files weight $\sim 4$ Gb per $10^6$ events;

In our case the directory containing the intermediate and final files weights $\sim 50$ Gb .
We notice that the intermediate files could be deleted after the production of the final ntuples.

In our study five classes are considered: $\{ q = (u,d), b, c, s, g\}$; for each class 5Millione

### Stage1


### Stage_ntuple







Notice: independently of the process which generated it

## How to run this example

# Evaluation
 

# What could be improved
* the stage 1 doesn't reduce the statistics if needed, only in stage_ntuple this is done; this implies heavier intermediate files and longer times even when I want to consider small fractions of the initial samples. Stage_ntuple runs only on the required statistics.
*  ... ```...```
