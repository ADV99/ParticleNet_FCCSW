# Preparation of dataset
## Generated samples
The samples used are stored in the directory `/eos/experiment/fcc/ee/generation/DelphesEvents/pre_fall2022_training/IDEA/` .
The events were simulated using Delphes. 
The processes considered are $e^+ e^- \to Z(\to \nu \nu) H(\to aa)$ with $a = u,d,b,c,s,g$.
For each process a sample of $\sim 10^6$ events was produced (i.e. $2 \times 10^6$ jets per sample).
Beamspot of 20 um size on Y-axis and 600 um on Z-axis was set.
Five classes are considered: $\{ q = (u,d), b, c, s, g\}$.

## Description
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
Since we are interested in the final ntuple, these two codes are executed omptimally by `produceTrainingTrees_mp.py` through the usage of multiprocessing.
So the production of the training dataset from the generated samples is performed in two steps, which in the folloing will be referred to as Stage1 and Stage_ntuple.
Even though the joint action of the two steps, an intermediate file is produced by Stage1, which will be saved in the ouptut directory (OUTDIR) with a recognizable name. 

### Stage1


### Stage_ntuple







Notice: independently of the process which generated it

## How to run this example

# Evaluation

``` aaa ``` 
