# Preparation of dataset
## Generated samples
The samples used are stored in the directory `/eos/experiment/fcc/ee/generation/DelphesEvents/pre_fall2022_training/IDEA/` .
The events were simulated using Delphes. 
The processes considered are $e^+ e^- \to Z(\to \nu \nu) H(\to aa)$ with $a = u,d,b,c,s,g$.
For the processes $a =  u,d,b,c,s$ samples of $\sim 10^6$ events were produced (i.e. $2 \times 10^6$ jets per sample), for $a = g$ $\sim 2 \times 10^6$ (i.e. $2 \times 10^6$ jets per sample).
Beamspot of 20 um size on Y-axis and 600 um on Z-axis was set.
Tre tree containing the events in the input .root file is called _events_ .


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
Even though the joint action of the two steps, an intermediate file is produced by _Stage1_, which will be saved in the ouptut directory (_OUTDIR_), together with the final ntuples but with a recognizable name.

For what concerns time: 
* _Stage1_ takes $\sim 3-4$ minutes per $10^6$ events (run on 8 cpus);
* _Stage_ntuple_ takes $\sim 5$ minutes per $10^6$ events (run on 1 cpu).

For what concerns memory usage:
* intermediate files weight $\sim 4$ Gb per $10^6$ events; 
* final files weight $\sim 4$ Gb per $10^6$ events;

In our case the directory containing the intermediate and final files weights $\sim 50$ Gb .
We notice that the intermediate files could be deleted after the production of the final ntuples.

In our study five classes are considered: $\{ q = (u,d), b, c, s, g\}$; for each class $10^6$ events were considered, and a $train/test$ split fraction of $9/1$ was used.

All the namespaces used are defined and developed inside the folder `analyzers`.
### Stage1 : `analysis_constituents_stage1_cluster.py`
As said, in this stage basically the initial edm4hep files are read and the interesting features are computed. Furthermore, in our version, the clustering is done explicitly. 
In the initial tree each entry corresponds to an event.
* runs with the support of analyzers, in particular we developed JetConstituentsUtils and ReconstructedParticle2Track
Let's go through the code.
1. explicit clustering. The clustering is done explicitly by the following lines:
```
            #===== CLUSTERING

            #define the RP px, py, pz and e
            .Define("RP_px",          "ReconstructedParticle::get_px(ReconstructedParticles)")
            .Define("RP_py",          "ReconstructedParticle::get_py(ReconstructedParticles)")
            .Define("RP_pz",          "ReconstructedParticle::get_pz(ReconstructedParticles)")
            .Define("RP_e",           "ReconstructedParticle::get_e(ReconstructedParticles)")
            .Define("RP_m",           "ReconstructedParticle::get_mass(ReconstructedParticles)")
            .Define("RP_q",           "ReconstructedParticle::get_charge(ReconstructedParticles)")
            
            #build pseudo jets with the RP
            .Define("pseudo_jets",    "JetClusteringUtils::set_pseudoJets(RP_px, RP_py, RP_pz, RP_e)")
            #run jet clustering with all reconstructed particles. ee_genkt_algorithm, R=1.5, inclusive clustering, E-scheme
            .Define("FCCAnalysesJets_ee_genkt", "JetClustering::clustering_ee_genkt(1.5, 0, 0, 0, 0, -1)(pseudo_jets)")
            #get the jets out of the struct
            .Define("jets_ee_genkt",           "JetClusteringUtils::get_pseudoJets(FCCAnalysesJets_ee_genkt)")
            #get the jets constituents out of the struct
            .Define("jetconstituents_ee_genkt","JetClusteringUtils::get_constituents(FCCAnalysesJets_ee_genkt)")
```
In the initial tree, all the particles measured in one event are saved in one entry of the branch _ReconstructedParticles_ in a                                                                 `ROOT::VecOps::RVec<ReconstructedParticleData>`.
The line `.Define("RP_px",          "ReconstructedParticle::get_px(ReconstructedParticles)")` takes this branch and for all entries computes px of each particle; the output of this call is a branch called _RP_px_ containing an `RVec<float>` per each event.

The jet clustering is performed using the 4-momenta of the reconstructed particles. This operation returns two outputs: 
  - `jets_ee_genkt` : RVec<fastjet::Pseudojet> , `Pseudojet` methods and attributes allow to access the overall jet properties;
  - `jetconstituents_ee_genkt` : RVec< RVec<int> > , i.e.
The fi
  - labeling 
  - output format fastjet
  - association with constituents (build constituents + labels)
  - treatment of constituents (vectors of vectors of RecPartData)
  - Validation of clustering : Plots of residuals

* computation of features: 
  - show good example
  - Validation (Michele comparison)

* output : pointer!!! to vector of vectors of floats (first index -> jet, second index -> constituents of the ith jet)


Alternative: implicit clustering (needs to be validated)

### Stage_ntuple : `MakeNtuple_constituents2.cpp`
* 4 arguments: inpath (path/infilename) outpath N_i N_f
* how to read a per event tree of vector of vectors of floats and translate to per jet tree of arrays (code) (EXAMPLE with code)
* read jets overall properties
* create arrays example (for constituents) + floats (for jet overall properties)
* usage of jet N_i, N_f + stopping when required + cases (see ipad)
* loops events - jet - constituents
* printing the strange cases

### Joint run of Stage1 and Stage_ntuple : `produceTrainingTrees_mp.py`
* explain what it is doing
* explain the multiprocessing choice




Notice: independently of the process which generated it

## How to run this example
 

## What could be improved
* the stage 1 doesn't reduce the statistics if needed, only in stage_ntuple this is done; this implies heavier intermediate files and longer times even when I want to consider small fractions of the initial samples. Stage_ntuple runs only on the required statistics.
* ```...```


# Evaluation
