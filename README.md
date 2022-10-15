# Intro: Training Chain of ParticleNet in FCCSW
The contents of this repository aim to furnish an example of how to set the training of ParticleNet inside FCCSW framework.

The implementation of the training chain of ParticleNet jet tagger inside FCCSW framework is presented here. 
The training chain consists of 5 stages:
1. generation of events;
2. reading of the samples and preparation of dataset for training;
3. training;
4. evaluation.

Here skip the first step, assuming the samples have already been generated and are available.
