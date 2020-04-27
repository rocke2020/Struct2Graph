# Struct2Graph
A PyTorch implementation of GCN with mutual attention for protein-protein interaction prediction

![Proposed GCN architecture with Mutual Attention Mechanism](approach.png)

Struct2Graph uses two graph  convolutional networks (GCNs) with weight sharing, and a mutual attention network to extract relevant geometric features related toquery protein pairs. The details of the GCN architecture is described in our paper (currently under review).

A dataset of 11160 interactions (5580 positive pairs, 5580 negative pairs) consisting of 4024 proteins was downloaded from UniProt Acc. and IntAct.

### Requirements
* PyTorch
* scikit-learn

### Usage
We provide two notebook files, one each for the multi-class GCN classifier and the multi-class RF classifier. The notebooks are self-sufficient and various relevant details have been marked in the files themselves.

### Contact
Contact: <a href="https://web.eecs.umich.edu/~mayankb/">Mayank Baranwal, Postdoctoral Fellow, University of Michigan at Ann Arbor</a>

### Acknowledgements
The proposed GCN architecture comprising of mutual attention mechanism for interaction prediction between query proteins is built upon our prior work on using GCNs for metabolic pathway prediction [1].

### References
    1. Baranwal, Mayank, Abram Magner, Paolo Elvati, Jacob Saldinger, Angela Violi, and Alfred Hero. "A deep learning architecture for metabolic pathway prediction." Bioinformatics (2019)
