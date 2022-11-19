# GNAEMDA
##
GNAEMDA: microbe-drug associations prediction on
graph normalized convolutional network

## Dataset
1)MDAD: including 2470 clinically or experimentally verified microbe-drug associations, between 1373 drugs and 173 microbes;

2)aBiofilm: including resource of anti-biofilm agents and their potential implications in antibiotic drug resistance;

3)DrugVirust: including the activity and development of related compounds of a variety of human viruses;

## Data description
- adj: interaction pairs between microbes and drugs.
- drugs: IDs and names for drugs.
- microbes/viruses: IDs and names for microbes/viruses.
- drug_features:  Drug network topological attribute.
- drug_similarity: Drug integrated similarity attribute.
- microbe_features: Microbe genome sequence attribut.
- microbe_similarity: Microbe functional similarity attribute.

## Dependencies
Recent versions of the following packages for Python 3 are required:

- Anaconda3
- Python 3.8.0
- Pytorch 1.8.1
- torch_geometric 1.7.0
- torch_scatter 2.0.6

## Run
Default examples and parameters have been set

run main.py on MDAD

GNAEMDA: epochs = 400, lr = 0.005, s = 1.6

VGNAEMDA: epochs = 3000, lr = 0.005, s = 1.6