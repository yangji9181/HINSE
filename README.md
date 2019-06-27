# HINSE
Meta-Graph Based HIN Spectral Embedding

This github repository is under construction.

## Data Prepreation
For example, we use DBLP dataset. The data is processed using https://github.com/macks22/dblp . Please put the processed data under the data folder.

## Data Preprocessing
Filter DBLP based on Vocab-Label author list
```
python dblp_data.py
```

## Metagraph Instance Matching
Match the metapath/metagraph instances from HIN. One need to specify the dataset flag. You need to install ```wine``` before running the commend!!!
```
python3 submodule.py dblp
```

## Eigenlist Generating

Calculated eigenvalues and plot the final eigenvalue plots using python. The 2nd argument is the dataset name, and the 3rd and 4th arguments are
```
python3 eigen_cal_all.py dblp
```

## Autoencoding Data

Compress the original embedding using the autoencoder. The parameters could be changed inside the autoencoder_dblp.py script.

```
python3.5 -u autoencoder_dblp.py --dataset dblp
```
## Some important files for modification

1. dblp.q : This is the selected metagraphs for SubMatch Program, please reference SubMatch about how to write this kind of *.q file.
2. metapath_dblp.txt : This is the text version of the dblp.q. Their order should be the same.
