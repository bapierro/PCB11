# Can behaviour‑trained ANNs reveal the brain’s temporal hierarchy in scene processing?

This project is about studying whether behaviour-trained ANNs learn feature representations that more closely reflect the temporal hierarchy of brain's scene processing than the feature representations learned by pre-trained ANN models. SYNS scene images and corresponding behavioral labels derived from [a study by Anderson et al.](https://doi.org/10.1167/jov.21.2.8) are used in fine-tuning the pretrained ANNs. The feature representations of each layer of the ANN models are compared with the feature representations from MEG data using representational similarity analysis. 

## Running the code

1. Download the following data from https://osf.io/jp26k/overview into the folder data/meg:

- MEGRDMs_2D.mat (a 4D matrix of size 36 × 36 × 1201 × 20, corresponding to the 36 test images, 1201 time points and 20 participants)
- three model RDMs derived from the behavioral labels of each label category:
    - semanticRDM_sq.mat
    - structureRDM_sq.mat
    - visualRDM_sq.mat
- time.mat (a time vector)

2. Download the extracted training data into data/scenes/syns_anderson_full. The folder data/scenes also contains the testing scene images, named syns_meg36. 

3. Run the script make_meg_images_good.py to get brighter test image set. These are stored into data/scenes/syns_meg36_real.

4. Run the scripts in the order they have been numbered. The results are stored in the outputs-folder. Before running the scripts, make sure you have picked a correct model at the beginning of the script. The code currently supports the models "alexnet" and "resnet50". The visualize_rdms.py is an additional visualization script that can be run after running the scripts for extracting features and computing RDMs.

