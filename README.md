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

4. Run the scripts in the order they have been numbered. The results are stored in the outputs-folder. Before running the scripts, make sure you have picked a correct model at the beginning of the script. The code currently supports the models "alexnet" and "resnet50". The scripts for computing the RDMs, RSA and plotting support an interactive choice of either a pre-trained or a fine-tuned model.

## Overview of the scripts:

- extract_features: Extracts the features from each layer of the prespecified pre-trained model. The features are saved as numpy arrays to outputs/clean_baseline/features/MODEL_NAME.

- extract_finetuned_features: Extracts the features from each layer of the fine-tuned model. The features are saved as numpy arrays to outputs/finetuned_behaviour/features/alexnet. Currently the finetuning is supported only for Alexnet.

- compute_rdms: Computes RDMs for either the pre-trained or the fine-tuned model, according to the user prompt. The results are saved to outputs/PIPELINE/rdms/MODEL_NAME.

- compute_rsa: Computes RSA using either the pre-trained or the fine-tuned model, according to the user prompt. The RSA is computed between the model layer RDMs and MEG RDMs, as well as between the model layer RDMs and the three behavioral RDMs (i.e. visual appearance, spatial structure and semantic content RDMs). The results are saved to outputs/PIPELINE/rsa/MODEL_NAME.

- plot_results: Plots the RSA results between the prespecified model and MEG RDMs (time series of Spearman correlations for each layer separately as well as the peak latencies for each layer). Additionally, plots the RSA results between the prespecified model and behavioral RDMs.

- extract_behavioral_labels: reads the behavioral labels for each scene, computes the consensus labels and stores them into consensus_labels.csv.

- train_behaviour_model: constructs training and validation data sets from scene images and the corresponding behavioral labels, and fine-tunes the pre-trained model with the data. Currently only supports Alexnet.

- check_dims: a helper function to check dimensions of a prespecified numpy array and confirm the dimensions make sense.

- make_meg_images_good: constructs a test image set from the Anderson images to get test images of the same brightness than the training images. 

- visualize_rdms: an additional visualization script for plotting the RDMs.