# Deep Learning on Visual and Location Data for V2I mmWave Beamforming

In this project, we propose a data fusion approach that takes inputs from visual edge devices and localization sensors to (i) reduce the beam selection overhead by narrowing down the search to a small set containing the best possible beam-pairs and (ii) detect blockage conditions between transmitters and receivers. We evaluate our approach through joint simulation of multi-modal data from vision and localization sensors and RF data. Additionally, we show how deep learning based fusion of images and Global Positioning System (GPS) data can play a key role in configuring vehicle-to-infrastructure (V2I) mmWave links. implement the testbed in the linked paper: https://ieeexplore.ieee.org/document/9751514

## Cite This paper
To use this repository, please refer to our paper: 

 `G. Reus-Muns et al., "Deep Learning on Visual and Location Data for V2I mmWave Beamforming," 2021 17th International Conference on Mobility, Sensing and Networking (MSN), 2021, pp. 559-566, doi: 10.1109/MSN53354.2021.00087.`
 
 ## Instructions to Run the Code: 
To run the fusion framework for visual IoT based beamforming:
1. Visual modality: `python main.py --data_folder path_to_your_train_data_folder --test_data_folder path_to_your_test_data_folder --val_data_folder path_to_your_val_data_folder --input imag` 
2. Coordinate modality: `python main.py --data_folder path_to_your_train_data_folder --test_data_folder path_to_your_test_data_folder --val_data_folder path_to_your_val_data_folder --input coord` 
3. Fusion of Coordinate and Image: `python main.py --data_folder path_to_your_train_data_folder --test_data_folder path_to_your_test_data_folder --val_data_folder path_to_your_val_data_folder --input coord imag` 
