# Visual IoT based Beamforming
To run the fusion framework for visual IoT based beamforming:
1. Visual modality: `python main.py --data_folder path_to_your_train_data_folder --test_data_folder path_to_your_test_data_folder --val_data_folder path_to_your_val_data_folder --input imag` 
2. Coordinate modality: `python main.py --data_folder path_to_your_train_data_folder --test_data_folder path_to_your_test_data_folder --val_data_folder path_to_your_val_data_folder --input coord` 
3. Fusion of Coordinate and Image: `python main.py --data_folder path_to_your_train_data_folder --test_data_folder path_to_your_test_data_folder --val_data_folder path_to_your_val_data_folder --input coord imag` 
