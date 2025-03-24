## Table of Contents
- [Overview](#overview)
- [Aim of the Project](#aim-of-the-project)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation & How to Run](#installation--how-to-run)
- [Demo](#demo)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Overview
League of Legends, a popular multiplayer online battle arena (MOBA) game, generates extensive data from matches, providing an excellent opportunity to apply machine learning techniques to real-world data.

## Aim of the Project
The aim of this project is to develop a logistic regression model that can accurately predict the outcomes of League of Legends matches. By leveraging various in-game statistics, I aim to provide valuable insights and predictions.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** scikit-learn, pandas, numpy
- **Tools:** Jupyter Notebook, Git, VS Code

## Prerequisites
- Python (version 3.6 or higher)
- Git
- Jupyter Notebook

## Installation & How to Run
To set up and run the project locally:

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/ImaadhRenosh/League-of-Legends-Match-Predictor.git
    cd League-of-Legends-Match-Predictor
    ```

2. **Install Dependencies**: Install the required libraries using:
    ```
    import os

    # Change to a stable directory
    os.chdir('/home')  # Adjust based on your environment (e.g., '/root' or '/home/jovyan' in some cloud environments)

    # Install required libraries
    !pip install pandas --user
    !pip install scikit-learn --user
    !pip install torch --user
    !pip install matplotlib --user

    # Verify installations
    import pandas as pd
    import sklearn
    import torch
    import matplotlib
    print("Libraries installed successfully!")
    print("pandas version:", pd.__version__)
    print("scikit-learn version:", sklearn.__version__)
    print("torch version:", torch.__version__)
    print("matplotlib version:", matplotlib.__version__)

    # Import Required Libraries
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler 
    from torch.utils.data import DataLoader, TensorDataset
    ```

4. **Run the Jupyter Notebook**: Launch Jupyter Notebook and open `match_predictor.ipynb` to train and evaluate the model.

## Demo
<img width="1024" alt="Screenshot 2025-03-24 at 05 57 30" src="https://github.com/user-attachments/assets/6bb8147c-2c46-4c14-a9f9-d0a46d3a3520" />

### Task 1: Load the League of Legends dataset and preprocess it for training

<img width="1005" alt="Screenshot 2025-03-24 at 05 58 33" src="https://github.com/user-attachments/assets/a483e73a-1618-4a02-976a-c0ad4c50311b" />

<img width="777" alt="Screenshot 2025-03-24 at 05 59 26" src="https://github.com/user-attachments/assets/df41475a-943c-4001-a9a6-af0e46e4a9e6" />

### Task 2: Implement a logistic regression model using PyTorch

<img width="1009" alt="Screenshot 2025-03-24 at 06 06 38" src="https://github.com/user-attachments/assets/a8d5ea0a-f73a-4cce-b6e1-dbcd088e9f5b" />

<img width="534" alt="Screenshot 2025-03-24 at 06 10 38" src="https://github.com/user-attachments/assets/9ba45d85-2d11-467d-8b9a-81e035cd8aa9" />

Library versions:

<img width="721" alt="Screenshot 2025-03-24 at 06 08 41" src="https://github.com/user-attachments/assets/5b15e6f9-d21b-47af-abcd-27e827a75da2" />

<img width="267" alt="Screenshot 2025-03-24 at 06 00 44" src="https://github.com/user-attachments/assets/554bdf5c-a993-4308-a568-4870abe535d2" />



### Task 3: Train the logistic regression model on the dataset
- Implement the train loop for a specified number of epochs.
- Make predictions and calculate the loss.
- Perform backpropagation and update the model parameters.
- Evaluate the model and print accuracy on training and testing sets.

### Task 4: Implement optimization techniques and evaluate the model's performance
- Implement L2 regularization in the optimizer (weight_decay parameter).
- Retrain the model with the same training loop.
- Evaluate the optimized model on training and testing sets.

### Task 5: Visualize the model's performance and interpret the results
- Import necessary libraries (matplotlib.pyplot, confusion_matrix, classification_report, roc_curve, auc).
- Generate and plot the confusion matrix.
- Print the classification report.
- Plot the ROC curve and calculate the AUC.

### Task 6: Save and load the trained model
- Use torch.save to save the model's state dictionary.
- Use torch.load to load the state dictionary into a new model instance.
- Set the loaded model to evaluation mode.
- Evaluate the loaded model and ensure consistent performance.

### Task 7: Perform hyperparameter tuning to find the best learning rate
- Define a list of learning rates to test.
- Reinitialize the model and optimizer for each learning rate.
- Train and evaluate the model for each learning rate.
- Print the best learning rate and corresponding test accuracy.

### Task 8: Evaluate feature importance to understand the impact of each feature on the prediction
- Extract the weights from the linear layer.
- Create a DataFrame with feature names and their corresponding importance.
- Sort the DataFrame by importance.
- Plot the feature importance using a bar plot.

## Usage
After launching the Jupyter Notebook, you can:
- Import match data.
- Train the logistic regression model.
- Predict the outcomes of new matches.

## Contributing
Contributions are welcome! If you’d like to improve this project, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push the branch.
4. Open a pull request detailing your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
- Thanks to the contributors of scikit-learn, pandas, and numpy.
- Special thanks to the League of Legends community for their support and data.

## Contact
For any questions or suggestions, feel free to reach out:
- **Email:** imaadhrenosh@gmail.com
- **LinkedIn profile**: [LinkedIn profile](https://www.linkedin.com/in/imaadh-renosh-007aba348)
