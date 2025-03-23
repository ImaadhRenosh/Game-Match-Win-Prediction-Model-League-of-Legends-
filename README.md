## Table of Contents
- [Overview](#overview)
- [Aim of the Project](#aim-of-the-project)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation & How to Run](#installation--how-to-run)
- [Screenshots / Demo](#screenshots--demo)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Overview
League of Legends, a popular multiplayer online battle arena (MOBA) game, generates extensive data from matches, providing an excellent opportunity to apply machine learning techniques to real-world scenarios. In this project, I will build a logistic regression model aimed at predicting the outcomes of League of Legends matches.

## Aim of the Project
The aim of this project is to develop a logistic regression model that can accurately predict the outcomes of League of Legends matches.

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

## Screenshots / Demo
- **Screenshot 1/ Data Import**:
  ![Data Import](path/to/screenshot1.png)

- **Screenshot 2/ Data Preprocessing**:
  ![Data Preprocessing](path/to/screenshot2.png)

- **Screenshot 3/ Model Training**:
  ![Model Training](path/to/screenshot3.png)

- **Screenshot 4/ Model Evaluation**:
  ![Model Evaluation](path/to/screenshot4.png)

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
