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
- **Libraries:** PyTorch, OpenCV, Matplotlib, scikit-learn, pandas, numpy, 
- **Tools:** Jupyter Notebook, Git, VS Code

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

3. **Run the Jupyter Notebook**: Launch Jupyter Notebook and open [`Final Project League of Legends Match Predictor.ipynb`](https://github.com/ImaadhRenosh/Game-Match-Win-Prediction-Model-League-of-Legends-/blob/main/Final%20Project%20League%20of%20Legends%20Match%20Predictor.ipynb) to train and evaluate the model.

## Demo
<img width="1024" alt="Screenshot 2025-03-24 at 05 57 30" src="https://github.com/user-attachments/assets/6bb8147c-2c46-4c14-a9f9-d0a46d3a3520" />

<div style="margin-top: 30px">
  <h2>Step 1: Load the League of Legends dataset and preprocess it for training</h2>
  <hr/>
</div>

<img width="1005" alt="Screenshot 2025-03-24 at 05 58 33" src="https://github.com/user-attachments/assets/a483e73a-1618-4a02-976a-c0ad4c50311b" />

<img width="777" alt="Screenshot 2025-03-24 at 05 59 26" src="https://github.com/user-attachments/assets/df41475a-943c-4001-a9a6-af0e46e4a9e6" />

<img width="267" alt="Screenshot 2025-03-24 at 06 00 44" src="https://github.com/user-attachments/assets/554bdf5c-a993-4308-a568-4870abe535d2" />

<div style="margin-top: 30px; margin-bottom: 30px;">
  <h2>Step 2: Implement a logistic regression model using PyTorch</h2>
  <hr/>
</div>

<img width="1009" alt="Screenshot 2025-03-24 at 06 06 38" src="https://github.com/user-attachments/assets/a8d5ea0a-f73a-4cce-b6e1-dbcd088e9f5b" />

<img width="534" alt="Screenshot 2025-03-24 at 06 10 38" src="https://github.com/user-attachments/assets/9ba45d85-2d11-467d-8b9a-81e035cd8aa9" />

<img width="721" alt="Screenshot 2025-03-24 at 06 08 41" src="https://github.com/user-attachments/assets/5b15e6f9-d21b-47af-abcd-27e827a75da2" />

<div style="margin-top: 30px; margin-bottom: 30px;">
  <h2>Step 3: Train the logistic regression model on the dataset</h2>
  <hr/>
</div>

<img width="972" alt="Screenshot 2025-03-25 at 03 42 14" src="https://github.com/user-attachments/assets/6f668cff-3f62-4b56-8293-cb6738d3392c" />
<img width="750" alt="Screenshot 2025-03-25 at 03 42 37" src="https://github.com/user-attachments/assets/548a09c3-922e-42e2-8223-1cba997bc09b" />
<img width="717" alt="Screenshot 2025-03-25 at 03 46 04" src="https://github.com/user-attachments/assets/0819f79c-2ed2-4cca-80c8-bfb6978e062c" />
<img width="608" alt="Screenshot 2025-03-25 at 03 49 29" src="https://github.com/user-attachments/assets/16f4068d-611b-4c63-8de6-3df5e1ac5704" />
<img width="550" alt="Screenshot 2025-03-25 at 03 50 00" src="https://github.com/user-attachments/assets/3cacd50e-8e93-4ba9-887c-59d3d1686c2e" />
<img width="360" alt="Screenshot 2025-03-25 at 03 50 22" src="https://github.com/user-attachments/assets/103dbd6a-cfad-4740-9813-8cb0a308d84c" />

<div style="margin-top: 30px; margin-bottom: 30px;">
  <h2>Step 4: Implement optimization techniques and evaluate the model's performance</h2>
  <hr/>
</div>

<img width="1009" alt="Screenshot 2025-03-26 at 04 50 36" src="https://github.com/user-attachments/assets/9ec7a5f3-08c1-4963-8dc1-d69eec8ee1e1" />
<img width="785" alt="Screenshot 2025-03-26 at 04 51 06" src="https://github.com/user-attachments/assets/ac93d756-a041-4862-9e79-86bc183d346d" />
<img width="814" alt="Screenshot 2025-03-25 at 03 59 19" src="https://github.com/user-attachments/assets/c3fe8515-45bb-41f9-a895-dadf6a729156" />
<img width="718" alt="Screenshot 2025-03-25 at 03 59 46" src="https://github.com/user-attachments/assets/5b19bfaf-ef83-40a7-8cad-2b3464a250b9" />
<img width="588" alt="Screenshot 2025-03-25 at 04 00 07" src="https://github.com/user-attachments/assets/0f68cd0c-f4ce-4db1-ba8c-84838645b1f9" />
<img width="1010" alt="Screenshot 2025-03-25 at 04 14 21" src="https://github.com/user-attachments/assets/3f19ee2c-0909-454d-8c05-39b35f6d95fa" />

<div style="margin-top: 30px; margin-bottom: 30px;">
  <h2>Step 5: Visualize the model's performance and interpret the results</h2>
  <hr/>
</div>

<img width="854" alt="Screenshot 2025-03-25 at 04 22 50" src="https://github.com/user-attachments/assets/7e78403e-e2ac-4b60-b90d-d7ba8ddd9400" />
<img width="905" alt="Screenshot 2025-03-25 at 04 23 33" src="https://github.com/user-attachments/assets/3c9bd463-2519-42f6-9a22-df5546a14871" />
<img width="608" alt="Screenshot 2025-03-25 at 04 24 58" src="https://github.com/user-attachments/assets/ba93455f-eb0e-4c30-81ca-48d9f7aa595c" />
<img width="593" alt="Screenshot 2025-03-25 at 04 25 22" src="https://github.com/user-attachments/assets/476e2e6c-6afa-42ea-98d1-e7772636488e" />

<div style="margin-top: 30px; margin-bottom: 30px;">
  <h2>Step 6: Save and load the trained model</h2>
  <hr/>
</div>

<img width="978" alt="Screenshot 2025-03-25 at 04 52 32" src="https://github.com/user-attachments/assets/6eebb217-f6e9-4dcd-87f5-0c3fad32a8ac" />
<img width="705" alt="Screenshot 2025-03-25 at 04 31 06" src="https://github.com/user-attachments/assets/a90b0e52-819d-4ef0-9545-8104ce8d4ad6" />
<img width="816" alt="Screenshot 2025-03-25 at 04 33 24" src="https://github.com/user-attachments/assets/300b14dc-7365-432f-9710-3eca76fb3ca5" />
<img width="544" alt="Screenshot 2025-03-25 at 04 33 48" src="https://github.com/user-attachments/assets/e16772a9-fcc7-4523-a4cc-82c1e4f899bb" />

<div style="margin-top: 30px; margin-bottom: 30px;">
  <h2>Step 7: Perform hyperparameter tuning to find the best learning rate</h2>
  <hr/>
</div>

<img width="974" alt="Screenshot 2025-03-25 at 04 46 47" src="https://github.com/user-attachments/assets/6cab6cfb-f7f9-452f-97f6-2569905d87ec" />
<img width="737" alt="Screenshot 2025-03-25 at 04 47 18" src="https://github.com/user-attachments/assets/5323d386-0e9b-40b7-bd73-95568beba839" />
<img width="786" alt="Screenshot 2025-03-25 at 04 47 54" src="https://github.com/user-attachments/assets/578c6da2-0757-46db-9797-32074a9a9689" />
<img width="666" alt="Screenshot 2025-03-25 at 04 48 34" src="https://github.com/user-attachments/assets/65278e0e-fd9e-452c-9dfc-dd25d9a06dcb" />

<div style="margin-top: 30px; margin-bottom: 30px;">
  <h2>Step 8: Evaluate feature importance to understand the impact of each feature on the prediction</h2>
  <hr/>
</div>

<img width="989" alt="Screenshot 2025-03-25 at 04 54 23" src="https://github.com/user-attachments/assets/074bcb29-2219-4708-bed1-2d1c5485e714" />
<img width="829" alt="Screenshot 2025-03-25 at 04 55 51" src="https://github.com/user-attachments/assets/fb3961ee-c97c-49ff-ab2c-684e55a51f36" />
<img width="821" alt="Screenshot 2025-03-25 at 04 57 22" src="https://github.com/user-attachments/assets/d5432173-d965-44cb-b93a-bb5fd9a64b32" />
<img width="725" alt="Screenshot 2025-03-25 at 04 57 42" src="https://github.com/user-attachments/assets/f35ea01d-67c6-421c-972e-6be4ab8e29d5" />
<img width="892" alt="Screenshot 2025-03-25 at 04 58 05" src="https://github.com/user-attachments/assets/9cc65d6d-dcf9-4e4c-95dc-78faf27e4d49" />
<img width="603" alt="Screenshot 2025-03-25 at 04 58 21" src="https://github.com/user-attachments/assets/96f23ac2-c3d4-4fc7-b99e-b0b7de40649a" />

<div style="margin-top: 30px;margin-bottom: 30px;">
  <h2>Conclusion</h2>
  <hr/>
</div>

<img width="1007" alt="Screenshot 2025-03-25 at 05 05 06" src="https://github.com/user-attachments/assets/fe26ea23-c306-4ad4-9f1d-ff49b6f8403d" />


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
This project is licensed under © IBM Corporation. All rights reserved.

## Acknowledgements
- Thanks to the contributors of scikit-learn, pandas, and numpy.
- Special thanks to the League of Legends community for their support and data.

## Contact
For any questions or suggestions, feel free to reach out:
- **Email:** imaadhrenosh@gmail.com
- **LinkedIn profile**: [LinkedIn profile](https://www.linkedin.com/in/imaadh-renosh-007aba348)
