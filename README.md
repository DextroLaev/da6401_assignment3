# 🔤 Character-Level Seq2Seq & Attention Transliteration

Wandb Report Link: (https://wandb.ai/cs24s031-indian-institute-of-technology-madras/dl-assignment3/reports/DA6401-Assignment-3-Report--VmlldzoxMjgyMzIwNQ?accessToken=b0sqn5yaods27wd0ekkgvs4yrl49hatfvj9iwyea2zobum1vi1hu6tqcwnw0cytl)

A PyTorch‐based toolkit for character-level transliteration for Dakshina transliteration dataset and supports hyperparameter sweeps, This project includes:

- A vanilla Seq2Seq model that uses Encoder-Decoder.
- A Attention based Seq2Seq model that uses Encoder-Decoder.
- Built-in training and testing pipeline that supports multiple language translation along with wandb logging.
- Evaluation using test data 

---

## 🧠 Model Overview

The Encoder-Decoder model (`Seq2Seq_Model`) supports:
- Encoder as well as Decoder with different layer size.
- Configurable cell type such as (`RNN`,`LSTM`,`GRU`).
- Supports different Embedding size layer.
- Optional dropout.
- Bidirectional support for Encoder.
- Training on GPU if available
  
The Encoder-Decoder model (`Attenion_Network`) supports above all along with the `attention` mechanism

---

## 📁 Directory Structure
    /dakshina_dataset_v1.0/
    /models/                        # after training, the model is saved in this directory
    /plots/                         # contains some plot like confusion matrix, attention-heatmap, etc.
    /web_page/                      # contains the html file that help to visualize the attention based heatmaps.
    /predicted_vanilla/             # Have a .txt file that have the prediction made by the vanilla model.(English_Word,Predicted_Word,Actual_Word)
    /predicted_attention/           # Have a .txt file that have the prediction made by the attention model.(English_Word,Predicted_Word,Actual_Word)
    ./
    ├── AttentionModel.py           # Attention-based decoder + training/eval loops
    ├── config.py                   # Global hyperparameter defaults & device config
    ├── dataset.py                  # Dakshina Dataset loader & preprocessing
    ├── model.py                    # Vanilla Seq2Seq encoder, decoder, beam search
    ├── run_sweep.py                # W&B sweep for vanilla Seq2Seq + beam search
    ├── run_sweep_attention.py      # W&B sweep for Attention model
    ├── train.py                    # Script to train vanilla Seq2Seq (beam sizes 1,2,4)
    ├── test.py                     # Script to load & evaluate saved Attention model or vanilla model
    ├── main.py                     # Runs the training of attention or vanilla by taking command line arguments
    ├── utils.py                     # Contains some of the helper function needed by other scripts.
    ├── connectitvity.py                     # Contains the script that creates the visualization of attention heatmap script.
    

## 📂 Repository Structure

### 🔬 Experimentation Notebooks (`.ipynb` files)
- These Jupyter Notebook files were initially used to **test** and **evaluate** the vanilla model.  
- They are primarily for **experimentation** and performance checks.  
- These files are **not** part of the actual working code.

### ⚙️ Core Python Files (`.py` files)
These files contain the essential code for building and running the neural network:

- **`dataset.py`** 📊  
  - Handles **data loading** and **pre-processing**.  
  - Currently supports **dakshina_dataset_v1.0**.

- **`config.py`** ⚡  
  - contains the model specific configuration to run and train the respective models.
  

- **`train.py`** ⚡  
  - This file is used to train the attention/vanilla model with the default configuration.

- **`model.py`** 🤖  
  - Implements the **Vanilla Model**.
  - Contains the `Seq2Seq` class with methods like `forward`, `train_model`, etc.
  - This file also have the seperate `Encoder` and `Decoder` class respectively
  - The code is modular, allowing easy modifications to the Seq2Seq Model.
  - The file aslo have the code for evalution of the vanilla model.
  - **Note:** This file only contains the algorithm of the Vanilla Seq2Seq network and is not meant to be executed directly.

- **`AttentionModel.py`** 🤖  
  - Implements the **Attention Based Model**.
  - Contains the `Attention_Network` class with methods like `forward`, `train_model`, etc.
  - The code is modular, allowing easy modifications to the attention network.
  - This file also have the code for the evaluation of attention model.
  - **Note:** This file only contains the algorithm of the Attention network and is not meant to be executed directly. There is two ways to train the model, all the three ways are mentioned below.

- **`run_sweep_net.py`** 🤖  
  - This code run the sweep for the vanilla Encoder-Decoder model.
  - This is used to log the selected parameters in the `wandb` platform.
  - One can check the performance of the vanilla network on various hyperparameter by running this file and checking the `wandb` site.
  - Currently it logs `train loss`, `train Accuracy`, `validation loss` and `validation accuracy`.
  - One can change the `sweep_config` present inside the code, to check the performance of vanilla network on different hyperparameters.

- **`run_sweep_attention.py`** 🤖  
  - This code run the sweep for the attention based Encoder-Decoder model.
  - This is used to log the selected parameters in the `wandb` platform.
  - One can check the performance of the attention network on various hyperparameter by running this file and checking the `wandb` site.
  - Currently it logs `train loss`, `train Accuracy`, `validation loss` and `validation accuracy`.
  - One can change the `sweep_config` present inside the code, to check the performance of vanilla network on different hyperparameters.


- **`main.py`**
  - This script is used to train attention/vanilla model for `dakshina_dataset` using configurable hyperparameters.  
  - Adjustable input dimension size. (Size depends on the input language)
  - Configurable encoder_layers number and decoder_layers number.
  - Flexible cell type to choose from (`RNN`,`LSTM`,`GRU`)
  - Optional dropout
  - Configurable hidden dimension 
  - Support for logging and visualizing experiments with Weights & Biases (wandb)
  - Supports model saving.
  
       **Arguments:**
      
      | Argument                     | Description |
      |------------------------------|-------------|
      | `--wandb_project` `(-wp)`     | WandB project name (required if logging). |
      | `--wandb_entity` `(-we)`      | WandB entity/user/team name (required if logging). |
      | `--hidden_dim` `(-h_dim)`     | hidden dim size (eg: 16,32,64,128,256). |
      | `--embed_dim` `(-e_dim)`      | Embedding layer dimension. |
      | `--enc_layers` `(-e_layers)` | Number of encoder layers (eg: 1,2,3). |
      | `--dec_layers` `(-d_layers)` | Number of decoder layers (eg: 1,2,3). |
      | `--cell_type` `(-c_type)`        | type of the encoder-decoder cell (LSTM/RNN/GRU). |
      | `--dropout` `(-do)`           | Dropout rate (e.g., 0.3). |      
      | `--epochs` `(-e)`             | Number of training epochs. |
      | `--batch_size` `(-b)`         | Training batch size. |
      | `--learning_rate` `(-lr)`     | Learning rate for optimizer. |
      | `--weight_decay` `(-w_d)`     | Weight decay (L2 regularization). |
      | `--teacher_forcing` `(-t_forcing)` | Teacher forcing ration used in decoder while training (0.2,0.3,0.4,etc). |
      | `--architecture` `(-arch)`     | Choose which architecture to use. (transformer,vanilla) |
      | `--evaluate`      | Make it True if you want to get testing accuracy. |
      | `--language` `(-lang)`      | language you want to translate to (eg: `hi`,`bn`,`gu`,`kn`,etc). |
      | `--input_dim` `(-i_dim)`      | Run the dataset.py seperately after making changes in the dataset.py, it will print the input dim, use that value. |
      | `--output_dim` `(-o_dim)`      | Run the dataset.py seperately after making changes in the dataset.py, it will print the output dim, use that value. |
      | `--save_model` `(--save_model)` | Save the trained model checkpoint (True/False). |
      | `--log_wandb` `(-logw)`       | Enable logging to wandb. |


- **`test.py`** 📊  
  - This code is used evaluate the performance of the attention/vanilla model on the testing dataset.
  - The arguments of this file is same as that of the `main.py` file.
  - While testing make sure that you pass the same argument values that are needed to test the model.
  - **Note:** This file don't take `--epochs`,`--learning_rate`,`--weight_decay`,`--evaluate`,`--save_model` as arguments, there is no need to use these arguments as this file onyl evaulate on the test dataset.
  
- **`conectivity.py`** 📊  
  - After running this code, you can directly check the visualization in the wandb run or by launching the python server. Procedure to check by launching the server is given below.

- **`utils.py`** 📊  
  - This code contains all the helper functions needed in `test.py` file and `connectivity.py` file.
  - This file contains the code to create the visualization also.
  - You can't run this file directly, it's just used in other files.


## 🚀 Training the Attention/Vanilla Based Encoder-Decoder Model

1. ### Before training the model you need to create a `models/` directory parent folder i.e `./` and also install the requirements.tx
    ```bash
    mkdir models
    pip install -r requirements.txt
    ```

2.  ### First install the necessary dependencies
    ```bash
    pip3 install -r requirements.txt
    ```

3. ### To train the attention/vanilla model (with best hyperparameters), run the following command:
    ```bash
    python train.py -arch <attention/vanilla>
    ```
    Either choose `-arch attention` or `-arch vanilla`. This script will train the model, log different `metrics` in wandb platform and also save the model in `models/` folder.
4. ### To get the `input_dim` and `output_dim` for the model, run the following command:
    ```bash
    python dataset.py
    ```
    to change the output_lang, open the `dataset.py` file and change the lang from `hi` to your own one that the current dataset supports, after that you can run the above command to get the `output_dim` of your own language.


5. ### To train the attention/vanilla (with user input), run the following command:
    ```bash
    python main.py -h_dim 256 -e_dim 64 --enc_layers 3 --dec_layers 3 --cell_type LSTM -do 0.2 -e 30 -b 16 -lr 0.001 -t_forcing 0.6 -arch attention --evaluate True -lang hi --input_dim 28 --output_dim 65 --save_model True --log_wandb
    ```
    This will start training the network and start logging in wandb. If you don't want to log the data in Wandb then remove the `--log_wandb` flag.
    
    Remember that `input_dim` and `output_dim` values can be obtained by running `dataset.py` file seperately.

6. ### After the training , run the following command to test it
   While running the test script, make sure you pass the same valid arguments used to train the model or else the script will not work.
    ```bash
    python test.py -h_dim 256 -e_dim 64 --enc_layers 3 --dec_layers 3 --cell_type LSTM -do 0.2 -b 16 -t_forcing 0.6 -arch attention -lang hi --input_dim 28 --output_dim 65 --log_wandb
    ```

6. ### After the training is done , Run the following command to visualize the attention heatmap by hover over the predicted characters.
   When you hover over a predicted character, it will highlight the characters in the input word by their attention score. `The more the attention score , the more the intensity of the highlighted word`.
    ```bash
    python connectivity.py
    ```
    This will do some processing and randomly picks 10 different data from test dataset and make the visualization script and fill the `.html` files present inside the `web_page/` folder. Once this is done, do the following.

    1. Either you can check the wandb run. THe link will be printed on the temrinal itself.
    2. Or you can run the server to launch this in your own browser.
  
    Run the following command to run the server and open the page in browser
    ```bash
    cd web_page/
    python -m http.server 8080
    ```
    now copy the followin link and paste it in your browser
    ```
    http://localhost:8080/attention.html
    ```






