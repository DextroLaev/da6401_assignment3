# 🔤 Character-Level Seq2Seq & Attention Transliteration

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


## 🚀 Training the Attention/Vanilla Based Encoder-Decoder Model

1. ### Before training the model you need to create a `models/` directory parent folder i.e `./`
    ```bash
    mkdir models
    ```

2.  ### First install the necessary dependencies
    ```bash
    pip3 install -r requirements.txt
    ```

3. ### To train the attention/vanilla model, run the following command:
    ```bash
    python train.py
    ```
4. ### To get the `input_dim` and `output_dim` for the model, run the following command:
    ```bash
    python dataset.py
    ```
    to change the output_lang, open the `dataset.py` file and change the lang from `hi` to your own one that the current dataset supports, after that you can run the above command to get the `output_dim` of your own language.


5. ### To train the attention/vanilla (with user input), run the following command:
    ```bash
    python main.py -h_dim 256 -e_dim 64 --enc_layers 3 --dec_layers 3 --cell_type LSTM -do 0.2 -e 30 -b 16 -lr 0.001 -t_forcing 0.6 -arch transformer --evaluate True -lang hi --input_dim 28 --output_dim 65 --save_model True --log_wandb
    ```
    This will start training the network and start logging in wandb. If you don't want to log the data in Wandb then remove the `--log_wandb` flag.
    
    Remember that `input_dim` and `output_dim` values can be obtained by running `dataset.py` file seperately.

4. ### After the training , run the following command to test it
    ```bash
    python test.py -h_dim 256 -e_dim 64 --enc_layers 3 --dec_layers 3 --cell_type LSTM -do 0.2 -b 16 -t_forcing 0.6 -arch transformer -lang hi --input_dim 28 --output_dim 65 --log_wandb
    ```





