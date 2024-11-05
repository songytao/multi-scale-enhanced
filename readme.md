This code has been implemented in python language using Pytorch library and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3
- Pytorch

## Prepare data, Train/Test
1) You can train this model using your own data by modifying it,This model defaults to the tn3k dataset, which you can use to reproduce this code
2) Run the following code to install the Requirements.

    `pip install -r requirements.txt`
3) Enter the address where the New_train.py file is located through the cd command,Use the following command for training.In this code, you can adjust the parameters -use_val and -use_test separately to verify and test the model
    python New_train.py 
   ```bash
    python New_train.py
   ```
4) If you just want to test the performance of the model, you can use the following code
   ```bash
    python eval.py 
   ```