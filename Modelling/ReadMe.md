# Model Training and Explanation Generation

## Training the Model

You can train the model using the following command:

``` train.py <dataset you want to train with```
ex: ```python train.py dataset_2.csv```

This will train and save four models (LightGBM, XGBoost, Random Forest, and Gradient Boosting) into log results and will save the best model it finds.

## Generating Explanations

```python generate_explanation.py --dataset <dataset> --modelname <modelname> --log <log> --session <session>```
    
ex: ```python generate_explanation.py --dataset dataset3.csv --modelname XGBoost --log log_1720005500 --session 52_1_Arena_Clash```

## Pipeline

The pipeline to get a prediction using new datafile is still WIP
    