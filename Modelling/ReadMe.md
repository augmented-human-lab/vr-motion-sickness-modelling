# Model Training and Explanation Generation

For the scripts train.py and generate_explanation.py, you have to edit the relevant root path to the root folder path of vr-motion-sickness-modelling folder at your local machine

## Training the Model

You can train the model using the following command:

``` train.py <dataset you want to train with```
ex: ```python train.py dataset_2.csv```

This will train and save four models (LightGBM, XGBoost, Random Forest, and Gradient Boosting) into log results and will save the best model it finds.

## Generating Explanations

```python generate_explanation.py --dataset <dataset> --modelname <modelname> --log <log> --session <session>```
    
ex: ```python generate_explanation.py --dataset dataset3.csv --modelname XGBoost --log log_1720005500 --session 52_1_Arena_Clash```

## Pipeline

This is the pipeline to get a prediction to a new file. 
Check the changes you need to do to pipeline.py which are commented out.
You can then run the file to get the predictions.