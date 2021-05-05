# Python Personality - Feature Selection 

This python tests run Run a set of experiments for selected algorithms. All results are saved to text files.
The client demo test read from a file and save to data frames in the server
1. features -- the feature matrix
2. target -- the observations

Then run an ado_run_experiment with 
1. model -- choose if the regression is linear or logistic
2. selected_size -- how many features will be selected

The client is choosing between 3  different feature selection algorithms
1. SDS_OMP
2. SDS_MA 
3. Top_k 

## Running the test

```
SERVER_IP=<your server IP address> python3 src/python/python-personality/omp/omp/tests/demo.py
```  

