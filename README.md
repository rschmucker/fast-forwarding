# Optimizing Mastery Learning by Fast-Forwarding

Supplementary repository of the paper "Optimizing Mastery Learning by Fast-Forwarding Over-Practice Steps" accepted as full paper to ECTEL '25.

## Citation

Xia, M., Schmucker, R., Borchers, C., & Aleven, V. (2025). Optimizing Mastery Learning by Fast-Forwarding Over-Practice Steps. In *Proceedings of the 20th European Conference on Technology Enhanced Learning* (ECTEL25). Springer Cham. 
```
@inproceedings{xia2025optimizing,
  title={Optimizing Mastery Learning by Fast-Forwarding Over-Practice Steps},
  author={Xia, Meng and Schmucker, Robin and Borchers, Conrad and Aleven, Vincent},
  booktitle={ECTEL25: 20th European Conference on Technology Enhanced Learning},
  organization={Springer Cham},
  year={2025}
}
```


## Repository structure

* `sim_study.ipynb`: Notebook to conduct simulation study of mastery learning with different problem selection algorithms. Evaluates how our method *fast-forwarding* enhances various existing problem selection algorithms.

* `empirical_analysis/`: Data processing and notebook for empirical analysis of student overpractice and multi-step solution paths. 

* `src/`: Implementations of simulation environment, selection algorithms and utility functions. 

* `data/`: Parameters and solution paths for simulation study.

* `requirements.txt`: Specifies Python packages required for the simulation study.


## Setup

The easiest way to use this project is to start with a new Conda environment. After that one can install all packages using the provided requirements file.

```
# set up a new environment 
conda create python==3.11 -n fast_forwarding
conda activate fast_forwarding

# use the requirements files
pip install -r ./requirements.txt
```

Afterwards you will be able to execute the `sim_study.ipynb` notebook.

To replicate the empirical analysis of student overpractice and multi-step solution paths first follow the steps outlined in `data_prep.R` and then execute the `empirical_analysis.ipynb` notebook. To run the empirical analysis you will have to request the required datasets from DataShop and put them in the `data/` folder. This process is described in the following section.


### Data setup

First, open the three CMU Datashop dataset webpages in your browser.

https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=5153
<br>https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=5549
<br>https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=5604

To access the datasets, you will need to request access with an account on DataShop. You can create an account using your Google or GitHub account, whichever is easiest.

Once you have created an account, navigate back to the dataset webpages and click on the button `Request Access`. Provide your reasoning to access the dataset and click `Confirm`. You should receive an email once the PI approves the request; however, you can also check by seeing whether you can click the `Export` button on the project webpage.

To get the final datasets, click the `Export` button. On the left hand side, make sure under `Shared Samples` that there is a checkbox next to `All Data` by clicking it. Then, click the `Export Transactions` button when it appears. Wait for the server to process your request, and then you should have three files `ds*_tx_All_Data_*_<timestamp>.txt`. 
You should then rename these three files to align with the paths referenced in `data_prep.R`. Afterwards execute `data_prep.R` to create the combined file that is used as input for `empirical_analysis.ipynb`.

Lastly, while we provide the set of AFM parameters used for the simulation study in the data/ folder, you can estimate the parameters yourself using the implementation by [Borchers et al.](https://github.com/conradborchers/peerchats-edm), based on the following two datasets:

https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=5549
<br>https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=5604
