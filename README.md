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

* `sim_study.ipynb`: Notebook to conduct simulation study of mastery learning with different problem selection algorithms. Evaluates how our method *fast-forwarding* enhances various existing selection algorithms.

* `src/`: Implementations of simulation environment, selection algorithms and utility functions. 

* `data/`: Parameters and solution paths for simulation study.

* `requirements.txt`: Specifies Python packaged required for the simulation study.


## Setup

The easiest way to use this project is to start with a new Conda environment. After that one can install all packages using the provided requirement file.

```
conda create python==3.11 -n fast_forwarding
conda activate fast_forwarding

# use one of the requirement files
pip install -r ./requirements.txt
```

Afterwards you will be able to execute the `sim_study.ipynb` notebook.
