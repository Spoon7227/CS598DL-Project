# CS598DL-Project

This project replicates the results from [DueTT](https://doi.org/10.48550/arXiv.2304.13017)

This project is ran with python 3.9.18. Dependencies are provided in requirements.txt

The jupyter notebook DL4H_Team_77.ipynb contains all nessisary cells to run this project.

## Results

PhysioNet-2012 Mortality Task Performance

Below is the comparison of this project's replication of the original paper's results with other methods. Other method's performance data is sourced from the original paper.
All methods use seed 2020 with ours using seed 2020-2022.

| Model           | ROC-AUC    | PR-AUC    |
|-----------------|------------|-----------|
| XGBoost         | 0.865 ± 0.001 | 0.531 ± 0.009 |
| LSTM            | 0.848 ± 0.002 | 0.494 ± 0.002 |
| mTAND           | 0.857 ± 0.001 | 0.515 ± 0.007 |
| Raindrop         | 0.838 ± 0.009 | 0.479 ± 0.002 |
| STraTS           | 0.852 ± 0.008 | 0.527 ± 0.006 |
| **DuETT (Ours)**    | **0.872 ± 0.001** | **0.554 ± 0.003** |


| Ablation Study           | PR-AUC    | $$\Delta$$    |
|-----------------|------------|-----------|
| **DuETT** 		  |	**0.554 ± 0.003** |  |
| First layer embedding only         | 0.550 ± 0.002 | -0.004 |
| Binning with mean aggregation            | 0.552 ± 0.009 | -0.002 |
| No Pre-training           | 0.515 ± 0.005 | -0.039 |
| Event Transformer Only         | 0.535 ± 0.006 | -0.019 |
| Time Transformer Only           | 0.522 ± 0.004 | -0.032 |



## References

This project utilizes code from [DuETT Official Code](https://github.com/layer6ai-labs/DuETT/tree/master).

Original Paper:
1.   Labach, A., Pokhrel, A., Huang, X. S., Zuberi, S., Yi, S. E., Volkovs, M., Poutanen, T., & Krishnan, R. G. DuETT: Dual Event Time Transformer for Electronic Health Records. arXiv, 2023, arXiv:2304.13017, doi: https://doi.org/10.48550/arXiv.2304.13017
