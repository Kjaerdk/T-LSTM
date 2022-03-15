## Time-Aware LSTM

- Implemetation of (supervised) [T-LSTM network](https://dl.acm.org/doi/10.1145/3097983.3097997) in Python using PyTorch
  - Unsupervised version of the T-LSTM network would require some slight changes. Don't hesitate to reach out if you are interested in this.
- Solves the problem of sorting sequences of different lengths into batches without padding

### Compatibility
- Python 3.9.7
- PyTorch 1.9.1
- PyTorch Lightning 1.4.9
- Optuna 2.10.0

### Data

Preprocessed deidentified data used in the [original T-LSTM paper](http://https://github.com/illidanlab/T-LSTM.git) and in another [paper](https://github.com/sheryl-ai/Nature-Scientific-Reports) using the T-LSTM model can be found [here](https://github.com/sheryl-ai/Nature-Scientific-Reports/blob/master/Subtyping%20Procedure/data_PPMI.mat). 


### Reference

Inci M. Baytas, Cao Xiao, Xi Zhang, Fei Wang, Anil K. Jain, Jiayu Zhou, "Patient Subtyping via Time-Aware LSTM Networks", KDD, 2017.

### Contact 

Let me know if you have any questions, comments, or suggestions.