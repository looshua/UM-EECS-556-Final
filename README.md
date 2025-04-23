# Faster LR3M
This repository was developed for the final project for the EECS 556 course of the University of Michigan.

In this project, we modify and improve on [LR3M](https://github.com/tonghelen/LR3M-Method/tree/master) by replacing the nuclear norm regularization with alternative denoising methods, resulting in a 50x speed-up with improved robustness. 

# Requirements
To install requirements,
```
pip install -r requirements.txt
```

# Evaluation
Evaluation can be done by running ```eval_brightener.py``` with appropriate image files. Metrics are computed in MATLAB, with the evaluators contained in the metrics file.
