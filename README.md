# GPT-Critic: Offline Reinforcement Learning for End-to-End Task-Oriented Dialogue Systems

The official implementation for the paper "GPT-Critic: Offline Reinforcement Learning for End-to-End Task-Oriented Dialogue Systems" in ICLR 2022.

### 1. Requirements

First of all, install following libraries and packages for conda environment:
```
conda env create -f environment.yml
conda activate gpt-critic
python -m spacy download en_core_web_sm
unzip data.zip
```


### 2. Train the GPT-Critic
Training the GPT-Critic can be started by running main.py as follows:
```
python main.py -mode train -algorithm $ALGORITHM -cfg iteration=$ITERATION seed=$SEED
```
- To choose among running the GPT-Critic, UBAR, Decision Transformer, and Weighted BC you need to set the value of variable `$ALGORITHM` to `GPT-Critic`, `UBAR`, `DT`, or `WBC` respectively.
- (Only for GPT-Critic) To choose the iteration, you need to change the value of variable `$ITERATION` to `0`, `1`, `2` or `3` respectively.
- To choose the random seed, you need to change the value of variable `$SEED` to `0`, `1`, or `2` respectively.

(Example)
```
python main.py -mode train -algorithm GPT-Critic -cfg iteration=3 seed=0
```

### 3. Evaluate the GPT-Critic

```
python main.py -mode test -algorithm $ALGORITHM -cfg iteration=$ITERATION seed=$SEED
```
- To choose among running the GPT-Critic, UBAR, Decision Transformer, and Weighted BC you need to set the value of variable `$ALGORITHM` to `GPT-Critic`, `UBAR`, `DT`, or `WBC` respectively.
- (Only for GPT-Critic) To choose the iteration, you need to change the value of variable `$ITERATION` to `0`, `1`, `2` or `3` respectively.
- To choose the random seed, you need to change the value of variable `$SEED` to `0`, `1`, or `2` respectively.

(Example)
```
python main.py -mode test -algorithm GPT-Critic -cfg iteration=3 seed=0
```

### Citation
If this repository helps you in your academic research, you are encouraged to cite our paper. Here is an example bibtex:
```bibtex
@inproceedings{jang2022gptcritic,
    title={{GPT}-Critic: Offline Reinforcement Learning for End-to-End Task-Oriented Dialogue Systems},
    author={Youngsoo Jang and Jongmin Lee and Kee-Eung Kim},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=qaxhBG1UUaS}
}
```

### Acknowledgement
This code is adapted and modified upon the [MultiWOZ](https://github.com/budzianowski/multiwoz) and [UBAR](https://github.com/TonyNemo/UBAR-MultiWOZ).
We appreciate their released dataset and code which are very helpful to our research.
