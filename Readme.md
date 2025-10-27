# Overview
## What?

* CudaNeatNetwork is an AI framework made from scratch using CUDA/C++
* Does not support transformers, and there is no plan for it.

## Technologies used
### Planned-WIP
 - [Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400)
 - Input / Reward Normalization
 - C++ Socket bindings (WIP)
 - [PPO](https://arxiv.org/abs/1707.06347) Reinforcement Learning Policy Gradient Method used in ChatGPT (Testing)
 - [MinLSTM](https://arxiv.org/abs/2410.01201) Simpler versions of LSTM (Planned)

### Done
 - [SoftMax Activation](https://en.wikipedia.org/wiki/Softmax_function)
 - [L1, L2 regularization](https://developers.google.com/machine-learning/glossary#l1-loss)
 - [Adam optimizer](https://arxiv.org/abs/1412.6980)
 - [GAE](https://arxiv.org/abs/1506.02438) (Generalized Advantage Estimator)
 - [LSTM architecture](https://i.sstatic.net/RHNrZ.jpg) 

## Why?
* CudaNeatNetwrok seeks model structure flexibility without trading significant speed loss.
	* Tensor core unit usage is not implemented yet.
