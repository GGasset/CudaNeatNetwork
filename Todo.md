# Todo
*80-90% done*

## Regularization - Optimizers

- ~~Add L1 and L2 regullarization~~
- ~~Add entropy regularization~~
- Learning rate scheduling
- ~~Add Optimizers (Take scalability into account)~~
	- ~~Adam~~
    - ~~Add saving/loading to disk~~
    - ~~Add hyperparameter structure for optimizers~~
        - ~~If that structure decides which optimizers to use at the last moment would be great. i.e. by passing it during subtract gradients and using the active hyperparameters, use a list that searches for a hyperparameter id (enum) and if it doesn't exists, it allocates more memory for it~~

## Code optimization
- Paralelize layer derivative calculation calls

- Optimize PRAM reduce
- ~~Add execution with output to GPU~~
- Optimize NEAT-Dense Connections with PRAM reduce
- Create Tensor NEAT-Dense Connections

## Fundamentals

- ~~LSTM~~
- ~~Neuron~~
- ~~NEAT connections~~ 
	- ~~Execute~~
	- ~~Gradients~~
    - ~~Optimize~~
        * ~~Various methods~~
        * ~~Evolution methods~~

- ~~Modularized generate random values for different data types~~
- Modular weight initialization
- ~~Add Xavier initialization~~
- Add Orthogonal initialization
- ~~Make droput set cost of neuron to 0 before its gradient calculation and remove previous dropout~~
    - ~~It just nullifies the gradient to substract to dropped out weights~~

- ~~Tensorflow-like class constructor~~
- ~~Save, Load, Cloning~~
- ~~Modify kernel launches to have more capacity of neurons (current max 1024)~~ 
    - New current max 65535 neurons in layer (maxGridDim.y)
        - remove that limit (Low priority)
            - DenseConnections
            - Output derivative calculation

## Training
- ~~return cost while training~~
- ~~adaptative learning rates~~
- ~~Supervised learning~~
- ~~reinforcement learning cost functions~~
    - ~~GAE (Generalized Advantage Estimator)~~
    - ~~PPO (Proximal Policy Optimization)~~
        - kl penalty
        - Vectorized environment
            - Gather a fixed number of steps from a fixed number of parallel environments and then train on the data
            - Each environment is not resetted with training and there is no need to finish the episode
            - If the environment finishes and the number of steps is not reached, it is restarted
        - MiniBatches 
            - After calculating advantages, inside the training loop of the training function
            - Shuffling is made to a copy of the arrays
            - With number of minibatches parameter.
            - Non recurrent version
                - Gets how many data_points per minibatch
                - Shuffles every index of every environment and environments too
                - Gets sequentially a mini-batch size of data point (the shuffled data) at a time
                    - trains on them
                    - appends the resulting gradients
            - Recurrent version
                - Get how many environments per minibatch
                - Shuffles the environments indices
                - if there are multiple environments per minibatch, just append its data
                    - The baselines implementation adds an assert nenvs % nminibatches == 0 for simplicity
        - Global gradient clipping 
            - scaling gradients in the update such as the l2 norm (sum of all values squared) does not exceed .5
            - basically a normalization
        - ~~Value loss clipping~~ (Not done, Irrelevant for results)
        - learning_rate_anhealing (Left to implement in main code, Note leave it small as the performace gain are also small)

- Create model links for multi-model networks
    - Create a training and execution function that accepts multiple networks and links
    - Links just copy output to input and cost backwards, with one connected neuron per neuron, without any computation

- Create logging options for training in csv format
    - Add python script for plotting it

## Socket

- Use poll instead of select
- Use epoll in linux
- Modularize core functionality to be swapped depending on OS
- Create destructor for socket interpreter -> NN manager
- Simplify file names to match class names
- Paralellize server side withing CPU
- Modularize parsing raw bytes (Higher priority than socket functions)
- Auto copy gradient hyperparameters header to client
- Accept gradient hyperparameters from socket

### Compatible protocols

* AF-INET
* ~~AF-UNIX Abstract address~~
* ~~AF-UNIX File binding~~

### Security
* ~~Bind to a file~~
* Use script for server init
    * Create file
    * Set permissions for socket file
    * Set max open fd for server
    * Start server
    * with flag -s
        - reboot

### Socket functions

- Add abstract address detection
- Add message to close server to stop loop
- ~~construct -- destruct~~
- Get ID of pointer to a NN
    you pass a id and returns another id that is a reference to a existing NN
    it has its own execution data to train in parallel
- training execute
- training functions
- save & load
- evolution methods
- Inference
- delete memory

## Evolution

- evolution
    * if removing neuron in a layer with 1 neuron remove layer (now it won't delete the neuron)
    * Check for neurons that are not pointed to or don't point to any other and delete them after calling evolve
    * ~~Constrain evolution metadata fields from 1E-5 to .3 for stable process~~
    * Crossover
        * If the less fit parent has the layer, get 50% of shared weights from each, else just from the more fit parent
- ~~constrain weights and biases to prevent nans (reset NaNs)~~

