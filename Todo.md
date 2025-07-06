# Todo
*80-90% done*

## Regularization - Optimizers

- Add L1 and L2 regullarization (Take scalability into account)
- Add entropy regularization
- Add layer regularization 
- ~~Add Optimizers (Take scalability into account)~~
	- ~~Adam~~
    - ~~Add saving/loading to disk~~

## Code optimization
- Replace atomic function use with mutexs for inter-neuron parallelization

- Paralelize layer derivative calculation calls withing CPU

- ~~Add execution with output to GPU~~

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

- Create model links for multi-model performance
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

