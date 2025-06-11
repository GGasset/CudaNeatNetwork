# Todo

## High priority

- Improve scalability for evolution methods, simplify and modularize them, current state is not modularized at all.
    - ~~Add compatibility with optimizers~~
    - Needs testing (Medium priority)
- Add L1 and L2 regullarization (Take scalability into account)
- ~~Add Optimizers (Take scalability into account)~~
	- ~~Adam~~
    - ~~Add saving/loading to disk~~
- Make a function that abstracts the process of parsing raw bytes from socket for mental health

## Normal cycle

- Replace atomic function use with mutexs for optimization
- ~~LSTM~~
- ~~Neuron~~
- ~~NEAT connections~~ 
	- ~~Execute~~
	- ~~Gradients~~
    - ~~Optimize~~
        * ~~Various methods~~
        * ~~Evolution methods~~
- ~~adaptative learning rates~~
    - ~~return cost while training~~
- ~~reinforcement learning cost function~~
    - ~~GAE in GPU~~
        - ~~CPU function that given a value function estimator NN and rewards, computes GAE~~
            - Add execution with output to GPU to optimize this
    - Proximal Policy optimization cost function
- ~~server socket~~
    - ~~Add windows compatibility (Just easy change in header)~~
        - Needs testing
    - ~~Create log file~~
    - Create destructor for socket interpreter -> NN manager
    - Use poll instead of select
    - Improve security
        1. Bind to a file for security
        2. Use script for server init
            * Create file
            * Set chmod for socket file
            * Set max open fd for server
            * Start server
            * with flag --strict-security or -s
                - reboot

- ~~client_socket~~
- socket functions
    - Paralellize withing CPU
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

- Create logging options for training in csv format
    - Add python script for plotting it

- Paralelize layer derivative calculation calls withing CPU
- ~~Modularized generate random values for different data types~~
- ~~Make droput set cost of neuron to 0 before its gradient calculation and remove previous dropout~~
    - ~~It just nullifies the gradient to substract to dropped out weights~~

# Bugs
 - ~~device array being used as host array in NeuronLayer and LSTMLayer remove neuron~~
 - ~~bug in add neuron~~
 - evolution
    * if removing neuron in a layer with 1 neuron remove layer (now it won't delete the neuron)
    * Check for neurons that are not pointed to or don't point to any other and delete them after calling evolve
 - ~~constrain weights and biases to prevent nans (reset NaNs)~~

### Not to do until all Layer constructors are made:

- ~~Tensorflow-like class constructor~~
- ~~Save, Load, Cloning~~
- Crossover
- ~~Evolution~~
- ~~Modify kernel launches to have more capacity of neurons (current max 1024)~~
