
# Notes
## For optimizer scalability

- You cannot create a class in host memory and use a virtual function inside it from device code and viceversa
- The class cannot be memcpy'ed into device memory, must be created in host/device code

### Method 1: Using a factory to avoid virtual functions
- This method would solve problems but it would make scaling it a pain and it gives me cognitive dissonance

### Method 2: Using a wrapper class
- Creating a class with virtual functions
	- So if it isn't an inherited class vanilla behaviour is used

- Creating inherited classes that specialize in a specific regularization/optimization method
- Creating a class to be called inside host code that manages the real classes that are on device memory
	- This class is just used for constructor-destructor-adaptation for evolution methods
- The device class is passed as argument and manages the gradient subtraction by itself


# Todo

## High priority

- ~~Improve scalability for evolution methods, modularize it, current state is not modularized at all.~~
    - Needs testing (Medium priority)
- Add L1 and L2 regullarization (Take scalability into account)
- Add Optimizers (Take scalability into account)
	- Just Adam for now

## Normal cycle

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
	- Needs testing, it has not even been compiled

# Bugs
 - ~~device array being used as host array in NeuronLayer and LSTMLayer remove neuron~~
 - ~~bug in add neuron~~
 - constrain evolution
    * if removing neuron in a layer with 1 neuron remove layer (now it won't delete the neuron)
 - ~~constrain weights and biases to prevent nans (reset NaNs)~~

### Not to do until all Layer constructors are made:

- ~~Tensorflow-like class constructor~~
- ~~Save, Load, Cloning~~
- Crossover
- ~~Evolution~~
- ~~Modify kernel launches to have more capacity of neurons (current max 1024)~~
