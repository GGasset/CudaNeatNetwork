# Todo
*80-90% done*

## Regularization - Optimizers

- Add L1 and L2 regullarization (Take scalability into account)
- ~~Add Optimizers (Take scalability into account)~~
	- ~~Adam~~
    - ~~Add saving/loading to disk~~

## Code optimization
- Replace atomic function use with mutexs for inter-neuron parallelization

- Paralelize layer derivative calculation calls withing CPU

- ~~Add execution with output to GPU~~

- Make a function that abstracts the process of parsing raw bytes from socket for mental health

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
    - New current max 65535 neurons in layer (maxGridDim.y), only applies if Dense connections are used, else infinite practically
        - remove that limit (Low priority)

## Training
- ~~return cost while training~~
- ~~adaptative learning rates~~
- ~~Supervised learning~~
- reinforcement learning cost functions
    - ~~GAE (Generalized Advantage Estimator)~~
    - PPO (Proximal Policy Optimization)

- Create logging options for training in csv format
    - Add python script for plotting it

### PPO psudocode in each PPO train batch called t_count times per batch
Accept reward array to be accessed only at last t
Modularize into 2 functions (PPO_execute, PPO_train)

#### Execution Function
* Save initial hidden state of first execution
* Execute given the input appending network output for training
* return execution output and cost

#### Training function:

* Clone NN (NN_tmp), set to initial hidden state
* Calculate GAE, train value function
* while True: (i) (Loop for each hidden training iteration)
    * Execute NN_tmp_ on Inputs t_count times
    * Calculate PPO costs
    * if KL_divergence > threshold, or clipped PPO is clipped:
        break
    * Calculate gradients on execution values
        * Append gradients to array for later substraction to NN
    * Subtract gradients to NN_tmp

* Subtracts all collected gradients to NN
* Set training variables to default (affects outside of function)

## Socket
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

## Evolution

- evolution
    * if removing neuron in a layer with 1 neuron remove layer (now it won't delete the neuron)
    * Check for neurons that are not pointed to or don't point to any other and delete them after calling evolve
    * ~~Constrain evolution metadata fields from 1E-5 to .3 for stable process~~
    * Crossover
        * If the less fit parent has the layer, get 50% of shared weights from each, else just from the more fit parent
- ~~constrain weights and biases to prevent nans (reset NaNs)~~

