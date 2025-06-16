# Todo
*80-90% done*

## Regularization - Optimizers

- Add L1 and L2 regullarization (Take scalability into account)
- ~~Add Optimizers (Take scalability into account)~~
	- ~~Adam~~
    - ~~Add saving/loading to disk~~

## Code optimization
- Replace atomic function use with mutexs for optimization

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
- reinforcement learning cost function
    - ~~GAE in GPU~~
        - ~~CPU function that given a value function estimator NN and rewards, computes GAE~~
    - Proximal Policy optimization cost function

- Create logging options for training in csv format
    - Add python script for plotting it

### PPO psudocode in each PPO train batch called t_count times per batch
Accept reward array to be accessed only at last t
Modularize into 2 functions (PPO_execute, PPO_train)

* Save every hidden layer state and append it to array pointer passed as argument
* Execute given the input appending network output for training
* If its last t:
    * foreach t in t_count:
        * Clone NN (NN_tmp)
        * Calculate GAE advantage, train value function
        * while KL_divergence < threshold, and clipped PPO is not clipped:
            * Execute NN_tmp on input[t] and hidden_state[t] saving training values
            * Calculate gradients with PPO_clip cost function with training values and output[t]
                * Append gradients to array so they can later be subtracted to NN
            * Subtract gradients to NN_tmp
    * Subtracts all collected gradients to NN
    * Set training variables to default (affects outside function)
* return execution output and cost

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

