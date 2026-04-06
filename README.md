# Homework 1

## Setup and Installation

    conda env create -f environment.yml
    conda activate cse579a1


## Running the assignment

    python main.py --env reacher/pointmaze --train behavior_cloning/dagger/diffusion --policy gaussian/autoregressive/diffusion

Examples:

    python main.py --env reacher --train behavior_cloning --policy gaussian
    python main.py --env pointmaze --train dagger --policy gaussian
    python main.py --env pointmaze --train diffusion --policy diffusion

## Files you need to touch:
More details in the [assignment spec](https://courses.cs.washington.edu/courses/cse579/24au/projects/homework1/CSE_579_Au24.pdf).
- main.py (only for hyperparameter tuning) 
    - The assignment will ask you to change certain hyperparameters in main.py like the batch_size or number of training steps.
- DiffusionPolicy.py (for the extra credit)
    - There are three different TODO blocks to implement diffusion policies.
- dagger.py (for your implementation of dagger)
    - There are two different TODO blocks to implement dagger.
- bc.py (for your implementation of bc)
    - There are one different TODO blocks to implement bc.
- utils.py (for the autoregressive model)
    - There is two different TODO blocks to implement the autoregressive model.
