## Perception-Driven Curiosity with Bayesian Surprise

### Running the Code

We have a Docker image in which the code can be run. However, we do not provide it here to retain anonymity.

The base requirements of our code which can mostly be installed with pip3 are:

``
python3
tensorflow-1.12
numpy
joblib
imageio
gym[atari]
``

Run demo_run.sh. 

This script has example parameters filled out for most of the key command line arguments. Paths for saving checkpoints and logging are default to local paths from where you run the script.

This code supports all command line arguments in OpenAI Baselines on top of which we built our model.

Additional command line arguments include but are not limited to intrinsic and extrinsic reward scaling, a choice of curiosity model to run, and weight tuning parameters.

### Acknowledgements

The PPO algorithm in this repository was taken from OpenAI Baselines: https://github.com/openai/baselines. The model infrastructure code and implementation of ICM was taken from the ICM project repository: https://github.com/pathak22/noreward-rl.
