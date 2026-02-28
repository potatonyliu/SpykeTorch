# Exploration

- wandb
- log baseline and records
- loop through parameters
  - sweep in Wanda with sweep.yaml
- Evalution metrics
  - Accuracy (recall & recision)
  - Spike rate
  - weight mean
  - weight_std
  - kernel_diversity
- Config driven
  - config = {    "k1": 5, "r1": 5,    "conv1_threshold": 30,    "ltp": 0.027, "ltd": -0.004,    "timesteps": 60,    "epochs": 5, }
- Logging
  - After each meaningful experiment or debugging session
    - Hypothesis
      - This experiement does the following
    - Configurations and fixed configs
      - What changed, numbers, algorithms...
    - Results
      - metrics, plots, wandb
    - Interpretation
      - Errors - what I know don't work / abandoned approach
      - What may caused xxx
    - next step

- ```bash
  ./run.sh
  torchrun --standalone --nproc_per_node=8 train_gpt.py
  ```

  - Logs/
    - steps
      - step
      - training time
      - Step_average_time
      - weight updates
      - winner diversity...
    - code
    - checkpoints (state.dict)