# Investigating interaction of mood and perceived reward

## `simulation.py`

This file contains the main code of the simulation. We define a reinforcement learning model in which mood and perceived reward interact. We run several experimental simulations in which we vary the parameter `f`, which defines the direction and magnitude of mood's impact on perceived reward. When `f > 1`, positive mood increases the perceived reward. When `f < 1`, positive mood decreases the perceived reward. We show that values of `f` above 1 lead to unstable oscillating mood states and reward states, while `f` values below 1 do not. 