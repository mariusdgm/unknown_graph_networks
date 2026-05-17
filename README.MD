# Draft up in overleaf, formulas for new dynamics and for the models we use for learning

# Rerun experiment with no control, make sure we cover the 0-1 space with random inits, for the last learned model then do control and check metrics
# Make baseline with uniform policy instead of concentrated
# Plot trajectories with complete learned model (learned A and alpha)

# 2 Step experiment:
use current setup to learn A and alpha (we do system identification here) -> freeze the model
use frozen model to simulate control experiments where one agent tries to learn an importance vector
use the vector to greedily assign the control budget

compare this learned control with the original black box env


# In case of revision for current article, maybe use these new prop methods with the previous RL setup