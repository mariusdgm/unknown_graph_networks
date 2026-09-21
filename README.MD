# Control with learned should be as good as true model + optimal control 

Run one experiment with sparsity constraint in the A learning, so we don't get diffuse learned A

Model based approach:
Try to deduce the analytic correct propagation function and add it to the baseline. For the cases where this is possible, we make one more explanation and add the results. Then we say for the other ones it's not that easy so we use the baseline with no prop method

We'll see about ablation

# In case of revision for current article, maybe use these new prop methods with the previous RL setup


## parallel ps start:
cd D:\Work\repos\RL\unknown_graph_networks\opinion_dynamics\experiments\
activate phd_rl_algos
run_lambda_mix_shards_60s_poll.cmd python 3 300
run_single_shot_exploration_shards.cmd python 3 60