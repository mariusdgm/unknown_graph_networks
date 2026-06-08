# Control with learned should be as good as true model + optimal control 

M_lambda(x) = (1 - lambda) A_hat + lambda (A_hat x alpha(x))




# 2 Step experiment:
use current setup to learn A and alpha (we do system identification here) -> freeze the model
use frozen model to simulate control experiments where one agent tries to learn an importance vector
use the vector to greedily assign the control budget

compare this learned control with the original black box env


# In case of revision for current article, maybe use these new prop methods with the previous RL setup
