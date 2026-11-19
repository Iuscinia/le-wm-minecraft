"""
Planner selects the next best action to take given the latent embedding of
predicted observations
"""
import numpy as np
import torch
import torch.nn.functional as F


class Planner:
    def __init__(
            self,
            max_iter,
            n_samples,
            n_elites,
            planning_horizon,
            action_dim,
            rollout_batch_size=64,
    ):
        self.max_iter = max_iter
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.horizon = planning_horizon
        self.action_dim = action_dim
        self.rollout_batch_size = rollout_batch_size

    def objective_function(self, z_curr, z_g):
        """
        Computes the squared difference between the predicted final latent embedding
        at the end of a time horizon (z_H) with the latent embedding of the target (z_g)
        to evaluate if the model reached the goal state
        """
        return F.mse_loss(z_curr.reshape(-1), z_g.reshape(-1))

    def planner(self, lewm, obs, obs_goal, cam_mean, cam_std, sigreg_lambd=0.1, warm_start=None):
        """
        Samples and selects the best action sequence to take given a single
        current observation and goal observation.
        - Cross Entropy Method (CEM): selects 300 candidate action sequences up to time horizon (H)
        - LeWM rollout: actions are applied to current observation to predict the next observation embedding
        - Action selection: the best action seq selected based on the lowest planning cost

        Input: current observation and goal observation frame
        Outputs: (H, A) best action sequence found after max_iter
        """

        # TODO: Implement planner here
        costs = []

        # Iterate the maximum number of times
        for t in range(self.max_iter):
            rng = np.random.default_rng()
            action_samples = rng.normal(0, np.eye(self.action_dim), self.n_samples)

            # Sample n_samples times
            for i in range(self.n_samples):
                forward_output = lewm.forward(obs, action_samples)
                # costs.append(self.objective_function())


        planning_losses = {
            "current_goal_mse": current_goal_mse,  # current diff with goal
            "cem_best_cost": best_score,
            "final_goal_mse": final_goal_mse,  # final diff at H with goal
            "current_latent": z1.detach(),
            "goal_latent": zg.detach(),
            # "imagined_latents": imagined_latents.detach(),
        }
        distribution_params = {
            "mu": mu,
            "sigma": sigma,
            "p": p,
        }
        # return best action sequence
        return samples[best_elite_idx], planning_losses, distribution_params