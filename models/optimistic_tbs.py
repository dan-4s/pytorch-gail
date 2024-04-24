import numpy as np
import time
import torch
import wandb
import copy

from torch.nn import Module

from models.nets import PolicyNetwork, ValueNetwork, Discriminator
from utils.funcs import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch

if torch.cuda.is_available():
    from torch.cuda import FloatTensor

    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class GAIL(Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            discrete,
            train_config=None
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)
        self.v = ValueNetwork(self.state_dim)
        self.d = Discriminator(self.state_dim, self.action_dim, self.discrete)

        self.pi_prev = copy.deepcopy(self.pi)
        self.v_prev = copy.deepcopy(self.v)
        self.d_prev = copy.deepcopy(self.d)

        self.max_grad_norm = 10.0

    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state):
        self.pi.eval()

        state = FloatTensor(state)
        distb = self.pi(state)

        action = distb.sample().detach().cpu().numpy()

        return action

    def collect_trajectories(self, env, policy, num_steps_per_iter, horizon, render=False):
        episodes = []
        total_steps = 0
        while total_steps < num_steps_per_iter:
            episode = {'obs': [], 'acts': [], 'rwds': [], 'rets': []}
            t = 0
            done = False
            ob, _ = env.reset()
            ob = np.round(ob, decimals=2)
            while not done and total_steps < num_steps_per_iter:
                if policy == self.pi:
                    act = self.act(ob)
                else:
                    act = policy.act(ob)
                episode['obs'].append(ob)
                episode['acts'].append(act)
                episode['rets'].append(self.d(torch.tensor(ob), torch.tensor(act)).cpu().detach())
                if render:
                    env.render()
                ob, rwd, done, info, _ = env.step(act)
                episode['rwds'].append(rwd)
                ob = np.round(ob, decimals=2)
                t += 1
                total_steps += 1

                if horizon is not None:
                    if t >= horizon:
                        done = True
                        break
            episodes.append(episode)
        return episodes

    def compute_is(self, obs, acts):
        policy_distribution_curr = self.pi(obs)
        log_probs_curr = policy_distribution_curr.log_prob(acts)
        probs_curr = torch.exp(log_probs_curr)
        policy_distribution_prev = self.pi(obs)
        log_probs_prev = policy_distribution_prev.log_prob(acts)
        probs_prev = torch.exp(log_probs_prev)
        is_factor = probs_prev / probs_curr
        return is_factor

    def optimistic_TBS_update_discriminator(self, opt_d, num_disc_iter, eta, exp_episodes, agent_episodes):

        # Collect all expert observations and actions
        exp_obs = torch.cat([FloatTensor(np.array(episode['obs'])) for episode in exp_episodes])
        exp_acts = torch.cat([FloatTensor(np.array(episode['acts'])) for episode in exp_episodes])
        # Collect all agent observations and actions
        agent_obs = torch.cat([FloatTensor(np.array(episode['obs'])) for episode in agent_episodes])
        agent_acts = torch.cat([FloatTensor(np.array(episode['acts'])) for episode in agent_episodes])

        with torch.no_grad():  # Context manager for no gradient computation
            # policy_distribution_curr = self.pi(agent_obs)
            # log_probs_curr = policy_distribution_curr.log_prob(agent_acts)
            # probs_curr = torch.exp(log_probs_curr)
            # policy_distribution_prev = self.pi(agent_obs)
            # log_probs_prev = policy_distribution_prev.log_prob(agent_acts)
            # probs_prev = torch.exp(log_probs_prev)
            is_factor = self.compute_is(agent_obs, agent_acts)

            exp_Dcurr = self.d(exp_obs, exp_acts)
            exp_Dprev = self.d_prev(exp_obs, exp_acts)
            agt_Dcurr = self.d(agent_obs, agent_acts)
            agt_Dprev = self.d_prev(agent_obs, agent_acts)
            # Calculate gradients of loss w.r.t. D(s,a)
            exp_grad_curr = 1 / exp_Dcurr
            agt_grad_curr = -1 / (1 - agt_Dcurr + 1e-8)
            exp_grad_prev = 1 / exp_Dprev
            agt_grad_prev = is_factor.unsqueeze(dim=1) * (-1 / (1 - agt_Dprev + 1e-8))

        loss_hist = []

        # update d_prev from t-1 to t, which will be t-1 in the next training loop
        self.d_prev = copy.deepcopy(self.d)

        for i in range(num_disc_iter):
            exp_D = self.d(exp_obs, exp_acts)
            agt_D = self.d(agent_obs, agent_acts)
            # Construct surrogate loss
            exp_surrogate = (-(2 * exp_grad_curr - 1 * exp_grad_prev) * (exp_D - exp_Dcurr) + (0.5 / eta) * torch.pow(
                exp_D - exp_Dcurr, 2))
            agt_surrogate = (-(2 * agt_grad_curr - 1 * agt_grad_prev) * (agt_D - agt_Dcurr) + (0.5 / eta) * torch.pow(
                agt_D - agt_Dcurr, 2))

            # Total surrogate loss
            total_loss = torch.cat((exp_surrogate, agt_surrogate)).mean()
            if torch.isnan(total_loss).any():
                print("NaN detected in total_loss")
                # Optionally, add more diagnostic outputs here:
                print("Gradients:", [p.grad for p in self.d.parameters() if p.grad is not None])
            # Backpropagation and optimizer step
            opt_d.zero_grad()
            total_loss.backward()
            opt_d.step()
            loss_hist.append(total_loss.item())

        return loss_hist  # Optionally return the loss value for logging or monitoring purposes

    # Return estimation is totally different. This basically gives advantage + value function, not discounted value...  I don't think this is a correct estimate of return.
    # TODO: Assuming this is not a big deal, I will come back to it later if nothing else works.
    def calculate_gae_and_returns(self, rewards, values, gamma, lambda_, next_value):
        """
        Calculate Generalized Advantage Estimation (GAE) and returns.
        :param rewards: list of rewards per time step
        :param values: list of value estimates per time step
        :param gamma: discount factor for rewards
        :param lambda_: GAE decay parameter
        :param next_value: the estimated value of the next state at the end of the trajectory
        :return: returns and advantages as tensors
        """

        def normalize_advantages(advantages):
            if len(advantages) > 1 and advantages.std(unbiased=False) > 0:
                normalized_advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            else:
                # When there's no variance, return zero-centered advantages
                normalized_advantages = advantages - advantages.mean()
            return normalized_advantages

        returns = []
        gae = 0
        advantages = []
        # Reverse the rewards and values to start from the end of the trajectory
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] + gamma * next_value - values[t]
            else:
                delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lambda_ * gae
            advantages.insert(0, gae)
        adv = FloatTensor(advantages)
        return adv + values.squeeze(), normalize_advantages(adv)

    def update_critic(self, opt_v, num_v_updates, episodes, gae_gamma, gae_lambda):
        loss_hist = []
        gae_final = []
        self.v_prev = copy.deepcopy(self.v)
        for i in range(num_v_updates):
            states = []
            targets = []
            gaes = []
            for episode in episodes:
                self.v.eval()
                state = FloatTensor(episode['obs'])
                curr_vals = self.v(state).detach()
                next_vals = FloatTensor([[0.]])
                target, gae = self.calculate_gae_and_returns(FloatTensor(episode['rets']), curr_vals, gae_gamma,
                                                             gae_lambda,
                                                             next_vals)
                states.append(state)
                targets.append(target)
                gaes.append(gae)
            if i == 0:
                gae_prev = torch.cat(gaes)
            if i == num_v_updates - 1:
                gae_final = torch.cat(gaes)

            self.v.train()
            loss_critic = (self.v(torch.cat(states)).squeeze() - torch.cat(targets)).pow(2).mean()
            opt_v.zero_grad()
            loss_critic.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm(self.v.parameters(), self.max_grad_norm)
            opt_v.step()
            loss_hist.append(loss_critic.item())

        return loss_hist, gae_final, gae_prev


    def optimistic_TBS_update_actor(self, episodes, gae_curr, gae_prev, eta, num_a_iters, optimizer):
        obs = torch.cat([torch.tensor(np.array(episode['obs']), dtype=torch.float32) for episode in episodes])
        acts = torch.cat([torch.tensor(np.array(episode['acts']), dtype=torch.float32) for episode in episodes])
        gae_curr = torch.tensor(gae_curr, dtype=torch.float32).detach()  # Ensure advantages are a tensor
        gae_prev = torch.tensor(gae_prev, dtype=torch.float32).detach()  # Ensure advantages are a tensor
        surrogates_hist = []

        # Calculate log probabilities and policy probabilities at time t
        with torch.no_grad():  # Context manager for no gradient computation
            is_factor = self.compute_is(obs, acts)

            policy_distribution_curr = self.pi(obs)
            log_probs_curr = policy_distribution_curr.log_prob(acts)
            probs_curr = torch.exp(log_probs_curr)
            full_log_probs_curr = policy_distribution_curr.logits
            full_probs_curr = policy_distribution_curr.probs

            policy_distribution_prev = self.pi_prev(obs)
            log_probs_prev = policy_distribution_prev.log_prob(acts)
            probs_prev = torch.exp(log_probs_prev)
            full_log_probs_prev = policy_distribution_prev.logits
            full_probs_prev = policy_distribution_prev.probs

        # Calculate the part of the gradient w.r.t. policy output
        grad_L_wrt_pi_curr = (gae_curr / probs_curr) - 1e-2 * (log_probs_curr + 1)
        grad_L_wrt_pi_prev = is_factor * (gae_prev / probs_prev) - 1e-2 * (log_probs_prev + 1)

        # update pi_prev from t-1 to t, which will be t-1 in the next training loop
        self.pi_prev = copy.deepcopy(self.pi)

        for i in range(num_a_iters):
            # Surrogate loss calculation
            policy_distribution = self.pi(obs)
            log_probs = policy_distribution.log_prob(acts)
            probs = torch.exp(log_probs)
            full_log_probs = policy_distribution.logits
            full_probs = policy_distribution.probs

            kl_divergence = torch.sum(full_probs_curr * (full_log_probs_curr - full_log_probs),
                                      dim=-1)  # Proper KL divergence for discrete distributions
            surrogate_loss = (-(2 * grad_L_wrt_pi_curr - 1 * grad_L_wrt_pi_prev) * (probs - probs_curr) + (
                    1 / eta) * kl_divergence).mean()
            # surrogate_loss = (-grad_L_wrt_pi_curr * (probs - probs_curr) + (1 / eta) * kl_divergence).mean()
            # Take optimization step
            optimizer.zero_grad()
            surrogate_loss.backward()
            optimizer.step()
            surrogates_hist.append(surrogate_loss.item())

        return surrogates_hist


    def train(self, env, expert, render=False):
        num_iters = self.train_config["num_iters"]
        # num_iters = 2
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        # num_steps_per_iter = 5000
        horizon = self.train_config["horizon"]
        lambda_ = self.train_config["lambda"]
        gae_gamma = self.train_config["gae_gamma"]
        gae_lambda = self.train_config["gae_lambda"]
        eps = self.train_config["epsilon"]
        max_kl = self.train_config["max_kl"]
        cg_damping = self.train_config["cg_damping"]
        normalize_advantage = self.train_config["normalize_advantage"]

        ################## Discriminator update setup ##################
        num_disc_iter = 20
        opt_d = torch.optim.Adam(self.d.parameters(), lr=2e-3)
        eta_d = 3e-1

        ################## Value network update setup ##################
        num_v_iter = 20
        opt_v = torch.optim.Adam(self.v.parameters(), lr=2e-3)

        ################## Actor network update setup ##################
        num_a_iter = 20
        opt_a = torch.optim.Adam(self.pi.parameters(), lr=2e-3)
        eta_a = 3e-1

        ################## Stage 1: Collect Expert Trajectories ##################
        start_time = time.time()
        expert_episodes = self.collect_trajectories(env, expert, num_steps_per_iter, horizon, render=False)
        exp_rwd_iter = [np.sum(ep['rwds']) for ep in expert_episodes]

        print(
            "Expert Reward Mean: {}".format(np.mean(exp_rwd_iter))
        )
        wandb.log({
            "Expert Reward Mean": np.mean(exp_rwd_iter),
            "Time (minutes)": (time.time() - start_time) / 60,
        })

        ################## Stage 2: Training Loop ##################
        for i in range(num_iters):
            agent_episodes = self.collect_trajectories(env, self.pi, num_steps_per_iter, horizon, render=False)
            # disc_loss = self.update_discriminator(opt_d, num_disc_iter, expert_episodes, agent_episodes)
            disc_loss = self.optimistic_TBS_update_discriminator(opt_d, num_disc_iter, eta_d, expert_episodes,
                                                                 agent_episodes)
            v_loss, gae_curr, gae_prev = self.update_critic(opt_v, num_v_iter, agent_episodes, gae_gamma, gae_lambda)
            # gae_curr, gae_prev = self.MC_estimation_Q(agent_episodes, gae_gamma)
            # self.update_actor(agent_episodes, gaes, gae_gamma)
            surrogates = self.optimistic_TBS_update_actor(agent_episodes, gae_curr, gae_prev, eta_a, num_a_iter, opt_a)

            rwd_iter = [np.sum(ep['rwds']) for ep in agent_episodes]
            print(
                "Iterations: {},   Reward Mean: {}"
                    .format(i + 1, np.mean(rwd_iter))
            )
            wandb.log({
                "Agent Reward Mean": np.mean(rwd_iter),
                "Time (minutes)": (time.time() - start_time) / 60,
            })
