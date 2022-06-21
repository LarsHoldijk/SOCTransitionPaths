import torch
from tqdm import tqdm


def PICE(env, policy, n_rollouts, n_samples, n_steps, dt, std, dim, R,
               logger, force, plotters=None, verbose=True, file=None, device='cpu', lr=0.00001, start_step=0):

    p_dim = dim
    u_dim = dim // 2 # Doesn't include the positions
    T = dt * n_steps

    std_ = torch.eye(u_dim).to(device) * (std)
    lambda_ = R @ ((std_ ** 2) / dt)
    lambda_ = lambda_[0, 0] # Every element should be the same anyway

    for r in range(start_step, n_rollouts):
        if verbose: print(f"Rollout: {r}")

        # Setup bookkeeping
        paths = torch.zeros((n_samples, n_steps+1, p_dim), device=device) 
        us = torch.zeros((n_samples, n_steps, u_dim), device=device) 
        costs_q = torch.zeros((n_samples, n_steps), device=device) 
        costs_action = torch.zeros((n_samples, n_steps), device=device) 
        costs_noise = torch.zeros((n_samples, n_steps), device=device) 

        # Setup bookkeeping for path costs
        path_cost = torch.zeros((n_samples), device=device)

        # Setup optimizers
        policy_optimizers = torch.optim.SGD(policy.parameters(), lr=lr)
        policy_optimizers.zero_grad()

        # We have to manually keep track of the grads as we need to weight them
        # by the final costs of the paths.
        grad_policy = []

        noise = torch.normal(torch.zeros(n_samples, n_steps, u_dim).to(device),
                             torch.ones(n_samples, n_steps, u_dim).to(device) * std)

        with torch.no_grad():
            if verbose:
                if plotters is not None:
                    for plotter in plotters:
                        if plotter.plot_now(r, 'before_epoch'):
                            plotter.plot(None, r, None)

        paths[:, 0, :] = env.starting_positions(n_samples)
        new_state = paths[:, 0, :].to(device)

        for s in tqdm(range(0, n_steps)):
            # if verbose: print(s)
            x = new_state.detach()
            t = torch.tensor(s * dt)
            # Determine the action
            if force:
                update_u = policy (x, t)
            else:
                _, update_u = policy(x, t)

            # We need to get the gradient wrt the parameters for each input individually as we need to scale them later based on the score for each trajectory
            for n in range(0, n_samples):
                tmp_ = -1 * R @ (noise[n, s, :]) * dt
                if n == (n_samples-1):
                    tmp = torch.autograd.grad(update_u[n], policy.parameters(), retain_graph=False, allow_unused=True, grad_outputs=tmp_)
                else:
                    tmp = torch.autograd.grad(update_u[n], policy.parameters(), retain_graph=True, allow_unused=True, grad_outputs=tmp_)# * dt
                tmp_new = []
                for element in tmp:
                    tmp_new.append(element)
                tmp = tmp_new
                if s == 0:
                    grad_policy.append(tmp)
                else:
                    grad_policy[n] = tuple(sum(x) for x in zip(grad_policy[n], tmp))

            update_eps = noise[:, s]
            update_action = (update_u + update_eps) * dt

            # Determine dynamics
            update_f = env.f(x, t) * dt
            update_g = torch.einsum('bji,bi->bj', env.G(x), update_action)

            # Create new state
            update = update_f + update_g
            new_state = x + update

            # Determine cost
            uR = torch.einsum("bi,ii -> bi", update_u, R)
            uRu = torch.einsum("bi,bi -> b", uR, update_u)
            uReps = torch.einsum("bi,bi -> b", uR, noise[:, s]).abs()

            cost_step_action = (uRu / 2) * dt
            cost_step_noise = uReps * dt
            cost_step_q = env.q(new_state) * dt

            cost_step = (cost_step_action + cost_step_noise) / lambda_ + cost_step_q
            path_cost += cost_step

            # Store
            paths[:, s + 1, :] = new_state
            us[:, s, :] = update_u
            costs_q[:, s] = cost_step_q
            costs_noise[:, s] = cost_step_noise
            costs_action[:, s] = cost_step_action

        # This is not in the original method, but it should help with us not setting T = 1
        path_cost /= T

        path_cost_phi = env.phi(new_state)
        path_cost_final = path_cost + path_cost_phi
        path_cost_final_subtract = path_cost_final - path_cost_final.min()
        path_cost_exp = torch.exp(-(path_cost_final_subtract))
        normalizing = path_cost_exp.sum()

        # Move back to GPU
        normalizing = normalizing.to(device)
        path_cost_exp = path_cost_exp.to(device)

        # Scale gradients
        for t_idx, t in enumerate(grad_policy):
            for param_idx, param in enumerate(t):
                param = param.to(device)
                if t_idx == 0:
                    policy_optimizers.param_groups[0]['params'][param_idx].grad = torch.zeros_like(param)
                scaled_param = (param * path_cost_exp[t_idx]) / normalizing

                if scaled_param.isnan().any():
                     print("here we go")

                # Update optimizer with scaled gradients
                policy_optimizers.param_groups[0]['params'][param_idx].grad += scaled_param


        # Take gradient step
        policy_optimizers.step()

        logger.log(paths[:, :, :].detach(), costs_q, costs_noise, costs_action, path_cost, path_cost_phi, path_cost_final, path_cost_exp, policy, r)


        env.reset()


