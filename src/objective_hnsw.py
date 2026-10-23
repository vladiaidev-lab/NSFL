### OBJECTIVE_HNSW.PY

import torch
import geoopt
import time
import numpy as np

class ObjectiveHNSW:
    def __init__(self):
        self.device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sphere = geoopt.manifolds.Sphere()
     
    def optimize_on_sphere(self, objective_fn, dim=128, steps=100, lr=0.2, early_stopping_patience=10, init_value=None, return_stats=False, min_step_before_stop=None):
        """Optimize objective function on sphere (from manopt.py)

        return_stats: if True, return (x, stats_dict) instead of just x.
        stats_dict has 'stopped_at_step' (1-indexed step the loop exited on,
        == steps if it ran to completion without early stopping),
        'early_stopped' (bool), and 'final_objective' (float, the objective
        value at the returned x - added for EXP 3 multi-start comparison).
        Added for the EXP 1/2 convergence diagnostics - does not change the
        returned x for existing callers that don't pass return_stats=True.

        min_step_before_stop: the "step > steps // 3" floor before early
        stopping is allowed to fire, exposed as a parameter for EXP 2b
        (default None reproduces the original hardcoded steps//3 behavior
        exactly; pass 0 to let patience alone decide, with no floor).
        """
        if min_step_before_stop is None:
            min_step_before_stop = steps // 3
        if init_value is not None:
            # x = torch.from_numpy(init_value).to(self.device)
            x = (init_value if isinstance(init_value, torch.Tensor) else torch.from_numpy(init_value)).to(self.device)
            if x.dim() == 1:
                assert x.shape[0] == dim, "Init value dimension mismatch"
            else:
                raise ValueError("Init value should be a 1D array")
        else:
            x = torch.randn(dim, device=self.device)
        x = self.sphere.projx(x)
        x = geoopt.ManifoldParameter(x, manifold=self.sphere)
        
        optimizer = geoopt.optim.RiemannianSGD([x], lr=lr)
        if early_stopping_patience > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.8, patience=5
            )
        
            best_value = float('-inf')
            patience_counter = 0

        times = []
        start_total = time.time()
        stopped_at_step = steps
        early_stopped = False
        for step in range(steps):
            start_time = time.time()
            optimizer.zero_grad()
            # Negate the objective to maximize
            loss = -objective_fn(x)
            if early_stopping_patience > 0:
                current_value = -loss.item()
                if current_value > best_value:
                    best_value = current_value
                    patience_counter = 0
                else:
                    patience_counter += 1

            loss.backward()
            # torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)
            optimizer.step()
            x.data = self.sphere.projx(x.data)
            end_time = time.time()
            times.append(end_time - start_time)
            if step % 10 == 0 or step == steps - 1:
                # print(f"Step {step+1}/{steps}: objective={objective_fn(x).item():.4f}")
                if early_stopping_patience > 0:
                    scheduler.step(current_value)

            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience and step > min_step_before_stop:
                stopped_at_step = step + 1
                early_stopped = True
                break


        end_total = time.time()
        avg_time = sum(times) / len(times)
        total_time = end_total - start_total
        final_objective = float(objective_fn(x).item())
        # print(f"Final optimized objective value: {objective_fn(x).item():.4f}")
        # print(f"Average time per optimization step: {avg_time:.6f} seconds")
        # print(f"Total optimization time: {total_time:.6f} seconds")
        result = x.data.cpu().numpy().astype(np.float32)
        if return_stats:
            return result, {
                "stopped_at_step": stopped_at_step,
                "early_stopped": early_stopped,
                "total_time": total_time,
                "final_objective": final_objective,
            }
        return result