"""SIGReg: sliced Epps-Pulley characteristic-function match to N(0, I)."""
import torch

def sigreg(x, step, num_slices=1024, t_points=17, t_range=5.0):
    """SIGReg: sliced Epps-Pulley characteristic-function match to N(0, I_K).

    x: (N, K) batch of embeddings. Returns a scalar loss that is ~O(1) for
    x ~ N(0, I_K) and grows as the batch distribution departs from it.

    Why each piece is the way it is:
      - fresh random unit directions every step (seeded by the global step):
        the *expected* loss integrates the 1-D test over the whole sphere,
        which by Cramer-Wold pins down the full K-dim distribution;
      - target CF exp(-t^2/2): the CF of N(0,1), because any unit projection
        of an isotropic Gaussian is exactly N(0,1);
      - weight exp(-t^2/2) on the squared CF error: the Epps-Pulley weighting;
        downweights large |t| where the ECF is pure noise, keeps the statistic
        and its gradients bounded;
      - real cos/sin arithmetic instead of exp(1j*...): mathematically identical
        (the target CF is real) and supported on MPS, where complex ops are not;
      - trapezoid quadrature over 17 points on [-5, 5]: the reference's grid;
      - "* N": calibrates the null value to O(1) -- ECF fluctuations have
        variance ~1/N, so the un-scaled integral would vanish as 1/N.
    """
    N, K = x.shape
    g = torch.Generator(device="cpu")
    g.manual_seed(int(step))            # resample directions each step, reproducibly
    A = torch.randn((K, num_slices), generator=g)
    A = (A / A.norm(p=2, dim=0, keepdim=True)).to(x.device, x.dtype)  # unit directions
    t = torch.linspace(-t_range, t_range, t_points, device=x.device, dtype=x.dtype)
    phi = torch.exp(-0.5 * t**2)                       # CF of N(0,1)
    proj = (x @ A).unsqueeze(2) * t                    # (N, num_slices, T)
    # |ECF(t) - phi(t)|^2, computed without complex numbers:
    err = (proj.cos().mean(0) - phi).square() + proj.sin().mean(0).square()
    err = err * phi                                    # Epps-Pulley weighting
    per_slice = torch.trapezoid(err, t, dim=1) * N     # quadrature, x batch size
    return per_slice.mean()
