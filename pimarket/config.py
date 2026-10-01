from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    # --- parameters named in the paper (Sections 4-5) ---
    N: int = 6            # number of community members
    M: float = 3.0        # each consumer's attention budget
    M_infl: float = 6.0   # the influencer's attention budget
    r_p: float = 1.0      # rate at which every producer posts
    r_0: float = 1.0      # rate of new content from outside sources
    B_0: float = 0.3      # chance an outside item interests a consumer

    # --- numerical choices the paper does NOT fix (ours) ---
    f_scale: float = 3.0      # f(d) = exp(-f_scale * d)
    g_scale: float = 2.0      # g(d) = exp(-g_scale * d)
    theta_scale: float = 1.0  # theta(mu) = 1 - exp(-theta_scale * mu)

    seed: int = 0             # for reproducible initial states

    def __post_init__(self):
        # runs right after the object is built; rejects nonsense values
        if self.N < 2:
            raise ValueError("need at least 2 community members")
        for name in ("M", "M_infl", "r_p", "r_0",
                     "f_scale", "g_scale", "theta_scale"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if not 0.0 <= self.B_0 <= 1.0:
            raise ValueError("B_0 is a probability, must be in [0, 1]")
