import torch, numpy as np
from typing import List, Optional, Tuple


EPS = 1e-8


def drnmf_sp(X: torch.Tensor, e21S: float, e22S: float, eCauchyS: float, alphas: List[float],
             n_iter: int = 300, r: Optional[int] = None, gamma: float = 1.0,
             device: str = "cpu", seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    
    assert isinstance(X, torch.Tensor) and X.dim() == 2 and len(alphas) == 3
    d, n = X.shape
    
    if r is None:
        r = max(1, min(d, n)//2)
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    X = X.to(device)
    t = X.dtype
    
    e_norm = torch.tensor([e21S/n, e22S/n, eCauchyS/n], dtype=t, device=device)
    a = torch.tensor(alphas, dtype=t, device=device)
    
    W = torch.rand((d, r), dtype=t, device=device)
    H = torch.rand((r, n), dtype=t, device=device)
    
    lam = torch.ones(3, dtype=t, device=device)/3
    lf = torch.ones(3, dtype=t, device=device)
    

    for it in range(n_iter):
        
        R = X - W @ H
        
        E1 = torch.norm(R, 2, 0).clamp(min=EPS)
        E2 = (E1**2).clamp(min=EPS)
        Ec = torch.log((E1**2 + gamma**2).clamp(min=EPS))
        
        th = (a * e_norm) / lam.clamp(min=1e-12)
        
        P1 = (E1 <= th[0]).to(t)
        P2 = (E2 <= th[1]).to(t)
        Pc = (Ec <= th[2]).to(t)
        
        d1 = (1/E1)*P1
        d2 = torch.ones_like(E1)*P2
        dc = (1/(E1**2 + gamma**2))*Pc
        
        c = (lf*lam)/e_norm
        Dv = (c[0]*d1 + c[1]*d2 + c[2]*dc).clamp(min=EPS)
        
        Xd = X * Dv.unsqueeze(0)
        
        Wu = Xd @ H.T
        Wd = (W @ (H * Dv.unsqueeze(0))) @ H.T
        W = W * (Wu / torch.maximum(Wd, torch.tensor(EPS, device=device, dtype=t)))
        
        Hu = W.T @ Xd
        Hd = (W.T @ W) @ (H * Dv.unsqueeze(0))
        H = H * (Hu / torch.maximum(Hd, torch.tensor(EPS, device=device, dtype=t)))
        
        e1 = lf[0]*torch.sum(E1)/e_norm[0]
        e2 = lf[1]*torch.sum(E2)/e_norm[1]
        ec = lf[2]*torch.sum(Ec)/e_norm[2]
        
        ef = torch.tensor([e1, e2, ec], dtype=t, device=device)
        
        eta = 1.0/(it+2)
        ls = torch.zeros_like(lam)
        ls[int(torch.argmax(ef).item())] = 1.0
        
        lam = ((1-eta)*lam + eta*ls)
        lam = lam / torch.maximum(torch.sum(lam), torch.tensor(EPS, device=device, dtype=t))
    

    return W.detach(), H.detach()
