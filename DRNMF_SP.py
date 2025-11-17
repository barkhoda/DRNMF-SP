import os
import warnings
import numpy as np
import torch
import scipy.io
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
EPS = 1e-10
DEVICE = torch.device("cpu")

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

def multiplicative_update_W_H(X, W, H, D_diag):
    D_row = D_diag.unsqueeze(0)
    X_D = X * D_row
    H_D = H * D_row
    W *= (X_D @ H.t()) / (W @ (H @ H_D.t()) + EPS)
    WT = W.t()
    H *= (WT @ X_D) / ((WT @ W) @ (H * D_row) + EPS)
    W.clamp_(min=0.0)
    H.clamp_(min=0.0)
    return W, H

def compute_instance_residuals(X, W, H):
    return torch.norm(X - W @ H, dim=0)

def pure_nmf_2_2(X, W, H, max_iter=300):
    for _ in range(max_iter):
        W *= (X @ H.t()) / (W @ (H @ H.t()) + EPS)
        H *= (W.t() @ X) / ((W.t() @ W) @ H + EPS)
    return W, H, torch.norm(X - W @ H) ** 2

def pure_nmf_2_1(X, W, H, max_iter=300):
    for _ in range(max_iter):
        e = compute_instance_residuals(X, W, H)
        D = 1.0 / torch.maximum(e, torch.tensor(EPS, device=DEVICE))
        W, H = multiplicative_update_W_H(X, W, H, D)
    return W, H, torch.sum(torch.norm(X - W @ H, dim=0))

def pure_nmf_cauchy(X, W, H, gamma=1.0, max_iter=300):
    for _ in range(max_iter):
        e = compute_instance_residuals(X, W, H)
        D = 1.0 / (e ** 2 + gamma ** 2)
        W, H = multiplicative_update_W_H(X, W, H, D)
    loss = torch.sum(torch.log(torch.norm(X - W @ H, dim=0) ** 2 + gamma ** 2))
    return W, H, loss

def idrnmf(X, n_components, max_iter=300, gamma=1.0,
           eta0=0.9, eta_final=0.01, compute_zeta_iter=100,
           init_seed=0, normalize_columns=True, verbose=True,
           alpha_init=(0.1, 0.1, 0.1)):

    set_seed(init_seed)
    m, n = X.shape
    if normalize_columns:
        X = X / (torch.norm(X, dim=0, keepdim=True) + EPS)

    W = torch.rand(m, n_components, device=DEVICE)
    H = torch.rand(n_components, n, device=DEVICE)

    if verbose:
        print("Computing normalization constants (ζ)...")

    _, _, z21 = pure_nmf_2_1(X, W.clone(), H.clone(), compute_zeta_iter)
    _, _, z22 = pure_nmf_2_2(X, W.clone(), H.clone(), compute_zeta_iter)
    _, _, zcau = pure_nmf_cauchy(X, W.clone(), H.clone(), gamma, compute_zeta_iter)
    z21, z22, zcau = [float(z.item()) for z in [z21, z22, zcau]]

    if verbose:
        print(f"ζ21={z21:.3e}, ζ22={z22:.3e}, ζcau={zcau:.3e}")

    lam = torch.ones(3, device=DEVICE) / 3.0
    eps21, eps22, epsc = 1 / z21, 1 / z22, 1 / zcau
    alpha = torch.tensor(alpha_init, dtype=torch.float32, device=DEVICE)

    thr_hist = []
    lam_hist = []

    def eta(t):
        return eta0 + (eta_final - eta0) * (t / max_iter)

    sum_p1_prev = torch.tensor(float(n), device=DEVICE)
    sum_p2_prev = torch.tensor(float(n), device=DEVICE)
    sum_pc_prev = torch.tensor(float(n), device=DEVICE)

    for t in range(max_iter):
        e = compute_instance_residuals(X, W, H)

        L1_i = e
        L2_i = e ** 2
        Lc_i = torch.log(e ** 2 + gamma ** 2)

        lam_safe = torch.clamp(lam, min=1e-8)

        eps1_inst = z21 * (1.0 / torch.clamp(sum_p1_prev, min=1.0))
        eps2_inst = z22 * (1.0 / torch.clamp(sum_p2_prev, min=1.0))
        epsc_inst = zcau * (1.0 / torch.clamp(sum_pc_prev, min=1.0))

        thr1 = (alpha[0] * eps1_inst) / lam_safe[0]
        thr2 = (alpha[1] * eps2_inst) / lam_safe[1]
        thrc = (alpha[2] * epsc_inst) / lam_safe[2]

        p1 = (L1_i <= thr1).float()
        p2 = (L2_i <= thr2).float()
        pc = (Lc_i <= thrc).float()

        if p1.sum() == 0:
            p1 = torch.ones_like(p1)
        if p2.sum() == 0:
            p2 = torch.ones_like(p2)
        if pc.sum() == 0:
            pc = torch.ones_like(pc)

        sum_p1 = torch.clamp(p1.sum(), min=1.0)
        sum_p2 = torch.clamp(p2.sum(), min=1.0)
        sum_pc = torch.clamp(pc.sum(), min=1.0)

        eps1_inst = z21 * (1.0 / sum_p1)
        eps2_inst = z22 * (1.0 / sum_p2)
        epsc_inst = zcau * (1.0 / sum_pc)

        thr1 = (alpha[0] * eps1_inst) / lam_safe[0]
        thr2 = (alpha[1] * eps2_inst) / lam_safe[1]
        thrc = (alpha[2] * epsc_inst) / lam_safe[2]

        thr_hist.append([thr1.item(), thr2.item(), thrc.item()])

        p1 = (L1_i <= thr1).float()
        p2 = (L2_i <= thr2).float()
        pc = (Lc_i <= thrc).float()

        if p1.sum() == 0:
            p1 = torch.ones_like(p1)
        if p2.sum() == 0:
            p2 = torch.ones_like(p2)
        if pc.sum() == 0:
            pc = torch.ones_like(pc)

        sum_p1_prev = torch.clamp(p1.sum(), min=1.0)
        sum_p2_prev = torch.clamp(p2.sum(), min=1.0)
        sum_pc_prev = torch.clamp(pc.sum(), min=1.0)

        d1 = (1.0 / (e + EPS)) * p1
        d2 = torch.ones_like(e) * p2
        dc = (1.0 / (e ** 2 + gamma ** 2)) * pc

        D = lam[0] * eps21 * d1 + lam[1] * eps22 * d2 + lam[2] * epsc * dc
        W, H = multiplicative_update_W_H(X, W, H, D)

        e = compute_instance_residuals(X, W, H)
        L1 = (torch.sum(e * p1) / torch.clamp(p1.sum(), min=1.0)) * eps21
        L2 = (torch.sum((e**2) * p2) / torch.clamp(p2.sum(), min=1.0)) * eps22
        Lc = (torch.sum(torch.log(e**2 + gamma**2) * pc) / torch.clamp(pc.sum(), min=1.0)) * epsc

        losses = np.array([L1.item(), L2.item(), Lc.item()])
        obj = float(np.sum(lam.cpu().numpy() * losses))

        j = np.argmax(losses)
        lam_star = torch.zeros_like(lam)
        lam_star[j] = 1.0
        lam = (1 - eta(t)) * lam + eta(t) * lam_star
        lam /= lam.sum()

        lam_hist.append(lam.cpu().numpy().copy())

        if verbose and ((t + 1) % 10 == 0 or t == max_iter - 1):
            print(f"Iter {t+1:03d} | Obj={obj:.3e} | λ={lam.cpu().numpy()}")

    return W, H, np.array(thr_hist), np.array(lam_hist)


if __name__ == "__main__":
    mat_path = "D://Yale.mat"
    if not os.path.exists(mat_path):
        raise FileNotFoundError(mat_path)

    mat = scipy.io.loadmat(mat_path)
    X_np = mat["X"]
    y_np = mat["y"].flatten().astype(int)

    if X_np.shape[0] == len(y_np):
        X_np = X_np.T
    elif X_np.shape[0] < X_np.shape[1]:
        X_np = X_np.T

    print(f"Loaded X shape: {X_np.shape}")

    X = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
    r = len(np.unique(y_np))

    W, H, thr_hist, lam_hist = idrnmf(
        X,
        n_components=r,
        max_iter=300,
        gamma=1.0,
        eta0=0.9,
        eta_final=0.05,
        compute_zeta_iter=100,
        init_seed=0,
        normalize_columns=True,
        verbose=True,
        alpha_init=(0.1, 0.1, 0.1)
    )

    H_np = H.cpu().detach().numpy().T
    kmeans = KMeans(n_clusters=r, n_init=20).fit(H_np)
    y_pred = kmeans.labels_

    