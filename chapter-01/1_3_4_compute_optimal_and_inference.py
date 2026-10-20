"""1.3.4  Compute-optimal vs over-trained.

    training compute   C ~ 6 N D
    inference / training cost ratio = 2 N T / (6 N D) = T / (3 D)     (N cancels)

Part A: for a compute budget C, find the N and D = C/(6N) that minimize L(N, D),
        numerically AND with the closed form, for two published constant sets.
Part B: how far real models sit from the optimum (tokens per parameter).
Part C: the inference break-even point.

    python chapter-01/1_3_4_compute_optimal_and_inference.py
"""
import numpy as np

from common import SCALING_CONSTANTS, header, scaling_loss


def optimum_numeric(C, c):
    Ns = np.logspace(6, 13, 4000)
    Ds = C / (6 * Ns)
    L = np.array([scaling_loss(n, d, c) for n, d in zip(Ns, Ds)])
    i = int(L.argmin())
    return Ns[i], Ds[i], L[i]


def optimum_closed_form(C, c):
    k = SCALING_CONSTANTS[c]
    a, b = k["alpha"], k["beta"]
    G = (a * k["A"] / (b * k["B"])) ** (1 / (a + b))
    N = G * (C / 6) ** (b / (a + b))
    return N, C / (6 * N)


def main():
    header("Part A: compute-optimal split for Chinchilla's own budget (C = 5.88e23 FLOPs)")
    C = 6 * 70e9 * 1.4e12
    print(f"  Chinchilla actually trained: N = 70B, D = 1.4T  ({1.4e12 / 70e9:.0f} tokens/param)\n")
    print(f"  {'constants':<10} {'exponent N~C^':>14} {'N_opt':>9} {'D_opt':>9} {'tok/param':>10}  numeric check")
    for c, k in SCALING_CONSTANTS.items():
        a, b = k["alpha"], k["beta"]
        N, D = optimum_closed_form(C, c)
        Nn, Dn, _ = optimum_numeric(C, c)
        print(f"  {c:<10} {b / (a + b):>14.2f} {N / 1e9:>8.1f}B {D / 1e12:>8.2f}T {D / N:>10.0f}  "
              f"N={Nn / 1e9:.1f}B (agrees: {abs(Nn - N) / N < 0.01})")
    print("\n  What to take from this:")
    print("  * Both fits say N and D should grow TOGETHER with compute (exponents near 0.5).")
    print("  * The exact tokens-per-parameter ratio is NOT robust. The paper's original Approach-3")
    print("    constants give ~93 here, while the Epoch AI replication gives ~18, in line with the")
    print("    ~20 tokens/param Chinchilla actually used. Replications found the original fit's")
    print("    implied policy inconsistent with the paper's own other approaches.")
    print("  * So quote '~20 tokens per parameter' as a rule of thumb, not a derived constant.")

    header("Part B: how far real models sit from ~20 tokens per parameter")
    print(f"  {'model':<16} {'params':>8} {'tokens':>8} {'tokens/param':>13} {'vs 20':>8}")
    for name, N, D in [("Chinchilla", 70e9, 1.4e12), ("Llama 3 405B", 405e9, 15.6e12), ("Llama 3 8B", 8e9, 15e12)]:
        print(f"  {name:<16} {N / 1e9:>7.0f}B {D / 1e12:>7.1f}T {D / N:>13,.0f} {D / N / 20:>7.0f}x")

    header("Part C: when does inference cost pass training cost?")
    D = 15e12
    print("  ratio = (2 N T) / (6 N D) = T / (3 D). Model size N cancels out.")
    print(f"  with D = 15T the break-even is T = 3D = {3 * D / 1e12:.0f} trillion tokens served")
    print(f"\n  {'tokens served T':>16} {'inference / training':>22}")
    for T in (5e12, 15e12, 45e12, 150e12, 450e12):
        print(f"  {T / 1e12:>14.0f}T {T / (3 * D):>22.2f}")
    print("\n  A model that is widely deployed passes the threshold, so it pays to train a")
    print("  SMALLER model on FAR MORE data: slightly worse for the training bill, much")
    print("  cheaper for everyone who runs it afterwards.")


if __name__ == "__main__":
    main()
