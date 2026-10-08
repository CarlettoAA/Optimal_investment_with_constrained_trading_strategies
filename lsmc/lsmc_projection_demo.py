"""
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from scipy.stats import norm

# GLOBAL VARIABLES
# Parameters
r = 0.05  # Interest rate
mu = 0.07  # Expected return (10%)
sigma = 0.2  # Volatility (20%)
delta = 0.1  # dividends rate
S0 = 100  # Initial stock price
S0 = 1
u = 9  # concavity param
rho = 1000  # special value
K = 110  # strike price
K = 1.1
T = 0.5  # Time horizon
#N = 40  # Number of time steps
N = 10 # easier
L = 10  # Number of simulated paths
D = 5  # Value for Basis function; not relevant to be defined
n = 0  # just for example
q = 1  # just for example



def simulate_gbm(S0, mu, sigma, T, N, M, W):
    dt = T / (N - 1)
    t = np.linspace(0, T, N)
    S = np.zeros((M, N))
    S[:, 0] = S0

    for i in range(1, N):
        S[:, i] = S[:, i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * W[:, i - 1])

    return t, S


def generator(t, s, y, z, weight):
    return r * y + (mu - r)*z/sigma


def G_option(s, delta, K, T):
    return np.maximum(s * np.exp(-delta * T) - K, 0)  # CALL

# --- Define the basis functions ---
def basis_1(x): return np.ones_like(x)
def basis_2(x): return x - S0
def basis_3(x): return (x - S0) ** 2
def basis_4(x): return (x - S0) ** 3
def basis_5(x): return (x - S0) ** 4  # added by me
def basis_6(x): return (x - S0) ** 5  # added by me
def basis_7(x): return (x - S0) ** 6  # added by me


# --- Evaluate the basis functions on simulated paths ---
basis_functions = [
    lambda x: basis_1(x),
    lambda x: basis_2(x),
    lambda x: basis_3(x),
    lambda x: basis_4(x),
    # lambda x: basis_5(x),
    # lambda x: basis_6(x),
    # lambda x: basis_7(x),
    lambda x: G_option(x, delta, K, T)
]


def orthonormalize_basis(S_tj, basis_functions):
    # Evaluate all basis functions on S_tj
    Phi = np.column_stack([f(S_tj) for f in basis_functions])  # Shape: (L, K)

    # SVD
    U, s, Vt = svd(Phi, full_matrices=False)

    # Retain columns corresponding to non-negligible singular values
    kappa_j = np.sum(s > 1e-10)
    Phi_orth = U[:, :kappa_j] * np.sqrt(L)  # scale to satisfy ⟨ψ_i, ψ_j⟩ = δ_ij = phi * phi / L (empirical scalar product)

    return Phi_orth  # that is L X kappa_j


if __name__ == "__main__":

    # Set seed for reproducibility
    np.random.seed(28)

    # Simulations
    W = np.random.standard_normal((L, N - 1)) * np.sqrt(T / (N - 1))

    t, S = simulate_gbm(S0, mu, sigma, T, N, L, W)

    # the list that will contain all the final converging values
    Y_0_final = [0]
    max_iter = 500

    list_of_weights = [1 * i for i in range(0, 5)]  # [0, 10, 20, 30, 40, 50, ...]
    # list_of_weights = [5*i for i in range(12,13)]
    # list_of_weights = [61]
    # list_of_weights = [0, 1, 2, 3, 5, 10, 20, 30, 40, 50]

    for num_weight in list_of_weights:

        print("weight number:", num_weight)

        # now we iterate and we do not stop untile the condition at time 0 is satisfied.
        condition = False
        n_iter = 0

        b_vec = np.zeros((1, L, N))  # (iteration, simulation, time step)
        Y_j = np.zeros((1, L, N))  # first dimension for convergence, second for simulation
        Z_j = np.zeros((1, L, N))  # first dimension for convergence, second for simulation

        while condition == False:
            n_iter += 1
            # print("This is iteration number:", n_iter)
            # if n_iter % 50 == 0 or n_iter == 1: print("This is iteration number:", n_iter)

            new_layer = np.zeros((1, L, N))
            b_vec = np.concatenate((b_vec, new_layer), axis=0)  # creating space for a new iteration
            Y_j = np.concatenate((Y_j, new_layer), axis=0)  # creating space for a new iteration
            Z_j = np.concatenate((Z_j, new_layer), axis=0)  # creating space for a new iteration
            b_vec[n_iter, :, N - 1] = G_option(S[:, -1], delta, K, T)  # It has to be like this for each iteration

            # to compute the rest of b_vec
            for j in range(N - 2, -1, -1):  # python loops excludes the stop value (-1)
                delta_j = t[j + 1] - t[j]

                # Version with vectors
                sum_generator = np.zeros(L)
                for i in range(j, N - 1):  # check the indexes
                    sum_generator = sum_generator + generator(t[i], S[:, i], Y_j[n_iter - 1, :, i],
                                                              Z_j[n_iter - 1, :, i], num_weight) * delta_j

                b_vec[n_iter, :, j] = b_vec[n_iter, :, N - 1] - sum_generator

                # for j in range(0, N-1):

                delta_j = t[j + 1] - t[j]

                # Optimize version
                Phi = orthonormalize_basis(S[:, j], basis_functions)  # shape: (L, kappa_j)

                # Project b_vec onto orthonormal basis at time j and j+1
                B_Y = Phi.T @ b_vec[-1, :, j] / L  # shape: (kappa_j,)

                dW_j = W[:, j]
                b_adj = b_vec[-1, :, j + 1] * dW_j / delta_j
                B_Z = Phi.T @ b_adj / L  # shape: (kappa_j,)

                # Reconstruct the projections
                Y_result_vec = Phi @ B_Y  # shape: (L,)
                Z_result_vec = Phi @ B_Z  # shape: (L,)

                approx_phi =  Phi.T @ Phi / L

                print("b_vec", b_vec[-1, :, j])
                print("projection B_Y", B_Y)
                print("approx", Y_result_vec)
                print("")
                # Store results
                Y_j[n_iter, :, j] = Y_result_vec
                Z_j[n_iter, :, j] = Z_result_vec
            #######################################################################

            # Now I have everything until j = 0.
            # to check condition for iteration
            if n_iter >= 1:
                if np.abs(np.mean(Y_j[n_iter - 1, :, 0]) - np.mean(Y_j[n_iter, :, 0])) <= 0.0001:
                    condition = True
                    Y_0_final.append(np.mean(Y_j[n_iter, :, 0]))
                    print("Setting condition to True. Iteration n:", n_iter)
                    print("Y at time t=0:\n", np.mean(Y_j[n_iter, :, 0]))
                    # just to test no shortselling constraint:
                    # print("Number of Z < 0:", np.sum(Z_j[-1, :, :] < 0), "over", np.sum(Z_j[-1, :, :] < 0) + np.sum(Z_j[-1, :, :] >= 0) )
                else:
                    print("n_iter = ", n_iter, "; Y_0 value = ", np.mean(Y_j[n_iter, :, 0]))
                    # print("Number of Z < 0:", np.sum(Z_j[-1, :, :] < 0), "over", np.sum(Z_j[-1, :, :] < 0) + np.sum(Z_j[-1, :, :] >= 0) )

            # to break in case of errors:
            if n_iter >= max_iter:
                max_iter = max_iter * 2
                condition = True
                Y_0_final.append(np.mean(Y_j[n_iter, :, 0]))
                print("NO CONVERGENCE, CHECK RESULTS, APPROXIMATIVE VALUE FOR Y")









"""

import numpy as np
import matplotlib.pyplot as plt

# 1) Simula dati Monte Carlo
np.random.seed(0)
L = 600                         # numero di campioni
X = np.random.uniform(-2, 2, L) # "stato" simulato
noise = 0.3 * np.random.randn(L)
Y = np.sin(X) + noise           # target (valore futuro/continua.) con rumore

# 2) Costruisci le funzioni base: Phi = [1, X, X^2]
Phi = np.column_stack([np.ones_like(X), X, X**2])  # (L x D), D=3

# 3) Proiezione L2 (minimi quadrati): Y_hat = Phi @ beta
#    beta = argmin ||Phi beta - Y||^2  ->  usa lstsq per stabilità
beta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
Y_hat = Phi @ beta

# 4) (Facoltativo) Verifica ortogonalità del residuo allo spazio delle basi
res = Y - Y_hat
print("||Phi^T (Y - Y_hat)|| ≈", np.linalg.norm(Phi.T @ res))

# 5) Visualizza: punti (X,Y) e curva stimata (proiezione)
xs = np.linspace(-2, 2, 400)
Phi_s = np.column_stack([np.ones_like(xs), xs, xs**2])
Y_hat_s = Phi_s @ beta


# 2) Costruisci le funzioni base: Phi = [1, X, X^2]
Phi2 = np.column_stack([np.ones_like(X), X, X**2, X**3, X**4])  # (L x D), D=3

# 3) Proiezione L2 (minimi quadrati): Y_hat = Phi @ beta
#    beta = argmin ||Phi beta - Y||^2  ->  usa lstsq per stabilità
beta2, *_ = np.linalg.lstsq(Phi2, Y, rcond=None)
Y_hat2 = Phi2 @ beta2

# 4) (Facoltativo) Verifica ortogonalità del residuo allo spazio delle basi
res2 = Y - Y_hat2
print("||Phi^T (Y - Y_hat)|| ≈", np.linalg.norm(Phi2.T @ res2))

# 5) Visualizza: punti (X,Y) e curva stimata (proiezione)
xs = np.linspace(-2, 2, 400)
Phi_s2 = np.column_stack([np.ones_like(xs), xs, xs**2, xs**3, xs**4])
Y_hat_s2 = Phi_s2 @ beta2

plt.figure(figsize=(7,4))
plt.scatter(X, Y, s=12, alpha=0.5, label="Samples (X, Y)")
plt.plot(xs, np.sin(xs), linestyle='--', label="True E[Y|X]=sin(X)")
plt.plot(xs, Y_hat_s, label="Projection (LSMC) on span{1,X,X^2}")
plt.plot(xs, Y_hat_s2, label="Projection (LSMC) on span{1,X,X^2,X^3,X^4}")
plt.xlabel("X"); plt.ylabel("Y / E[Y|X]")
plt.title("Least-Squares Projection: Y onto span{basis(X)}")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()
