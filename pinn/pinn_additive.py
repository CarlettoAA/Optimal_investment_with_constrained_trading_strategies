# === GLOBAL VARIABLES ===
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D  # abilita 3D plotting

#np.random.seed(5)
#tf.random.set_seed(5)

r = 0.05
mu = 0.07
sigma = 0.2
delta = 0.1
K = 1.1    # just to try, should be 110
#T = 0.5
T = tf.constant(0.5, dtype=tf.float32)
S0 = 1      # just to try, should be 100
N_steps, N_paths = 10, 100000
p = 2
epochs = 25000
dt = T / N_steps
t_min, t_max = 0.0, T
x_min, x_max = 0.0, 2 # original scale = 200
weight = 3
concav = tf.constant(9, dtype=tf.float32)
q = tf.constant(0.25, dtype=tf.float32)    #0.0, 0.25, 0.5, 0.75, 1
rho = tf.constant(1000, dtype=tf.float32)
corrector = tf.constant(100, dtype=tf.float32)
epsilon = tf.constant(1e-8, tf.float32)
eps = 0.01
N_neurons, N_layers = 256, 4
batch_size = 400


# === FUNCTIONS ===
def simulate_gbm_paths(S0, mu, sigma, T, N_steps_f, N_paths_f, W):
    # stock pays dividends: S_t = S_t*((mu-delta)dt + sigma*dW_t) (we are under P, real risk measure)
    t_values = np.linspace(0, T, N_steps_f + 1)
    S = np.zeros((N_paths_f, N_steps_f + 1))
    S[:, 0] = S0
    for i in range(1, N_steps_f + 1):
        Z = W[:, i - 1]
        # S[:, i] = S[:, i - 1] * np.exp((mu - delta - 0.5 * sigma ** 2) * dt + sigma * Z)
        S[:, i] = S[:, i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * Z)

    t_grid = np.tile(t_values, (N_paths_f, 1))
    return tf.convert_to_tensor(t_grid, dtype=tf.float32), tf.convert_to_tensor(S, dtype=tf.float32)


def terminal_condition(x):
    # return tf.maximum(x*tf.exp(-delta * T) - K, 0.0)
    return eps * tf.math.log(1 + tf.exp(((x - 3 * eps) * tf.exp(-delta * T) - K) / eps))  # SMART CALL CORRECTED
    # return eps * tf.math.log(1 + tf.exp(((x)  * tf.exp(-delta * T) - K) / eps)) # SMART CALL


def U_star(model, t, x):
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    U_val = model(tf.concat([t, x], axis=1))
    g_val = terminal_condition(x)

    return (T - t) * U_val + g_val


def black_scholes_call_div(S, K, t, T, r, delta, sigma):
    # it requires risk neutral measure. I must use it only as reference value for t=0.
    Tt = T - t
    d1 = (np.log(S / K) + (r - delta + 0.5 * sigma ** 2) * Tt) / (sigma * np.sqrt(Tt))
    d2 = d1 - sigma * np.sqrt(Tt)
    call_price = S * np.exp(-delta * Tt) * norm.cdf(d1) - K * np.exp(-r * Tt) * norm.cdf(d2)
    put_price = K * np.exp(-r * Tt) * norm.cdf(-d2) - S * np.exp(-delta * Tt) * norm.cdf(-d1)
    return call_price


def price_call_at_given_time(S_np, W, t):
    # I must price using RN derivative since I am under the measure P
    lambda_ = (mu - r) / sigma
    # payoff = np.maximum(S_np[:, -1] - K, 0)
    payoff = np.maximum(S_np[:, -1] * np.exp(-delta * 0.5) - K, 0)
    # Determina l'indice j corrispondente a t
    j = int(np.round(t / dt))
    j = min(j, N_steps)  # sicurezza per evitare overflow

    # Browniano fino a t
    if j == 0:
        W_t = np.zeros(len(W))  # np.zeros(N_paths)
    else:
        W_t = np.cumsum(W, axis=1)[:, j - 1]

    # Browniano fino a T
    W_T = np.cumsum(W, axis=1)[:, -1]
    # Incremente dW da t a T
    dW = W_T - W_t
    tau = T - j * dt

    rn_derivative = np.exp(-lambda_ * dW - 0.5 * lambda_ ** 2 * tau)
    price = np.exp(-r * tau) * (payoff * rn_derivative)
    return price



def f_con(x, concav, q, rho):

    # maschere (intervalli disgiunti)
    mask1 = x <= 0.0
    mask2 = tf.logical_and(x > 0.0, x <= 1.0)
    mask3 = tf.logical_and(x > 1.0, x <= rho)
    mask4 = x > rho

    base_lin = x / corrector  # corretto >0, quindi nessuna /0

    # y1, y2 semplici
    y1 = tf.zeros_like(x)
    y2 = concav * base_lin

    # y3: potenza frazionaria SOLO dove mask3; altrove base=1 (neutra)
    base_pow = tf.where(mask3, base_lin, tf.ones_like(base_lin))
    base_pow = tf.maximum(base_pow, 0.0) + epsilon         # evita base <0 e 0^p
    y3 = concav * tf.math.pow(base_pow, q + 1.0)

    # y4: rho^q con base non negativa
    rho_safe = tf.maximum(rho, 0.0) + epsilon
    y4 = concav * base_lin * tf.math.pow(rho_safe, q)

    # composizione piecewise
    y = tf.where(mask1, y1,
         tf.where(mask2, y2,
         tf.where(mask3, y3, y4)))
    return y

def f_con_q0(x, concav): # only for q=0
    # 0 per x<=0, concav*x per x>0
    return concav * tf.nn.relu(x)


def f_frac(x): # precision e-6
    precision = 1e1
    x_scaled_3 = tf.floor(x * precision) * (1e6 - precision)
    x_scaled_6 = tf.floor(x * 1e6)
    frac_part = x_scaled_6 - x_scaled_3
    norm_frac_part = frac_part / (1e6 - precision) # to make it between 0 and 1
    return tf.minimum(norm_frac_part, 1.0 - norm_frac_part)

def f_frac_N(x):
    x_scaled_1 = tf.floor(x * 1e6)
    x_scaled_2 = tf.floor(x) * 1e6
    frac_part = x_scaled_1 - x_scaled_2
    norm_frac_part = frac_part / 1e6 # to make it between 0 and 1
    return tf.minimum(norm_frac_part, 1.0 - norm_frac_part)

def f_digits(x):
    # porta le cifre dalla 4ª alla 6ª in posizione intera
    x_scaled = tf.floor(x * 1e6)

    # prendi modulo 1000 → ultime 3 cifre (cioè dalla 4ª alla 6ª)
    last_3_digits = tf.cast(tf.math.floormod(x_scaled, 1000), tf.float32)

    return last_3_digits/100


def no_short_pen(z): #it works
    return tf.maximum(-z,0.0)

def no_borrowing(z, y): # it works
    return tf.maximum(z - y, 0.0)


def draw_vertical_density(ax, t0, xvals, y_min, y_max,
                          width_frac=0.02, bins=40, smooth_sigma=1.0):
    """
    Disegna una piccola curva di densità orizzontale a destra di t0 usando un
    istogramma normalizzato (no SciPy). L’area tra t0 e la curva viene riempita.

    - t0: posizione temporale (float)
    - xvals: array dei valori di x a quel t
    - y_min, y_max: limiti verticali (asse x del modello)
    - width_frac: larghezza orizzontale della curva in frazione della larghezza dell'asse t
    - bins: numero di bin per la densità
    - smooth_sigma: smoothing gaussiano della densità (0 = nessuno)
    """

    # istogramma densità
    counts, edges = np.histogram(xvals, bins=bins, range=(y_min, y_max), density=True)
    y = 0.5 * (edges[:-1] + edges[1:])
    d = counts.astype(float)

    # smoothing gaussiano semplice via convoluzione
    if smooth_sigma > 0:
        ksize = max(3, int(6 * smooth_sigma) + 1)
        xk = np.linspace(-3, 3, ksize)
        kernel = np.exp(-0.5 * xk ** 2)
        kernel /= kernel.sum()
        d = np.convolve(d, kernel, mode='same')

    # normalizza e scala a piccola larghezza orizzontale
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    width = width_frac * x_range
    d_norm = d / (np.max(d) + 1e-12)
    x_curve = t0 + d_norm * width

    # disegna curva + riempimento
    ax.plot(x_curve, y, linewidth=1.5, color='red', zorder=3)
    ax.fill_betweenx(y, t0, x_curve, alpha=0.25, color='red', zorder=3)


# === MODEL ===
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = [tf.keras.layers.Dense(N_neurons, activation='tanh', kernel_initializer='glorot_uniform') for _ in
                       range(N_layers)]  # try relu or tanh
        self.out = tf.keras.layers.Dense(1)

    def call(self, X):
        x = X
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)


# === LOSS FUNCTION ===
@tf.function
def loss_fn_bsde(model, t_grid, x_grid, W, K=K, delta=delta, p=p):
    h = dt
    PDE_loss = 0.0
    Const_loss = 0.0
    Total_w_loss = 0.0
    PDE_loss_i = []
    lambda_i = tf.zeros(shape=(10,))
    eps = 0.0  # valore di fallback per le voci non finite (puoi mettere un piccolo numero es. 1e-8)
    clip_val = 1e3  # clipping opzionale delle derivate (None per disattivare)

    for i in range(N_steps):
        ti = t_grid[:, i:i + 1]
        Xi = x_grid[:, i:i + 1]
        Xi_next = x_grid[:, i + 1:i + 2]
        dW_i = W[:, i:i + 1]  # W(i+1) - W(i)

        # generalized transformation
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([ti, Xi])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([ti, Xi])
                U_star_i = U_star(model, ti, Xi)
            DU_star_i = tape1.gradient(U_star_i, Xi)
        D2U_star_i = tape2.gradient(DU_star_i, Xi)
        del tape1, tape2

        U_star_next = U_star(model, t_grid[:, i + 1:i + 2], Xi_next)

        """
        # Se una derivata è None, rimpiazzala con zeri della stessa shape
        if DU_star_i is None:
            DU_star_i = tf.zeros_like(Xi, dtype=U_star_i.dtype)
        if D2U_star_i is None:
            D2U_star_i = tf.zeros_like(Xi, dtype=U_star_i.dtype)

        # Rimpiazza NaN/Inf con eps mantenendo i valori buoni
        U_star_i  = tf.where(tf.math.is_finite(U_star_i),  U_star_i,  tf.fill(tf.shape(U_star_i),  tf.cast(eps, U_star_i.dtype)))
        U_star_next  = tf.where(tf.math.is_finite(U_star_next),  U_star_next,  tf.fill(tf.shape(U_star_next),  tf.cast(eps, U_star_next.dtype)))
        DU_star_i  = tf.where(tf.math.is_finite(DU_star_i),  DU_star_i,  tf.fill(tf.shape(DU_star_i),  tf.cast(eps, DU_star_i.dtype)))
        D2U_star_i = tf.where(tf.math.is_finite(D2U_star_i), D2U_star_i, tf.fill(tf.shape(D2U_star_i), tf.cast(eps, D2U_star_i.dtype)))

        # Clipping opzionale per stabilizzare
        if clip_val is not None:
            U_star_next  = tf.clip_by_value(U_star_next,  -clip_val, clip_val)
            U_star_i  = tf.clip_by_value(U_star_i,  -clip_val, clip_val)
            DU_star_i  = tf.clip_by_value(DU_star_i,  -clip_val, clip_val)
            D2U_star_i = tf.clip_by_value(D2U_star_i, -clip_val, clip_val)
        #"""

        # + tf.maximum(DU_star_i - U_star_i - f_con(U_star_i, concav, q, rho), 0.0)
        # + tf.nn.relu(DU_star_i*corrector - U_star_i*corrector - f_con(U_star_i*corrector, concav, q, rho))/corrector
        # +tf.nn.relu(DU_star_i - U_star_i - f_con_q0(U_star_i, concav)) #not working properly
        # + tf.nn.relu(DU_star_i - (concav + 1)* tf.nn.relu(U_star_i))
        # + tf.maximum(DU_star_i - (U_star_i + f_con(U_star_i*corrector, concav, q, rho)), 0.0 )*500  it goes to 0 but the solution is still too high
        # + tf.nn.relu(DU_star_i - (U_star_i + f_con(U_star_i*corrector, concav, q, rho)))/9
        # + f_frac_N(DU_star_i) With N ten, weight = 1 u* = 0.26
        # *tf.nn.relu(DU_star_i) NO LONG
        # ORIGINAL
        L_i = (
                U_star_next - U_star_i
                + (r * U_star_next + (mu - r) * DU_star_i + f_frac_N(DU_star_i)) * h  # or U_star_next
                - DU_star_i * sigma * dW_i
                - (mu) * D2U_star_i * sigma * h * dW_i
                - 0.5 * dW_i * sigma * D2U_star_i * sigma * dW_i
                + 0.5 * sigma ** 2 * D2U_star_i * h
            # let us add penalization here
        )
        # """

        PDE_loss_i.append(tf.reduce_mean(tf.abs(L_i) ** p))
        PDE_loss += tf.reduce_mean(tf.abs(L_i) ** p)
        Const_loss += tf.reduce_mean(f_frac_N(DU_star_i) ** p)

    XT = x_grid[:, -1:]
    tT = tf.ones_like(XT) * T
    payoff = terminal_condition(XT)
    U_star_T = U_star(model, tT, XT)
    term_loss = tf.reduce_mean(tf.abs(U_star_T - payoff) ** p)

    L_p = h ** (p / 2 - 1) * term_loss + PDE_loss

    # if tf.reduce_any(tf.math.is_nan(L_p)): tf.print("ERRORE. U* =", U_star_i, "DU* =", DU_star_i, "D2U* =", D2U_star_i)

    weighted_loss = L_p / (L_p + (Const_loss + 0.0001)) * Const_loss + (Const_loss + 0.0001) / (
                L_p + (Const_loss + 0.0001)) * L_p

    # return weighted_loss, PDE_loss_i, Const_loss # to get loss without weighted loss
    return L_p, PDE_loss_i, Const_loss  # to get loss without weighted loss


@tf.function
def train_step(model, t_grid, x_grid, W):
    with tf.GradientTape() as tape:
        loss, term_loss, PDE_loss = loss_fn_bsde(model, t_grid, x_grid, W)

    grads = tape.gradient(loss,
                          model.trainable_variables)  # usare la loss calcolata con U_star per calcolare i gradienti di U. KEY POINT
    # Gradient clipping applicato qui (per q=0.25, 0.5, 0.75)
    # grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # ottimizare U
    return loss, term_loss, PDE_loss


# === MAIN ===
def main():
    # print(f"Parameters for penalization: q = {q}, weight = {weight}")
    print(f"Network architecture: Neurons = {N_neurons}, layers = {N_layers}")
    print(f"Simulation: steps = {N_steps}, paths = {N_paths}")

    mean = 0
    bs_value_1 = black_scholes_call_div(S0, K, 0, T, r, delta, sigma)
    print("bs_call: ", bs_value_1)

    W = tf.convert_to_tensor(np.random.randn(N_paths, N_steps) * np.sqrt(dt), dtype=tf.float32)
    t_grid, x_grid = simulate_gbm_paths(S0, mu, sigma, T, N_steps, N_paths, W)
    # Converti a numpy per il plot
    t_grid_np = t_grid.numpy()
    S_np = x_grid.numpy()
    T_final = 0.5

    print("vediamo:", np.mean(price_call_at_given_time(S_np, W.numpy(), 0.1)))
    print("vediamo:", np.mean(price_call_at_given_time(S_np, W.numpy(), 0.2)))
    print("vediamo:", np.mean(price_call_at_given_time(S_np, W.numpy(), 0.3)))
    print("vediamo:", np.mean(price_call_at_given_time(S_np, W.numpy(), 0.4)))
    print("vediamo:", price_call_at_given_time(S_np, W.numpy(), 0.2))
    print("Sample averate at T:", S_np[:, -1].mean())

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))  # 1 riga, 2 colonne

    # --- First plot: GBM Paths ---
    for i in range(np.minimum(100, N_paths)):
        axs[0].plot(t_grid_np[i], S_np[i], alpha=0.8)
    axs[0].set_title('Simulated GBM Paths')
    axs[0].set_xlabel('Time $t$')
    axs[0].set_ylabel('Asset Price $S_t$')
    axs[0].grid(True)

    # --- Second plot: Histogram of x_grid[:, -1] ---
    axs[1].hist(x_grid[:, -1].numpy(), bins=50, color='steelblue', edgecolor='black')
    axs[1].set_xlabel("x at final time step")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title("Histogram of x_grid[:, -1]")

    plt.tight_layout()
    plt.show()

    # modified learning rate
    lr_schedule_exp = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=3000,
        decay_rate=0.7)

    """
    # Definizione dello scheduler
    step_size = 5000
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[step_size * i for i in range(1, epochs // step_size)],
    values=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    """

    model = PINN()
    global optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_exp)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # fixed learning rate

    list_PDE_loss_i = []
    list_pde_loss = []
    list_loss = []

    idx = np.random.choice(t_grid.shape[0], size=batch_size, replace=False)
    # Estrai i batch
    t_batch = tf.gather(t_grid, idx)
    x_batch = tf.gather(x_grid, idx)
    W_batch = tf.gather(W, idx)

    #############################################################################################################################################
    #############################################################################################################################################

    for epoch in range(epochs + 1):

        if epoch >= 7000 & epoch % 2000 == 0:
            # new batch
            # Estrai indici casuali per il batch per ogni epoca
            idx = np.random.choice(t_grid.shape[0], size=batch_size, replace=False)
            # Estrai i batch
            t_batch = tf.gather(t_grid, idx)
            x_batch = tf.gather(x_grid, idx)
            W_batch = tf.gather(W, idx)
        # """

        loss, PDE_loss_i, PDE_loss = train_step(model, t_batch, x_batch, W_batch)
        # loss, PDE_loss_i, PDE_loss = train_step(model, t_grid, x_grid, W,)
        if epoch % 100 == 0 or epoch == epochs:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4e}, Constraint L^p: {PDE_loss.numpy():.4e}")
            PDE_loss_i_numpy = [x.numpy() for x in PDE_loss_i]
            list_PDE_loss_i.append(PDE_loss_i_numpy)
            list_pde_loss.append(PDE_loss.numpy())
            list_loss.append(loss.numpy())

    #############################################################################################################################################
    #############################################################################################################################################

    # Trasposizione: ora abbiamo N_steps liste da 25 elementi (una per ogni i)
    transposed = list(map(list, zip(*list_PDE_loss_i)))
    # Plotting loss over epochs
    plt.figure(figsize=(8, 5))
    for i, loss_i in enumerate(transposed):
        plt.plot(loss_i, label=f"Time_{i}")
    plt.plot(list_pde_loss, label='PDE Loss')
    plt.plot(list_loss, label='Total Loss', linestyle='--')
    plt.xlabel("Epochs (x100)")
    plt.ylabel("Losses")
    plt.title("Evolution of losses over epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plotting temporary losses
    # Trasposizione: ora abbiamo N_steps liste da 25 elementi (una per ogni i)
    transposed = list(map(list, zip(*list_PDE_loss_i)))
    plt.figure(figsize=(8, 5))
    for i, loss_i in enumerate(transposed):
        plt.plot(loss_i, label=f"Time_{i}")
    plt.xlabel("Epochs (x100)")
    plt.ylabel("Losses for each time")
    plt.title("Evolution of losses over epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # =============================================================================================
    # ========================= Original ==========================================================
    # =============================================================================================
    # === Predizione al tempo t = 0 ===
    x_min_tf = tf.reduce_min(x_grid)
    x_max_tf = tf.reduce_max(x_grid)
    x0 = np.linspace(x_min_tf.numpy(), x_max_tf.numpy(), 200).reshape(-1, 1)
    t0 = np.zeros_like(x0)
    X0 = np.hstack((t0, x0))

    # U (not needed)
    u0_pred = model(X0).numpy()
    uT_pred = model(np.hstack((np.ones_like(x0) * T, x0))).numpy()

    # U_star
    u0_star_pred = U_star(model, t0, x0)
    uT_star_pred = U_star(model, np.ones_like(t0) * T, x0)

    # prediction at time 0 (to check initial value)
    x_ref = tf.reduce_min(x_grid[:, 0]).numpy()
    bs_value_1 = black_scholes_call_div(x_ref, K, 0, T, r, delta, sigma)
    bs_empirical = mean = np.mean(price_call_at_given_time(S_np, W.numpy(), 0))
    plt.figure(figsize=(8, 4))
    plt.plot(x0, u0_star_pred, label='predicted u(0, x)*')
    # plt.axhline(y = bs_value_1, color='red', linestyle='--', label=f'BS call price {bs_value_1}')
    plt.scatter(x_ref, bs_value_1, color='green', label=f'BS call price {bs_value_1:.6f}')
    plt.scatter(x_ref, bs_empirical, color='blue', label=f'Empirical BS call price {bs_empirical:.6f}')
    plt.scatter(np.linspace(x_min_tf.numpy(), x_max_tf.numpy(), 200),
                black_scholes_call_div(np.linspace(x_min_tf.numpy(), x_max_tf.numpy(), 200), K, 0, T, r, delta, sigma),
                color='yellow', label="BS_prices")
    plt.xlabel('x')
    plt.ylabel('u(0, x)*')
    plt.title('PINN at time t=0')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # U_star
    u_star_single_pred = U_star(model, np.array([[0.0]]), np.array([[1.0]]))[0, 0]
    print(f"Predicted u*(0, 100): {u_star_single_pred:.6f}")

    # prediction at time t=0.1
    step_val = int(0.2 * N_steps)
    t_intermediate = step_val * dt
    x_int = tf.convert_to_tensor(
        np.linspace(tf.reduce_min(x_grid[:, step_val]), tf.reduce_max(x_grid[:, step_val]), 100)[:, None],
        dtype=tf.float32)
    t_int = tf.convert_to_tensor(np.ones_like(x_int) * t_intermediate, dtype=tf.float32)
    # Predizione U_star (non U!)
    U_star_int = U_star(model, t_int, x_int).numpy()
    # Plot
    plt.figure(figsize=(8, 4))
    plt.axvline(x=tf.reduce_min(x_grid[:, step_val]).numpy(), color='red', linestyle='--',
                label=f'x_min at t= {t_intermediate:.3f}')
    plt.axhline(
        y=black_scholes_call_div(tf.reduce_min(x_grid[:, step_val]).numpy(), K, t_intermediate, T, r, delta, sigma),
        color='blue', linestyle='--', label='BS call price')
    plt.axvline(x=tf.reduce_max(x_grid[:, step_val]).numpy(), color='red', linestyle='--',
                label=f'x_max at t= {t_intermediate:.3f}')
    plt.axhline(
        y=black_scholes_call_div(tf.reduce_max(x_grid[:, step_val]).numpy(), K, t_intermediate, T, r, delta, sigma),
        color='blue', linestyle='--', label='BS call price')
    plt.scatter(x_grid[:, step_val].numpy(),
                black_scholes_call_div(x_grid[:, step_val].numpy(), K, t_intermediate, T, r, delta, sigma),
                color='yellow', label="BS_prices")
    plt.plot(x_int, U_star_int, label=f"U* at t = {t_intermediate:.3f}")
    plt.xlabel("x")
    plt.ylabel("U*")
    plt.title(f"PINN's prediction at t = {t_intermediate:.3f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # prediction at time t=0.2
    step_val = int(0.4 * N_steps)
    t_intermediate = step_val * dt
    x_int = tf.convert_to_tensor(
        np.linspace(tf.reduce_min(x_grid[:, step_val]), tf.reduce_max(x_grid[:, step_val]), 100)[:, None],
        dtype=tf.float32)
    t_int = tf.convert_to_tensor(np.ones_like(x_int) * t_intermediate, dtype=tf.float32)
    # Predizione U_star (non U!)
    U_star_int = U_star(model, t_int, x_int).numpy()
    # Plot
    plt.figure(figsize=(8, 4))
    plt.axvline(x=tf.reduce_min(x_grid[:, step_val]).numpy(), color='red', linestyle='--',
                label=f'x_min at t= {t_intermediate:.3f}')
    plt.axhline(
        y=black_scholes_call_div(tf.reduce_min(x_grid[:, step_val]).numpy(), K, t_intermediate, T, r, delta, sigma),
        color='blue', linestyle='--', label='BS call price')
    plt.axvline(x=tf.reduce_max(x_grid[:, step_val]).numpy(), color='red', linestyle='--',
                label=f'x_max at t= {t_intermediate:.3f}')
    plt.axhline(
        y=black_scholes_call_div(tf.reduce_max(x_grid[:, step_val]).numpy(), K, t_intermediate, T, r, delta, sigma),
        color='blue', linestyle='--', label='BS call price')
    plt.scatter(x_grid[:, step_val].numpy(),
                black_scholes_call_div(x_grid[:, step_val].numpy(), K, t_intermediate, T, r, delta, sigma),
                color='yellow', label="BS_prices")
    plt.plot(x_int, U_star_int, label=f"U* at t = {t_intermediate:.3f}")
    plt.xlabel("x")
    plt.ylabel("U*")
    plt.title(f"PINN's prediction at t = {t_intermediate:.3f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # prediction at time t=0.3
    step_val = int(0.6 * N_steps)
    t_intermediate = step_val * dt
    x_int = tf.convert_to_tensor(
        np.linspace(tf.reduce_min(x_grid[:, step_val]), tf.reduce_max(x_grid[:, step_val]), 100)[:, None],
        dtype=tf.float32)
    t_int = tf.convert_to_tensor(np.ones_like(x_int) * t_intermediate, dtype=tf.float32)
    # Predizione U_star (non U!)
    U_star_int = U_star(model, t_int, x_int).numpy()
    # Plot
    plt.figure(figsize=(8, 4))
    plt.axvline(x=tf.reduce_min(x_grid[:, step_val]).numpy(), color='red', linestyle='--',
                label=f'x_min at t= {t_intermediate:.3f}')
    plt.axhline(
        y=black_scholes_call_div(tf.reduce_min(x_grid[:, step_val]).numpy(), K, t_intermediate, T, r, delta, sigma),
        color='blue', linestyle='--', label='BS call price')
    plt.axvline(x=tf.reduce_max(x_grid[:, step_val]).numpy(), color='red', linestyle='--',
                label=f'x_max at t= {t_intermediate:.3f}')
    plt.axhline(
        y=black_scholes_call_div(tf.reduce_max(x_grid[:, step_val]).numpy(), K, t_intermediate, T, r, delta, sigma),
        color='blue', linestyle='--', label='BS call price')
    plt.scatter(x_grid[:, step_val].numpy(),
                black_scholes_call_div(x_grid[:, step_val].numpy(), K, t_intermediate, T, r, delta, sigma),
                color='yellow', label="BS_prices")
    plt.plot(x_int, U_star_int, label=f"U* at t = {t_intermediate:.3f}")
    plt.xlabel("x")
    plt.ylabel("U*")
    plt.title(f"PINN's prediction at t = {t_intermediate:.3f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # prediction at time t=0.4
    step_val = int(0.8 * N_steps)
    t_intermediate = step_val * dt
    x_int = tf.convert_to_tensor(
        np.linspace(tf.reduce_min(x_grid[:, step_val]), tf.reduce_max(x_grid[:, step_val]), 100)[:, None],
        dtype=tf.float32)
    t_int = tf.convert_to_tensor(np.ones_like(x_int) * t_intermediate, dtype=tf.float32)
    # Predizione U_star (non U!)
    U_star_int = U_star(model, t_int, x_int).numpy()
    # Plot
    plt.figure(figsize=(8, 4))
    plt.axvline(x=tf.reduce_min(x_grid[:, step_val]).numpy(), color='red', linestyle='--',
                label=f'x_min at t= {t_intermediate:.3f}')
    plt.axhline(
        y=black_scholes_call_div(tf.reduce_min(x_grid[:, step_val]).numpy(), K, t_intermediate, T, r, delta, sigma),
        color='blue', linestyle='--', label='BS call price')
    plt.axvline(x=tf.reduce_max(x_grid[:, step_val]).numpy(), color='red', linestyle='--',
                label=f'x_max at t= {t_intermediate:.3f}')
    plt.axhline(
        y=black_scholes_call_div(tf.reduce_max(x_grid[:, step_val]).numpy(), K, t_intermediate, T, r, delta, sigma),
        color='blue', linestyle='--', label='BS call price')
    plt.scatter(x_grid[:, step_val].numpy(),
                black_scholes_call_div(x_grid[:, step_val].numpy(), K, t_intermediate, T, r, delta, sigma),
                color='yellow', label="BS_prices")
    plt.plot(x_int, U_star_int, label=f"U* at t = {t_intermediate:.3f}")
    plt.xlabel("x")
    plt.ylabel("U*")
    plt.title(f"PINN's prediction at t = {t_intermediate:.3f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # prediction at time T (to check the payoff)
    payoffT = np.maximum(np.sort(S_np[:, -1]) * np.exp(-delta * 0.5) - K, 0)
    # payoffT = np.maximum(np.sort(S_np[:, -1]) - K, 0)
    plt.figure(figsize=(8, 4))
    plt.plot(x0, uT_star_pred, label='predicted u(T, x)*')
    plt.plot(np.sort(S_np[:, -1]), payoffT, label='Payoff call', color='red')
    plt.xlabel('x')
    plt.ylabel('u(0, x)')
    plt.title('PINN at time t=T')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # print("Loss at t=T",  np.mean(np.abs(uT_star_pred - payoffT) ** p))
    # print("Value x0 for uT_pred:", [x0, uT_pred])

    # =============================================================================================
    # ========================= Scaled ============================================================
    # =============================================================================================
    # U_star takes t[0,T], x(1). so for original scale: U(t, S) = 100 * U_star(t, S/100)

    # U_star_s
    x0_s = np.linspace(100 * tf.reduce_min(x_grid).numpy(), 100 * tf.reduce_max(x_grid).numpy(), 200).reshape(-1, 1)
    u0_star_s_pred = 100 * U_star(model, t0, x0_s / 100)
    uT_star_s_pred = 100 * U_star(model, np.ones_like(t0) * T, x0_s / 100)
    # bs_empirical = bs_call_empirical_div(S_np[:, -1], K, 0, T, r, delta, sigma) * 100

    # prediction at time 0 (to check initial value)
    x_ref_s = S0 * 100
    bs_value_s = black_scholes_call_div(x_ref_s, K * 100, 0, T, r, delta, sigma)
    bs_empirical = np.mean(price_call_at_given_time(S_np, W.numpy(), 0)) * 100

    # bs_value_s = np.mean(bs_call_price_empirical_P_to_Q(S_np * 100, W, K*100, r, mu, sigma, 0, T_final))
    plt.figure(figsize=(8, 4))
    plt.plot(x0_s, u0_star_s_pred, label='predicted u(0, x)*')
    plt.scatter(x_ref_s, bs_value_s, color='green', label=f'BS call price {bs_value_s:.4f}')
    plt.scatter(x_ref_s, bs_empirical, color='blue', label=f'Empirical BS call price {bs_empirical:.6f}')
    plt.xlabel('x')
    plt.ylabel('u(0, x)*')
    plt.title('PINN at time t=0')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # U_star_s
    u_star_s_single_pred = 100 * U_star(model, np.array([[0.0]]), np.array([[1.0]]))[0, 0]
    print(f"Predicted u*(0, 100): {u_star_s_single_pred:.6f}")

    # ====================================================================================================================
    # ====================================================================================================================

    # final plot (rectangular)
    t_plot, x_plot = np.meshgrid(np.linspace(t_min, T, 100), np.linspace(x_min_tf, x_max_tf, 100))
    X_plot = np.hstack((t_plot.flatten()[:, None], x_plot.flatten()[:, None]))

    u_star_pred = U_star(model, X_plot[:, 0:1], X_plot[:, 1:2]).numpy().reshape(100, 100)
    BS_matrix = np.zeros((100, 100))
    for k in range(100 - 1):
        BS_matrix[:, k] = black_scholes_call_div(np.linspace(x_min_tf, x_max_tf, 100), K, k * T / (100 - 1), T, r,
                                                 delta, sigma)
    # BS_matrix[:, -1] = terminal_condition(np.linspace(x_min_tf, x_max_tf, 100)).numpy()
    BS_matrix[:, -1] = np.maximum(np.linspace(x_min_tf, x_max_tf, 100) * np.exp(-delta * 0.5) - K, 0)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))  # 1 row, 2 columns

    # S_np ha shape (N_paths, N_steps+1)
    N_samples = 10000
    t_cols = np.linspace(0, 0.5, N_steps + 1)  # tempi per ciascuna colonna
    t_plot_pts = np.tile(t_cols, N_samples)  # vettore t flatten
    x_plot_pts = S_np[:N_samples, :].ravel()  # vettore x flatten

    # --- First plot: PINN's solution ---
    c1 = axs[0].contourf(t_plot, x_plot, u_star_pred, 100, cmap='plasma')
    fig.colorbar(c1, ax=axs[0])

    axs[0].scatter(t_plot_pts, x_plot_pts, s=10, c='k', alpha=0.25, linewidths=0, zorder=3, label='Training pts')

    y_min, y_max = float(x_min_tf), float(x_max_tf)
    # ricava i tempi unici (se sono numeri floating identici per colonna va bene)
    t_uni = np.unique(t_plot_pts)

    for t0 in t_uni:
        y_min, y_max = float(np.min(S_np[:, int(t0 * N_steps / 0.5)])), float(np.max(S_np[:, int(t0 * N_steps / 0.5)]))
        mask = np.isclose(t_plot_pts, t0, rtol=0, atol=1e-12)
        x_at_t = x_plot_pts[mask]
        draw_vertical_density(axs[0], float(t0), x_at_t, y_min, y_max, width_frac=0.02, bins=35, smooth_sigma=1.2)

    axs[0].set_xlabel("t")
    axs[0].set_ylabel("x")
    axs[0].set_title("PINN's solution for u(t,x)")

    # --- Second plot: BS
    error = np.abs(BS_matrix)
    c2 = axs[1].contourf(t_plot, x_plot, error, 100, cmap='plasma')
    fig.colorbar(c2, ax=axs[1])
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("x")
    axs[1].set_title("BS Solution")

    plt.tight_layout()
    plt.show()

    #########################################################################################
    #########################################################################################

    # 3D plot
    fig = plt.figure(figsize=(14, 6))  # allarga la figura orizzontalmente
    # U_star
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(t_plot, x_plot, u_star_pred, cmap='plasma', edgecolor='none')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_zlabel('U*')
    ax1.set_title('U* prediction')
    # Black-Scholes
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(t_plot, x_plot, BS_matrix, cmap='plasma', edgecolor='none')
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('u_BS')
    ax2.set_title('Black-Scholes')
    plt.tight_layout()
    plt.show()

    # --- ERROR PLOTS --- #
    fig = plt.figure(figsize=(14, 6))  # Larghezza aumentata per i due subplot
    # --- First subplot: 3D surface ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(t_plot, x_plot, np.abs(u_star_pred - BS_matrix), cmap='viridis', edgecolor='none')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.set_zlabel('Absolute Error')
    ax1.set_title('Error Surface (3D)')

    # --- Second subplot: 2D heatmap ---
    ax2 = fig.add_subplot(1, 2, 2)
    c = ax2.contourf(t_plot, x_plot, np.abs(u_star_pred - BS_matrix), 100, cmap='viridis')
    fig.colorbar(c, ax=ax2)
    ax2.scatter(t_plot_pts, x_plot_pts, s=10, c='k', alpha=0.25, linewidths=0, zorder=3,
                label='Training pts')  # distribution of points
    y_min, y_max = float(x_min_tf), float(x_max_tf)
    # ricava i tempi unici (se sono numeri floating identici per colonna va bene)
    t_uni = np.unique(t_plot_pts)

    for t0 in t_uni:
        y_min, y_max = float(np.min(S_np[:, int(t0 * N_steps / 0.5)])), float(np.max(S_np[:, int(t0 * N_steps / 0.5)]))
        mask = np.isclose(t_plot_pts, t0, rtol=0, atol=1e-12)
        x_at_t = x_plot_pts[mask]
        draw_vertical_density(ax2, float(t0), x_at_t, y_min, y_max, width_frac=0.02, bins=35, smooth_sigma=1.2)
    ax2.set_xlabel("t")
    ax2.set_ylabel("x")
    ax2.set_title("Absolute Error (2D Heatmap)")

    plt.tight_layout()
    plt.show()

    #########################################################################################
    #########################################################################################
    # add a test set here.
    # make a grid and compute MSE (against BS standard)
    # THE GRID IS ALWAYS 100*N_steps, it must be conic

    MSE_i = []
    for interval in range(N_steps):
        t_intermediate = interval * dt
        x_test = tf.convert_to_tensor(
            np.linspace(tf.reduce_min(x_grid[:, interval]), tf.reduce_max(x_grid[:, interval]), 100)[:, None],
            dtype=tf.float32)
        t_test = tf.convert_to_tensor(np.ones_like(x_test) * t_intermediate, dtype=tf.float32)
        # test and correct values
        U_star_test = U_star(model, t_test, x_test).numpy()
        correct_values = black_scholes_call_div(x_test.numpy(), K, t_intermediate, T, r, delta, sigma)
        # MSE for each interval
        MSE_i.append(np.square(U_star_test - correct_values).mean())

    print("Total MSE for a conic grid considering final condition", sum(MSE_i) / (len(MSE_i) + 1))
    print("Total MSE for a conic grid adjusted", sum(MSE_i) / len(MSE_i))
    print("MSE:", MSE_i)

    MSE_i_simulated = []
    ABS_ERROR_i = []
    N_steps_test = N_steps
    N_paths_test = 100
    W = tf.convert_to_tensor(np.random.randn(N_paths_test, N_steps_test) * np.sqrt(dt), dtype=tf.float32)
    t_grid_simu, x_grid_simu = simulate_gbm_paths(S0, mu, sigma, T, N_steps_test, N_paths_test, W)
    S_np_simu = x_grid_simu.numpy()

    for i in range(N_steps):
        U_star_test = U_star(model, t_grid_simu[:, i:i + 1], x_grid_simu[:, i:i + 1]).numpy()
        correct_values = black_scholes_call_div(t_grid_simu[:, i:i + 1].numpy(), K, i * dt, T, r, delta, sigma)
        MSE_i_simulated.append(np.square(U_star_test - correct_values).mean())
        ABS_ERROR_i.append(np.abs(U_star_test - correct_values).mean())

    t_final = tf.ones_like(x_grid_simu[:, N_steps: N_steps + 1]) * T
    U_star_test = U_star(model, t_final, x_grid_simu[:, N_steps: N_steps + 1]).numpy()
    real_terminal_condition = terminal_condition(
        x_grid_simu[:, N_steps: N_steps + 1] + 3 * eps).numpy()  # here NO CORRECTION PAYOFF
    # real_terminal_condition = terminal_condition(x_grid_simu[:, N_steps: N_steps+1]).numpy()   # here NO CORRECTION PAYOFF
    MSE_i_simulated.append(np.square(U_star_test - real_terminal_condition).mean())
    ABS_ERROR_i.append(np.abs(U_star_test - real_terminal_condition).mean())

    # ========== MSE ===========
    # print("Non-zero elements:", np.count_nonzero(np.square(U_star_test - real_terminal_condition)))
    # print(U_star_test - real_terminal_condition)
    print("Total MSE for a simulated paths", sum(MSE_i_simulated) / (len(MSE_i_simulated)))
    print("Total MSE for a simulated paths depending on t", MSE_i_simulated)

    # ========== ABS ERROR ===============
    print("Total Absolute error for a simulated paths", sum(ABS_ERROR_i) / (len(ABS_ERROR_i)))
    print("Total Absolute error for a simulated paths depending on t", ABS_ERROR_i)

    # =============================================================================
    # =============================================================================

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Primo plot: MSE
    axs[0].plot(MSE_i_simulated, marker='o', linestyle='-', color='b')
    axs[0].set_title("MSE over simulated trajectories")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("MSE")
    axs[0].grid(True)

    # Secondo plot: ABS ERROR
    axs[1].plot(ABS_ERROR_i, marker='o', linestyle='-', color='r')
    axs[1].set_title("ABSOLUTE ERROR over simulated trajectories")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("ABS")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


    
