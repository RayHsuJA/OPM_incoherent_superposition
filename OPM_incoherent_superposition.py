# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import gridspec
from matplotlib import colors
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import fftpack

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# %%
# ==========================================
# 1. Core Physics & Signal Functions
# ==========================================


def generate_2dsin(
    nsubs: int,
    nbins: int,
    fp2: float,
    fp3: float,
    Imin: float = 0.0,
    Imax: float = 1.0,
    theta0: float = 0.0,
):
    """
    Generate a 2D sinusoidal pulse stack.
    """
    n = np.arange(nsubs)
    phi = np.linspace(0, 1, nbins, endpoint=False)
    x, y = np.meshgrid(phi, n)

    # 2D Sine wave: (cos + 1)/2 normalized to [0,1]
    norm_wave = (np.cos(2 * np.pi * (y * fp3 - x * fp2) + theta0) + 1) / 2
    pulsestack = Imin + (Imax - Imin) * norm_wave

    return pulsestack


def LRFS(stack: np.ndarray, fft_size: int, zero_DC: bool = True) -> np.ndarray:
    """
    Compute LRFS or PLRFS depending on input shape.

    Args:
        stack (np.ndarray): shape (N_pulses, N_longitudes) or (N_pols, N_pulses, N_longitudes)
    Returns:
        complex spec (np.ndarray): shape (N_blocks, N_pols, N_freqs, N_longitudes)
    """
    if stack.ndim == 2:
        # Single polarization case
        stack_3d = stack[np.newaxis, :, :]
        # print(
        #     "Input is 2D, assumed to be single polarization case with shape (N_pulses, N_longitudes) and automatically reshaped to 3D for processing."
        # )
    elif stack.ndim == 3:
        stack_3d = stack
    else:
        raise ValueError(f"Input stack must be 2D or 3D, got shape {stack.shape}")

    n_pols, n_pulses, n_longitudes = stack_3d.shape

    # 2. Block Calculation and Truncation
    n_blocks = n_pulses // fft_size

    if n_blocks == 0:
        raise ValueError(
            f"fft_size ({fft_size}) cannot be larger than input rows ({n_pulses})."
        )

    cutoff = n_blocks * fft_size

    # 3. Reshape and Transpose to Target Structure
    # Step A: Reshape to split pulses into blocks
    # Intermediate shape: (N_pols, N_blocks, fft_size, N_longitudes)
    stack_4d = stack_3d[:, :cutoff, :].reshape(n_pols, n_blocks, fft_size, n_longitudes)

    # Step B: Transpose to move N_blocks to the front
    # Target shape: (N_blocks, N_pols, fit_size, N_longitudes)
    # Permutation: (1, 0, 2, 3) -> indices of the intermediate shape
    stack_4d = stack_4d.transpose(1, 0, 2, 3)

    # Apply FFT along the time axis (which is now axis 2)
    # Result shape: (N_pol, N_pulses, N_phi)
    complex_spec = fftpack.fftn(stack_4d, axes=2) / fft_size

    # Shift zero frequency to center
    complex_spec = fftpack.fftshift(complex_spec, axes=2)

    complex_spec = np.roll(complex_spec, shift=-1, axis=2)

    # Zero out the DC component
    if zero_DC:
        complex_spec[:, :, fft_size // 2 - 1, :] = 0

    return complex_spec


def sort_modes_by_continuity(m1_raw, m2_raw):
    """
    Sorts/tracks two arrays of complex numbers (modes) based on continuity
    in the complex plane to avoid mode-swapping artifacts.

    This function solves the "eigenvalue crossing" or "mode crossing" problem
    where sorting by magnitude causes non-physical discontinuities (kinks)
    when two modes have similar magnitudes but distinct phases/trajectories.

    Parameters:
    -----------
    m1_raw : array_like
        The first array of complex numbers (e.g., mode 1).
    m2_raw : array_like
        The second array of complex numbers (e.g., mode 2).

    Returns:
    --------
    m1_tracked, m2_tracked : tuple of ndarrays
        The sorted arrays where m1_tracked[i] is the physical continuation
        of m1_tracked[i-1].
    """

    # Ensure inputs are numpy arrays representing complex numbers
    m1 = np.array(m1_raw, dtype=complex)
    m2 = np.array(m2_raw, dtype=complex)

    # Check if input arrays have the same length
    if len(m1) != len(m2):
        raise ValueError("Input arrays m1 and m2 must have the same length.")

    n_points = len(m1)

    # Initialize arrays to store the tracked (sorted) results
    m1_tracked = np.zeros(n_points, dtype=complex)
    m2_tracked = np.zeros(n_points, dtype=complex)

    # Initialize the first point.
    # We assume the starting point (index 0) is physically correct or
    # sufficiently separated. If needed, you can sort index 0 by magnitude here.
    m1_tracked[0] = m1[0]
    m2_tracked[0] = m2[0]

    # Iterate through the arrays to enforce continuity
    for i in range(1, n_points):
        # Retrieve the values from the previous step (the "history")
        prev_m1 = m1_tracked[i - 1]
        prev_m2 = m2_tracked[i - 1]

        # Retrieve the candidates for the current step (raw data)
        # These might be swapped in the raw data due to the solver's internal sorting
        curr_a = m1[i]
        curr_b = m2[i]

        # Calculate the "distance" in the complex plane for both permutations.
        # Ideally, the physical mode changes slowly, so the distance between
        # step[i] and step[i-1] should be minimal.

        # Case 1: Assume no swap occurred (Direct connection)
        # distance = |current_a - prev_m1| + |current_b - prev_m2|
        dist_direct = np.abs(curr_a - prev_m1) + np.abs(curr_b - prev_m2)

        # Case 2: Assume a swap is needed (Cross connection)
        # distance = |current_b - prev_m1| + |current_a - prev_m2|
        dist_swap = np.abs(curr_b - prev_m1) + np.abs(curr_a - prev_m2)

        # Choose the permutation that minimizes the change (distance)
        if dist_direct < dist_swap:
            m1_tracked[i] = curr_a
            m2_tracked[i] = curr_b
        else:
            m1_tracked[i] = curr_b
            m2_tracked[i] = curr_a

    return m1_tracked, m2_tracked


# %%
def perform_spa_analysis(I, Q, U, V, target_fp3):
    """
    Perform Single Pulse Analysis (SPA) / OPM Decomposition at a specific frequency.

    Returns dictionary containing feature vectors: L, A, B, m1, m2, profile
    """
    fft_size = I.shape[0]  # only one block

    # 1. Pulse Profile
    profile = np.mean(I, axis=0)

    # 2. Compute LRFS
    # Stokes I
    L_blocks = LRFS(I, fft_size, zero_DC=False)
    # Stokes P (Q, U, V)
    P_blocks = LRFS(np.stack([Q, U, V]), fft_size, zero_DC=False)

    # 3. Find frequency index k
    freqs = np.linspace(-0.5, 0.5, fft_size + 1)[1:]
    k = np.argmin(np.abs(freqs - target_fp3))

    # 4. Extract complex envelopes at freq k
    L = L_blocks[0, 0, k, :]
    P = P_blocks[0, :, k, :]

    # 5. OPM Decomposition Logic
    phi = 0.5 * np.angle(np.sum(P * P, axis=0))
    A = np.real(P * np.exp(-1j * phi[None, :]))
    B = np.imag(P * np.exp(-1j * phi[None, :]))

    A_abs = np.sqrt(np.sum(A * A, axis=0))
    B_abs = np.sqrt(np.sum(B * B, axis=0))

    # Calculate projection term for m1/m2
    with np.errstate(divide="ignore", invalid="ignore"):
        proj = np.sum(P * A, axis=0) / A_abs
        proj[np.isnan(proj)] = 0.0

    m1 = 0.5 * (L + proj)
    m2 = 0.5 * (L - proj)

    m1, m2 = sort_modes_by_continuity(m1, m2)

    return {
        "profile": profile,
        "L": L,
        "phi": phi,
        "A_abs": A_abs,
        "B_abs": B_abs,
        "m1": m1,
        "m2": m2,
    }


# %%
# ==========================================
# 2. Simulation State
# ==========================================


class SimState:
    def __init__(self):
        self.nsubs = 50
        self.nbins = 200

        # Defaults
        self.noise = 0.05
        self.loc1, self.sig1 = int(self.nbins / 2 * 0.8), int(self.nbins / 10)
        self.loc2, self.sig2 = int(self.nbins / 2 * 1.2), int(self.nbins / 10)
        self.fp3_int = 0.04
        self.analysis_fp3 = self.fp3_int

        # Mode 1 Params
        self.m1_q_min = 0.0
        self.m1_q_max = 1.0
        self.m1_q_th = 0.0
        self.m1_q_f3 = self.fp3_int
        self.m1_q_f2 = 2.0
        self.m1_u_min = 0.0
        self.m1_u_max = 1.0
        self.m1_u_th = 0.0
        self.m1_u_f3 = self.fp3_int
        self.m1_u_f2 = 2.0

        # Mode 2 Params (Inverse sign convention for demo)
        self.m2_q_min = -1.0
        self.m2_q_max = 0.0
        self.m2_q_th = np.pi
        self.m2_q_f3 = self.fp3_int
        self.m2_q_f2 = 2.0
        self.m2_u_min = -1.0
        self.m2_u_max = 0.0
        self.m2_u_th = np.pi
        self.m2_u_f3 = self.fp3_int
        self.m2_u_f2 = 2.0

        # Data storage
        self.res_m1 = {}  # Stores I,Q,U,PA for Mode 1
        self.res_m2 = {}  # Stores I,Q,U,PA for Mode 2
        self.res_sum = {}  # Stores I,Q,U,PA for Sum
        self.analysis = {}  # Stores SPA results

    def _gen_comp(self, min_val, max_val, theta, f3, f2, env):
        s = generate_2dsin(
            self.nsubs,
            self.nbins,
            fp2=f2,
            fp3=f3,
            Imin=min_val,
            Imax=max_val,
            theta0=theta,
        )
        return s * env

    def run(self):
        x = np.arange(self.nbins)
        env1 = np.exp(-(((x - self.loc1) ** 2) / (2 * self.sig1**2)))
        env2 = np.exp(-(((x - self.loc2) ** 2) / (2 * self.sig2**2)))

        # --- Generate Mode 1 ---
        q1 = self._gen_comp(
            self.m1_q_min, self.m1_q_max, self.m1_q_th, self.m1_q_f3, self.m1_q_f2, env1
        )
        q1 += np.random.normal(0, self.noise, q1.shape)

        u1 = self._gen_comp(
            self.m1_u_min, self.m1_u_max, self.m1_u_th, self.m1_u_f3, self.m1_u_f2, env1
        )
        u1 += np.random.normal(0, self.noise, u1.shape)

        v1 = np.random.normal(0, self.noise, (self.nsubs, self.nbins))

        i1 = np.sqrt(q1**2 + u1**2 + v1**2)

        # --- Generate Mode 2 ---
        q2 = self._gen_comp(
            self.m2_q_min,
            self.m2_q_max,
            self.m2_q_th + np.pi,
            self.m2_q_f3,
            self.m2_q_f2,
            env2,
        )
        q2 += np.random.normal(0, self.noise, q2.shape)

        u2 = self._gen_comp(
            self.m2_u_min,
            self.m2_u_max,
            self.m2_u_th + np.pi,
            self.m2_u_f3,
            self.m2_u_f2,
            env2,
        )
        u2 += np.random.normal(0, self.noise, u2.shape)

        v2 = np.random.normal(0, self.noise, (self.nsubs, self.nbins))

        q2 /= 5
        u2 /= 5
        v2 /= 5

        i2 = np.sqrt(q2**2 + u2**2 + v2**2) * 5

        # --- Incoherent Sum ---
        q_sum = q1 + q2
        u_sum = u1 + u2
        v_sum = v1 + v2
        i_sum = i1 + i2

        # --- Calculate PAs ---
        # Helper to calc PA
        def get_pa(q, u):
            pa = np.degrees(0.5 * np.arctan2(u, q))
            return pa

        # Store Stack Data
        self.res_m1 = {"I": i1, "Q": q1, "U": u1, "V": v1, "PA": get_pa(q1, u1)}
        self.res_m2 = {"I": i2, "Q": q2, "U": u2, "V": v2, "PA": get_pa(q2, u2)}
        self.res_sum = {
            "I": i_sum,
            "Q": q_sum,
            "U": u_sum,
            "V": v_sum,
            "PA": get_pa(q_sum, u_sum),
        }

        # --- Perform Analysis on SUPERPOSITION (Total) ---
        # 改用 self.analysis_fp3 而不是 self.fp3_int，保持使用者調整的 Anal. 1/P3
        self.analysis = perform_spa_analysis(
            i_sum, q_sum, u_sum, v_sum, self.analysis_fp3
        )


# %%
# ==========================================
# 3. Visualization Layout
# ==========================================

state = SimState()
state.run()

fig = plt.figure(figsize=(14, 8))
# Removed global subplots_adjust to use specific GridSpec layouts
# plt.subplots_adjust(...)

# Layout Grid
# Top section: Rows 0-2 (Stacks) - Tight vertical spacing
gs_top = gridspec.GridSpec(
    3,
    4,
    figure=fig,
    left=0.05,
    right=0.95,
    top=0.95,
    bottom=0.55,
    hspace=0.08,
    wspace=0.3,
)

# Bottom section: Row 3 (Analysis) - Separated from top
gs_bot = gridspec.GridSpec(
    1,
    3,
    figure=fig,
    left=0.05,
    right=0.95,
    top=0.48,
    bottom=0.3,
    wspace=0.25,
    width_ratios=[1, 3.5, 3.5],
)

# --- Image Placeholders ---
axes_stacks = []  # List of lists [Mode1_axes, Mode2_axes, Sum_axes]
titles = ["OPM1", "OPM2", "Incoherent Sum"]

for row in range(3):
    row_ax = []
    # I, Q, U, PA
    for col, label in enumerate(["I", "Q", "U", "PA"]):
        ax = fig.add_subplot(gs_top[row, col])
        if col == 0:
            ax.set_ylabel(titles[row], fontsize=10)
        if row == 0:
            ax.set_title(f"{label}")
        if row != 2:
            plt.setp(ax.get_xticklabels(), visible=False)
        row_ax.append(ax)
    axes_stacks.append(row_ax)

# --- Analysis Placeholders ---
ax_3d = fig.add_subplot(gs_bot[0, 0], projection="3d")
ax_amp = fig.add_subplot(gs_bot[0, 1])
ax_phase = fig.add_subplot(gs_bot[0, 2])

# --- Plotting Helpers ---

imgs = [[None] * 4 for _ in range(3)]  # 3x4 grid of image objects
lines = {}  # Store line objects


def init_plots():
    # 1. Stacks
    flat_huslmap = colors.ListedColormap(sns.color_palette("husl", 256))
    cmaps = ["Greys", "coolwarm", "coolwarm", flat_huslmap]
    vmins = [0, -1, -1, -90]
    vmaxs = [1, 1, 1, 90]

    datasets = [state.res_m1, state.res_m2, state.res_sum]

    for r in range(3):
        data = datasets[r]
        for c, key in enumerate(["I", "Q", "U", "PA"]):
            arr = data[key]
            # Dynamic range for I
            curr_vmin = vmins[c]
            curr_vmax = vmaxs[c]
            if key == "I":
                curr_vmax = np.max(arr)

            if key == "PA":
                # Use I for alpha channel
                norm = colors.Normalize(vmin=curr_vmin, vmax=curr_vmax)
                rgba = cmaps[c](norm(arr))

                i_arr = data["I"]
                i_max = np.max(i_arr)
                alpha = i_arr / i_max if i_max > 0 else np.zeros_like(i_arr)
                rgba[..., 3] = np.clip(alpha, 0, 1)

                im = axes_stacks[r][c].imshow(
                    rgba,
                    aspect="auto",
                    origin="lower",
                )
                # Set scalar mappable properties for colorbar
                im.set_cmap(cmaps[c])
                im.set_clim(curr_vmin, curr_vmax)
            else:
                im = axes_stacks[r][c].imshow(
                    arr,
                    aspect="auto",
                    origin="lower",
                    cmap=cmaps[c],
                    vmin=curr_vmin,
                    vmax=curr_vmax,
                )
            imgs[r][c] = im

    # Add shared colorbars for each column (spanning rows 0-2)
    for c in range(4):
        # Group axes by column for rows 0, 1, 2
        col_axes = [axes_stacks[r][c] for r in range(3)]
        # Use the last row (Sum) as the mappable reference
        mappable = imgs[2][c]
        fig.colorbar(
            mappable, ax=col_axes, fraction=0.05, pad=0.02, aspect=40, format="%.1f"  # type: ignore
        )

    # 2. Analysis
    x = np.arange(state.nbins)
    res = state.analysis

    # Amplitudes
    # Normalize profile for comparison
    prof_norm = (
        res["profile"] / np.max(res["profile"])
        if np.max(res["profile"]) > 0
        else res["profile"]
    )

    (lines["prof"],) = ax_amp.plot(
        x, prof_norm, "k-", lw=1, alpha=0.3, label="Pulse Profile"
    )
    (lines["L"],) = ax_amp.plot(x, np.abs(res["L"]), "k:", lw=1.5, label="$|m_I|$")
    (lines["A"],) = ax_amp.plot(x, res["A_abs"], "C1--", lw=1.5, label="$|A|$")
    (lines["B"],) = ax_amp.plot(x, res["B_abs"], "C3--", lw=1.5, label="$|B|$")
    (lines["m1"],) = ax_amp.plot(
        x, np.abs(res["m1"]), "C2-", lw=2, label="$|m_{{OPM1}}|$"
    )
    (lines["m2"],) = ax_amp.plot(
        x, np.abs(res["m2"]), "C0-", lw=2, label="$|m_{{OPM2}}|$"
    )

    # ax_amp.set_title("Modulation Amplitude (Superposition Analysis)")
    xlim_left = state.loc1 - state.sig1 * 2
    xlim_right = state.loc2 + state.sig2 * 2
    xlim_left, xlim_right = max(0, min(xlim_left, xlim_right)), min(
        max(xlim_left, xlim_right), state.nbins
    )
    ax_amp.set_xlim(xlim_left, xlim_right)
    ax_amp.set_ylim(0, 1.2)
    ax_amp.legend(
        ncol=6,
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        handlelength=1,
        handletextpad=0.5,
        borderaxespad=0.5,
        # labelspacing=0.3,
        columnspacing=1,
    )
    ax_amp.grid(True, alpha=0.3)

    # Add vertical guide line for current phase bin on amplitude plot
    phase_x = state.loc1
    lines["vline_amp"] = ax_amp.axvline(
        phase_x, color="gray", lw=2, alpha=0.3, linestyle="--"
    )

    # 3D Scatter
    # Initial phase bin = state.loc1
    phase_bin = state.loc1
    q_data = state.res_sum["Q"][:, phase_bin]
    u_data = state.res_sum["U"][:, phase_bin]
    v_data = state.res_sum["V"][:, phase_bin]
    limit = max(np.max(np.abs(q_data)), np.max(np.abs(u_data)), np.max(np.abs(v_data)))
    if limit == 0:
        limit = 1
    ax_3d.scatter(
        q_data, u_data, v_data, c=np.arange(len(q_data)), cmap="viridis", alpha=0.6
    )
    # Draw Sphere Grid (Wireframe)
    phi_vals = np.linspace(0, 2 * np.pi, 12)
    theta_vals = np.linspace(-np.pi / 2, np.pi / 2, 8)
    theta_highres = np.linspace(-np.pi / 2, np.pi / 2, 50)
    phi_highres = np.linspace(0, 2 * np.pi, 50)

    # Longitudes
    for phi in phi_vals:
        x_ = limit * np.cos(theta_highres) * np.cos(phi)
        y_ = limit * np.cos(theta_highres) * np.sin(phi)
        z_ = limit * np.sin(theta_highres)
        ax_3d.plot(x_, y_, z_, color="gray", alpha=0.2, lw=0.5)

    # Latitudes
    for theta in theta_vals:
        x_ = limit * np.cos(theta) * np.cos(phi_highres)
        y_ = limit * np.cos(theta) * np.sin(phi_highres)
        z_ = limit * np.sin(theta) * np.ones_like(x_)
        ax_3d.plot(x_, y_, z_, color="gray", alpha=0.2, lw=0.5)

    # Draw Axes
    ax_3d.plot([-limit, limit], [0, 0], [0, 0], "k-", lw=1, alpha=0.5)
    ax_3d.plot([0, 0], [-limit, limit], [0, 0], "k-", lw=1, alpha=0.5)
    ax_3d.plot([0, 0], [0, 0], [-limit, limit], "k-", lw=1, alpha=0.5)

    # Add Axis Labels
    ax_3d.text(
        limit * 1.3,
        0,
        0,  # type: ignore
        "$Q$",  # type: ignore
        color="black",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        alpha=0.5,
    )
    ax_3d.text(
        0,
        limit * 1.3,
        0,  # type: ignore
        "$U$",  # type: ignore
        color="black",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        alpha=0.5,
    )
    ax_3d.text(
        0,
        0,
        limit * 1.3,  # type: ignore
        "$V$",  # type: ignore
        color="black",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        alpha=0.5,
    )
    ax_3d.grid(False)
    ax_3d.set_axis_off()
    # Enforce tighter view and equal aspect
    ax_3d.set_xlim(-limit * 0.7, limit * 0.7)
    ax_3d.set_ylim(-limit * 0.7, limit * 0.7)
    ax_3d.set_zlim(-limit * 0.7, limit * 0.7)  # type: ignore
    ax_3d.set_box_aspect((1, 1, 1))  # type: ignore
    # Look vertically down the Z axis
    ax_3d.view_init(elev=90, azim=-90)  # type: ignore

    # Phases
    lines["ph_L"] = ax_phase.scatter(
        x,
        np.degrees(-np.angle(res["L"])) % 360,
        marker="s",
        c="k",
        s=10,
        alpha=0.5,
        label="$\\angle m_I$",
    )
    lines["ph_A"] = ax_phase.scatter(
        x,
        np.degrees(-np.angle(np.exp(1j * np.unwrap(res["phi"], period=np.pi)))) % 360,
        marker="^",
        c="C1",
        s=10,
        alpha=0.5,
        label="$\\angle m_A$",
    )
    lines["ph_m1"] = ax_phase.scatter(
        x,
        np.degrees(-np.angle(res["m1"])) % 360,
        marker="o",
        c="C2",
        s=10,
        alpha=0.5,
        label="$\\angle m_{\\rm OPM1}$",
    )
    lines["ph_m2"] = ax_phase.scatter(
        x,
        np.degrees(-np.angle(res["m2"])) % 360,
        marker="*",
        c="C0",
        s=10,
        alpha=0.5,
        label="$\\angle m_{\\rm OPM2}$",
    )

    # ax_phase.set_title("Subpulse Phase (Superposition Analysis)")
    ax_phase.set_ylim(0, 360)
    xlim_left = state.loc1 - state.sig1 * 1.5
    xlim_right = state.loc2 + state.sig2 * 1.5
    xlim_left, xlim_right = max(0, min(xlim_left, xlim_right)), min(
        max(xlim_left, xlim_right), state.nbins
    )
    ax_phase.set_xlim(xlim_left, xlim_right)
    # ax_phase.set_ylabel("Subpulse phase")
    ax_phase.legend(
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        handlelength=0.8,
        handletextpad=0.4,
        borderaxespad=0.4,
        # labelspacing=0.2,
        columnspacing=1,
        markerscale=2,
    )
    ax_phase.grid(True, alpha=0.3)

    # Add vertical guide line for current phase bin on phase plot
    lines["vline_phase"] = ax_phase.axvline(
        phase_x, color="gray", lw=2, alpha=0.3, linestyle="--"
    )


init_plots()

# ==========================================
# 4. Sliders
# ==========================================

ax_bg = "lightgoldenrodyellow"
sliders = {}


def add_sl(label, vmin, vmax, vinit, valstep, col, row, color=None):
    # Calculate position: Bottom area is 0.0 to 0.35
    # 5 columns
    cw, cs = 0.1, 0.08  # width, spacing
    left = 0.08 + col * (cw + cs)
    bottom = 0.23 - row * 0.03
    ax = plt.axes((left, bottom, cw, 0.02), facecolor=ax_bg)
    s = Slider(
        ax,
        label,
        vmin,
        vmax,
        valstep=valstep,
        valinit=vinit,
        color=color,
    )
    s.label.set_size(7)  # type: ignore
    s.label.set_position((-0.07, 0.5))
    s.valtext.set_position((1.07, 0.5))
    s.valtext.set_size(7)  # type: ignore
    return s


# --- Col 0: Global / Analysis ---
sliders["phase_bin"] = add_sl("Phase Bin", 0, state.nbins - 1, state.loc1, 1, 0, 0)
# 使用 state.analysis_fp3 而不是 state.fp3_int，避免視覺與狀態不同步
sliders["fp3"] = add_sl("Anal. 1/P3", 0, 0.5, state.analysis_fp3, 1 / state.nsubs, 0, 1)
sliders["noise"] = add_sl("Noise", 0, 1.0, state.noise, 0.01, 0, 2)

# --- Col 1: Mode 1 Q ---

sliders["q1_min"] = add_sl(
    "Q1 Min", -1, state.m1_q_max, state.m1_q_min, 0.1, 1, 0, "salmon"
)
sliders["q1_max"] = add_sl(
    "Q1 Max", state.m1_q_min, 1, state.m1_q_max, 0.1, 1, 1, "salmon"
)
sliders["q1_th"] = add_sl(
    "Q1 $\\theta_0$", 0, 2 * np.pi, state.m1_q_th, np.pi / 20, 1, 2, "salmon"
)
sliders["q1_f3"] = add_sl("Q1 1/P3", 0, 0.5, state.m1_q_f3, 0.01, 1, 3, "salmon")
sliders["loc1"] = add_sl("Env1 Loc", 0, state.nbins, state.loc1, 1, 1, 4, "salmon")

# --- Col 2: Mode 1 U ---

sliders["u1_min"] = add_sl(
    "U1 Min", -1, state.m1_u_max, state.m1_u_min, 0.1, 2, 0, "salmon"
)
sliders["u1_max"] = add_sl(
    "U1 Max", state.m1_u_min, 1, state.m1_u_max, 0.1, 2, 1, "salmon"
)
sliders["u1_th"] = add_sl(
    "U1 $\\theta_0$", 0, 2 * np.pi, state.m1_u_th, np.pi / 20, 2, 2, "salmon"
)
sliders["u1_f3"] = add_sl("U1 1/P3", 0, 0.5, state.m1_u_f3, 0.01, 2, 3, "salmon")
sliders["sig1"] = add_sl("Env1 Sig", 1, state.nbins, state.sig1, 1, 2, 4, "salmon")

# --- Col 3: Mode 2 Q ---

sliders["q2_min"] = add_sl(
    "Q2 Min", -1, state.m2_q_max, state.m2_q_min, 0.1, 3, 0, "skyblue"
)
sliders["q2_max"] = add_sl(
    "Q2 Max", state.m2_q_min, 1, state.m2_q_max, 0.1, 3, 1, "skyblue"
)
sliders["q2_th"] = add_sl(
    "Q2 $\\theta_0$", 0, 2 * np.pi, state.m2_q_th, np.pi / 20, 3, 2, "skyblue"
)
sliders["q2_f3"] = add_sl("Q2 1/P3", 0, 0.5, state.m2_q_f3, 0.01, 3, 3, "skyblue")
sliders["loc2"] = add_sl("Env2 Loc", 0, state.nbins, state.loc2, 1, 3, 4, "skyblue")

# --- Col 4: Mode 2 U ---

sliders["u2_min"] = add_sl(
    "U2 Min", -1, state.m2_u_max, state.m2_u_min, 0.1, 4, 0, "skyblue"
)
sliders["u2_max"] = add_sl(
    "U2 Max", state.m2_u_min, 1, state.m2_u_max, 0.1, 4, 1, "skyblue"
)
sliders["u2_th"] = add_sl(
    "U2 $\\theta_0$", 0, 2 * np.pi, state.m2_u_th, np.pi / 20, 4, 2, "skyblue"
)
sliders["u2_f3"] = add_sl("U2 1/P3", 0, 0.5, state.m2_u_f3, 0.01, 4, 3, "skyblue")
sliders["sig2"] = add_sl("Env2 Sig", 1, state.nbins, state.sig2, 1, 4, 4, "skyblue")

# --- Sync Buttons ---
# Place under Mode 1 (Col 1)
ax_sync1 = plt.axes((0.33, 0.05, 0.12, 0.04))
ax_sync1.set_frame_on(False)

btn_sync1 = Button(ax_sync1, "Sync Q & U", color="lightgray", hovercolor="0.95")
btn_sync1.label.set_fontsize(9)

# Rounded corner hack
r_rect1 = FancyBboxPatch(
    (0.05, 0.05),
    0.9,
    0.9,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    transform=ax_sync1.transAxes,
    mutation_aspect=3,
    color="lightgray",
)
ax_sync1.add_patch(r_rect1)
btn_sync1.custom_patch = r_rect1  # type: ignore


# Place under Mode 2 (Col 3)
ax_sync2 = plt.axes((0.73, 0.05, 0.12, 0.04))
ax_sync2.set_frame_on(False)

btn_sync2 = Button(ax_sync2, "Sync Q & U", color="lightgray", hovercolor="0.95")
btn_sync2.label.set_fontsize(9)

# Rounded corner hack
r_rect2 = FancyBboxPatch(
    (0.05, 0.05),
    0.9,
    0.9,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    transform=ax_sync2.transAxes,
    mutation_aspect=3,
    color="lightgray",
)
ax_sync2.add_patch(r_rect2)
btn_sync2.custom_patch = r_rect2  # type: ignore

# Sync State
sync_states = {"m1": True, "m2": True}  # default ON

# Track previous values for sync logic
sync_keys_m1 = [
    ("q1_min", "u1_min"),
    ("q1_max", "u1_max"),
    ("q1_th", "u1_th"),
    ("q1_f3", "u1_f3"),
]
sync_keys_m2 = [
    ("q2_min", "u2_min"),
    ("q2_max", "u2_max"),
    ("q2_th", "u2_th"),
    ("q2_f3", "u2_f3"),
]

last_vals = {}
for k in sliders:
    last_vals[k] = sliders[k].val


def update_btn_color(btn, ax, active, active_color):
    c = active_color if active else "lightgray"
    btn.color = c
    btn.hovercolor = c
    # Update the custom patch color directly
    btn.custom_patch.set_facecolor(c)
    fig.canvas.draw_idle()


def toggle_sync1(event):
    sync_states["m1"] = not sync_states["m1"]
    update_btn_color(btn_sync1, ax_sync1, sync_states["m1"], "salmon")
    # Force sync immediately if turning on
    if sync_states["m1"]:
        for qk, uk in sync_keys_m1:
            val = sliders[qk].val
            sliders[uk].set_val(val)


def toggle_sync2(event):
    sync_states["m2"] = not sync_states["m2"]
    update_btn_color(btn_sync2, ax_sync2, sync_states["m2"], "skyblue")
    # Force sync immediately if turning on
    if sync_states["m2"]:
        for qk, uk in sync_keys_m2:
            val = sliders[qk].val
            sliders[uk].set_val(val)


btn_sync1.on_clicked(toggle_sync1)
btn_sync2.on_clicked(toggle_sync2)

# ---- Initial sync activation (default ON) ----
for qk, uk in sync_keys_m1:
    sliders[uk].set_val(sliders[qk].val)
for qk, uk in sync_keys_m2:
    sliders[uk].set_val(sliders[qk].val)

update_btn_color(btn_sync1, ax_sync1, True, "salmon")
update_btn_color(btn_sync2, ax_sync2, True, "skyblue")

# Refresh last_vals after forced sync
for k in ["u1_min", "u1_max", "u1_th", "u1_f3", "u2_min", "u2_max", "u2_th", "u2_f3"]:
    last_vals[k] = sliders[k].val


def update_phase_bin(val):
    # Only update the 3D Poincaré sphere when phase_bin changes
    phase_bin = int(val)
    ax_3d.clear()

    q_data = state.res_sum["Q"][:, phase_bin]
    u_data = state.res_sum["U"][:, phase_bin]
    v_data = state.res_sum["V"][:, phase_bin]

    limit = max(
        np.max(np.abs(q_data)),
        np.max(np.abs(u_data)),
        np.max(np.abs(v_data)),
    )
    if limit == 0:
        limit = 1

    ax_3d.scatter(
        q_data, u_data, v_data, c=np.arange(len(q_data)), cmap="viridis", alpha=0.6
    )

    # --- Sphere grid (wireframe) and axes ---
    phi_vals = np.linspace(0, 2 * np.pi, 12)
    theta_vals = np.linspace(-np.pi / 2, np.pi / 2, 8)
    theta_highres = np.linspace(-np.pi / 2, np.pi / 2, 50)
    phi_highres = np.linspace(0, 2 * np.pi, 50)

    for phi in phi_vals:
        x = limit * np.cos(theta_highres) * np.cos(phi)
        y = limit * np.cos(theta_highres) * np.sin(phi)
        z = limit * np.sin(theta_highres)
        ax_3d.plot(x, y, z, color="gray", alpha=0.2, lw=0.5)

    for theta in theta_vals:
        x = limit * np.cos(theta) * np.cos(phi_highres)
        y = limit * np.cos(theta) * np.sin(phi_highres)
        z = limit * np.sin(theta) * np.ones_like(x)
        ax_3d.plot(x, y, z, color="gray", alpha=0.2, lw=0.5)

    ax_3d.plot([-limit, limit], [0, 0], [0, 0], "k-", lw=1, alpha=0.5)
    ax_3d.plot([0, 0], [-limit, limit], [0, 0], "k-", lw=1, alpha=0.5)
    ax_3d.plot([0, 0], [0, 0], [-limit, limit], "k-", lw=1, alpha=0.5)

    ax_3d.text(
        limit * 1.3,
        0,
        0,  # type: ignore
        "$Q$",  # type: ignore
        color="black",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        alpha=0.5,
    )
    ax_3d.text(
        0,
        limit * 1.3,
        0,  # type: ignore
        "$U$",  # type: ignore
        color="black",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        alpha=0.5,
    )
    ax_3d.text(
        0,
        0,
        limit * 1.3,  # type: ignore
        "$V$",  # type: ignore
        color="black",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        alpha=0.5,
    )

    ax_3d.grid(False)
    ax_3d.set_axis_off()
    ax_3d.set_xlim(-limit * 0.7, limit * 0.7)
    ax_3d.set_ylim(-limit * 0.7, limit * 0.7)
    ax_3d.set_zlim(-limit * 0.7, limit * 0.7)  # type: ignore
    ax_3d.set_box_aspect((1, 1, 1))  # type: ignore
    ax_3d.view_init(elev=90, azim=-90)  # type: ignore

    # Update vertical guide lines to the new phase bin
    lines["vline_amp"].set_xdata([phase_bin, phase_bin])
    lines["vline_phase"].set_xdata([phase_bin, phase_bin])

    fig.canvas.draw_idle()


def update_fp3(val):
    # ...existing code before computing res...
    target_fp3 = float(val)
    # 同步更新狀態 (讓之後 run() 使用最新分析頻率)
    state.analysis_fp3 = target_fp3
    # ...existing code (I,Q,U,V extraction)...
    I = state.res_sum["I"]
    Q = state.res_sum["Q"]
    U = state.res_sum["U"]
    V = state.res_sum["V"]

    res = perform_spa_analysis(I, Q, U, V, target_fp3)
    # 覆寫 state.analysis 以避免後續 update() 回退到舊頻率
    state.analysis = res
    # ...existing code updating lines...
    prof_max = np.max(res["profile"])
    prof = res["profile"] / prof_max if prof_max > 0 else res["profile"]

    lines["prof"].set_ydata(prof)
    lines["L"].set_ydata(np.abs(res["L"]))
    lines["A"].set_ydata(res["A_abs"])
    lines["B"].set_ydata(res["B_abs"])
    lines["m1"].set_ydata(np.abs(res["m1"]))
    lines["m2"].set_ydata(np.abs(res["m2"]))

    # Scale amplitude plot (keep x-limits unchanged)
    mx = max(np.max(res["A_abs"]), np.max(np.abs(res["m1"])))
    ax_amp.set_ylim(0, mx * 1.2 if mx > 0 else 1.0)

    # Update phase scatter points
    x_vals = np.arange(state.nbins)
    lines["ph_L"].set_offsets(np.c_[x_vals, np.degrees(-np.angle(res["L"])) % 360])
    lines["ph_A"].set_offsets(
        np.c_[
            x_vals,
            np.degrees(-np.angle(np.exp(1j * np.unwrap(res["phi"], period=np.pi))))
            % 360,
        ]
    )
    lines["ph_m1"].set_offsets(np.c_[x_vals, np.degrees(-np.angle(res["m1"])) % 360])
    lines["ph_m2"].set_offsets(np.c_[x_vals, np.degrees(-np.angle(res["m2"])) % 360])

    # Keep vertical guide lines in sync with current phase bin slider
    phase_bin = int(sliders["phase_bin"].val)
    lines["vline_amp"].set_xdata([phase_bin, phase_bin])
    lines["vline_phase"].set_xdata([phase_bin, phase_bin])

    fig.canvas.draw_idle()


# Helper: dynamically constrain min/max slider pairs
def _adjust_min_max(min_key, max_key):
    smin = sliders[min_key]
    smax = sliders[max_key]
    # Set dynamic bounds
    smin.valmax = smax.val
    smax.valmin = smin.val
    # Clamp values if out of new bounds
    if smin.val > smin.valmax:
        smin.set_val(smin.valmax)
    if smax.val < smax.valmin:
        smax.set_val(smax.valmin)
    # Refresh slider axes limits
    smin.ax.set_xlim(smin.valmin, smin.valmax)
    smax.ax.set_xlim(smax.valmin, smax.valmax)


def update(val):
    # 確保分析頻率不回退：每次其它 slider 觸發時抓目前 fp3 slider 值（不強制重算 slider）
    state.analysis_fp3 = sliders["fp3"].val

    # Handle Synchronization

    def process_sync(pairs, active):
        for qk, uk in pairs:
            cq, cu = sliders[qk].val, sliders[uk].val
            lq, lu = last_vals[qk], last_vals[uk]

            if active:
                if cq != lq:
                    sliders[uk].eventson = False
                    sliders[uk].set_val(cq)
                    sliders[uk].eventson = True
                    cu = cq
                elif cu != lu:
                    sliders[qk].eventson = False
                    sliders[qk].set_val(cu)
                    sliders[qk].eventson = True
                    cq = cu

            last_vals[qk] = cq
            last_vals[uk] = cu

    process_sync(sync_keys_m1, sync_states["m1"])
    process_sync(sync_keys_m2, sync_states["m2"])

    # Dynamic constraint of min/max pairs before reading into state
    _adjust_min_max("q1_min", "q1_max")
    _adjust_min_max("u1_min", "u1_max")
    _adjust_min_max("q2_min", "q2_max")
    _adjust_min_max("u2_min", "u2_max")

    # Update State (do NOT touch fp3 here)
    state.noise = sliders["noise"].val
    state.loc1 = sliders["loc1"].val
    state.sig1 = sliders["sig1"].val
    state.loc2 = sliders["loc2"].val
    state.sig2 = sliders["sig2"].val

    state.m1_q_min = sliders["q1_min"].val
    state.m1_q_max = sliders["q1_max"].val
    state.m1_q_th = sliders["q1_th"].val
    state.m1_q_f3 = sliders["q1_f3"].val
    state.m1_u_min = sliders["u1_min"].val
    state.m1_u_max = sliders["u1_max"].val
    state.m1_u_th = sliders["u1_th"].val
    state.m1_u_f3 = sliders["u1_f3"].val

    state.m2_q_min = sliders["q2_min"].val
    state.m2_q_max = sliders["q2_max"].val
    state.m2_q_th = sliders["q2_th"].val
    state.m2_q_f3 = sliders["q2_f3"].val
    state.m2_u_min = sliders["u2_min"].val
    state.m2_u_max = sliders["u2_max"].val
    state.m2_u_th = sliders["u2_th"].val
    state.m2_u_f3 = sliders["u2_f3"].val

    state.run()

    # Update Images
    datasets = [state.res_m1, state.res_m2, state.res_sum]
    for r in range(3):
        d = datasets[r]
        imgs[r][0].set_data(d["I"])  # type: ignore
        imgs[r][0].set_clim(0, np.max(d["I"]))  # type: ignore
        imgs[r][1].set_data(d["Q"])  # type: ignore
        imgs[r][2].set_data(d["U"])  # type: ignore

        # Update PA with alpha from I
        pa_data = d["PA"]
        i_data = d["I"]

        # Reconstruct RGBA
        cmap = imgs[r][3].get_cmap()  # type: ignore
        norm = colors.Normalize(vmin=-90, vmax=90)
        rgba = cmap(norm(pa_data))

        i_max = np.max(i_data)
        alpha = i_data / i_max if i_max > 0 else np.zeros_like(i_data)
        rgba[..., 3] = np.clip(alpha, 0, 1)

        imgs[r][3].set_data(rgba)  # type: ignore

    # Update Analysis
    res = state.analysis

    # Update 3D Scatter
    phase_bin = int(sliders["phase_bin"].val)
    ax_3d.clear()
    q_data = state.res_sum["Q"][:, phase_bin]
    u_data = state.res_sum["U"][:, phase_bin]
    v_data = state.res_sum["V"][:, phase_bin]
    limit = (
        max(np.max(np.abs(q_data)), np.max(np.abs(u_data)), np.max(np.abs(v_data))) * 1
    )
    if limit == 0:
        limit = 1
    ax_3d.scatter(
        q_data, u_data, v_data, c=np.arange(len(q_data)), cmap="viridis", alpha=0.6
    )

    # Draw Sphere Grid (Wireframe)
    phi_vals = np.linspace(0, 2 * np.pi, 12)
    theta_vals = np.linspace(-np.pi / 2, np.pi / 2, 8)
    theta_highres = np.linspace(-np.pi / 2, np.pi / 2, 50)
    phi_highres = np.linspace(0, 2 * np.pi, 50)

    # Longitudes
    for phi in phi_vals:
        x = limit * np.cos(theta_highres) * np.cos(phi)
        y = limit * np.cos(theta_highres) * np.sin(phi)
        z = limit * np.sin(theta_highres)
        ax_3d.plot(x, y, z, color="gray", alpha=0.2, lw=0.5)

    # Latitudes
    for theta in theta_vals:
        x = limit * np.cos(theta) * np.cos(phi_highres)
        y = limit * np.cos(theta) * np.sin(phi_highres)
        z = limit * np.sin(theta) * np.ones_like(x)
        ax_3d.plot(x, y, z, color="gray", alpha=0.2, lw=0.5)

    # Draw Axes
    ax_3d.plot([-limit, limit], [0, 0], [0, 0], "k-", lw=1, alpha=0.5)
    ax_3d.plot([0, 0], [-limit, limit], [0, 0], "k-", lw=1, alpha=0.5)
    ax_3d.plot([0, 0], [0, 0], [-limit, limit], "k-", lw=1, alpha=0.5)

    # Add Axis Labels
    ax_3d.text(
        limit * 1.3,
        0,
        0,  # type: ignore
        "$Q$",  # type: ignore
        color="black",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        alpha=0.5,
    )
    ax_3d.text(
        0,
        limit * 1.3,
        0,  # type: ignore
        "$U$",  # type: ignore
        color="black",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        alpha=0.5,
    )
    ax_3d.text(
        0,
        0,
        limit * 1.3,  # type: ignore
        "$V$",  # type: ignore
        color="black",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        alpha=0.5,
    )
    ax_3d.grid(False)
    ax_3d.set_axis_off()
    # Enforce tighter view and equal aspect
    ax_3d.set_xlim(-limit * 0.7, limit * 0.7)
    ax_3d.set_ylim(-limit * 0.7, limit * 0.7)
    ax_3d.set_zlim(-limit * 0.7, limit * 0.7)  # type: ignore
    ax_3d.set_box_aspect((1, 1, 1))  # type: ignore
    # Look vertically down the Z axis
    ax_3d.view_init(elev=90, azim=-90)  # type: ignore

    # Norm Profile
    prof_max = np.max(res["profile"])
    prof = res["profile"] / prof_max if prof_max > 0 else res["profile"]

    lines["prof"].set_ydata(prof)
    lines["L"].set_ydata(np.abs(res["L"]))
    lines["A"].set_ydata(res["A_abs"])
    lines["B"].set_ydata(res["B_abs"])
    lines["m1"].set_ydata(np.abs(res["m1"]))
    lines["m2"].set_ydata(np.abs(res["m2"]))

    # Scale Amp plot
    mx = max(np.max(res["A_abs"]), np.max(np.abs(res["m1"])))
    ax_amp.set_ylim(0, mx * 1.2 if mx > 0 else 1.0)

    # Phases
    x_vals = np.arange(state.nbins)
    lines["ph_L"].set_offsets(np.c_[x_vals, np.degrees(-np.angle(res["L"])) % 360])
    lines["ph_A"].set_offsets(
        np.c_[
            x_vals,
            np.degrees(-np.angle(np.exp(1j * np.unwrap(res["phi"], period=np.pi))))
            % 360,
        ]
    )
    lines["ph_m1"].set_offsets(np.c_[x_vals, np.degrees(-np.angle(res["m1"])) % 360])
    lines["ph_m2"].set_offsets(np.c_[x_vals, np.degrees(-np.angle(res["m2"])) % 360])

    # Keep vertical guide lines in sync after any full update
    lines["vline_amp"].set_xdata([phase_bin, phase_bin])
    lines["vline_phase"].set_xdata([phase_bin, phase_bin])

    fig.canvas.draw_idle()


# Connect sliders: phase_bin -> lightweight handler, fp3 -> lightweight SPA update, others -> full update
phase_slider = sliders["phase_bin"]
phase_slider.on_changed(update_phase_bin)

sliders["fp3"].on_changed(update_fp3)

for name in (
    "noise",
    "q1_min",
    "q1_max",
    "q1_th",
    "q1_f3",
    "loc1",
    "u1_min",
    "u1_max",
    "u1_th",
    "u1_f3",
    "sig1",
    "q2_min",
    "q2_max",
    "q2_th",
    "q2_f3",
    "loc2",
    "u2_min",
    "u2_max",
    "u2_th",
    "u2_f3",
    "sig2",
):
    sliders[name].on_changed(update)

plt.show()
