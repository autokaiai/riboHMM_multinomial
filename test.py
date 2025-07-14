import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

# --- 1. Hilfsfunktionen (unverändert) ---
def rolling_window(a, window_size):
    """ Erstellt eine gleitende Fensteransicht eines Arrays. """
    shape = (a.shape[0] - window_size + 1, window_size)
    strides = (a.strides[0], a.strides[0])
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# --- 2. Mock-Daten erstellen für read_counts_1 (unverändert) ---
seq_length_1, cds_start_pos_1, cds_end_pos_1 = 800, 200, 650
window_size, p_site_offset = 31, 13
read_counts_1 = np.zeros(seq_length_1, dtype=int)

read_counts_1[0:cds_start_pos_1 - p_site_offset] = np.random.poisson(0.2, size=cds_start_pos_1 - p_site_offset)
read_counts_1[cds_end_pos_1 - p_site_offset:] = np.random.poisson(0.2, size=seq_length_1 - (cds_end_pos_1 - p_site_offset))
footprint_start_1, footprint_end_1 = cds_start_pos_1 - p_site_offset, cds_end_pos_1 - p_site_offset
for i in range(footprint_start_1, footprint_end_1, 3):
    if i + 2 < footprint_end_1:
        read_counts_1[i] += np.random.poisson(10); read_counts_1[i+1] += np.random.poisson(5); read_counts_1[i+2] += np.random.poisson(2)
read_counts_1[cds_start_pos_1 - 14] += np.random.poisson(25); read_counts_1[cds_start_pos_1 - 13] += np.random.poisson(70); read_counts_1[cds_start_pos_1 - 12] += np.random.poisson(25)
read_counts_1[cds_end_pos_1 - 14] += np.random.poisson(20)
read_counts_1[cds_end_pos_1 - 13] += np.random.poisson(60)
read_counts_1[cds_end_pos_1 - 12] += np.random.poisson(20)

read_counts_1 = read_counts_1.astype(int)
padding = window_size // 2
padded_counts_1 = np.pad(read_counts_1, (padding, padding), 'constant', constant_values=0)
X_full_1 = rolling_window(padded_counts_1, window_size)


# --- 2b. Zweite Mock-Daten erstellen für read_counts_2 (NEU) ---
seq_length_2, cds_start_pos_2, cds_end_pos_2 = 600, 150, 500 # Andere Länge und Positionen
read_counts_2 = np.zeros(seq_length_2, dtype=int)

read_counts_2[0:cds_start_pos_2 - p_site_offset] = np.random.poisson(0.3, size=cds_start_pos_2 - p_site_offset) # Leicht andere Raten
read_counts_2[cds_end_pos_2 - p_site_offset:] = np.random.poisson(0.3, size=seq_length_2 - (cds_end_pos_2 - p_site_offset))
footprint_start_2, footprint_end_2 = cds_start_pos_2 - p_site_offset, cds_end_pos_2 - p_site_offset
for i in range(footprint_start_2, footprint_end_2, 3):
    if i + 2 < footprint_end_2:
        read_counts_2[i] += np.random.poisson(12); read_counts_2[i+1] += np.random.poisson(6); read_counts_2[i+2] += np.random.poisson(3)
read_counts_2[cds_start_pos_2 - 14] += np.random.poisson(30); read_counts_2[cds_start_pos_2 - 13] += np.random.poisson(80); read_counts_2[cds_start_pos_2 - 12] += np.random.poisson(30)
read_counts_2[cds_end_pos_2 - 14] += np.random.poisson(25)
read_counts_2[cds_end_pos_2 - 13] += np.random.poisson(70)
read_counts_2[cds_end_pos_2 - 12] += np.random.poisson(25)

read_counts_2 = read_counts_2.astype(int)
padded_counts_2 = np.pad(read_counts_2, (padding, padding), 'constant', constant_values=0)
X_full_2 = rolling_window(padded_counts_2, window_size)

# Kombiniere die Feature-Matrizen und ihre Längen
X_combined = np.concatenate((X_full_1, X_full_2), axis=0)
lengths_combined = [len(X_full_1), len(X_full_2)]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# HMM-Modell: Initialisierung und Training
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
state_map = {
    "5'UTR": 0, "Pre-INIT": 1, "INIT": 2, "T1": 3, "T2": 4, "T3": 5,
    "Pre-TERM": 6, "TERM": 7, "3'UTR": 8
}
n_components = len(state_map)
n_features = window_size

fixed_emission_states = [state_map["5'UTR"], state_map["3'UTR"]]
learnable_params_str = 'se'
init_params_str = ''

final_model = hmm.MultinomialHMM(n_components=n_components,
                                 n_iter=200,
                                 tol=1e-4,
                                 verbose=True,
                                 params=learnable_params_str,
                                 init_params=init_params_str,
                                 random_state=42)

# --- Manuelle Initialisierung der Parameter (unverändert bis zu den Emissionen) ---
final_model.startprob_ = np.array([1.0] + [0.0] * (n_components - 1))
initial_transmat = np.zeros((n_components, n_components))
initial_transmat[state_map["5'UTR"], state_map["5'UTR"]] = 0.9999
initial_transmat[state_map["5'UTR"], state_map["Pre-INIT"]] = 0.0001
initial_transmat[state_map["Pre-INIT"], state_map["Pre-INIT"]] = 0.99
initial_transmat[state_map["Pre-INIT"], state_map["INIT"]] = 0.01
initial_transmat[state_map["INIT"], state_map["INIT"]] = 0.02
initial_transmat[state_map["INIT"], state_map["T1"]] = 0.98
p_elong_cycle = 0.99
p_elong_to_pre_term = 1 - p_elong_cycle
initial_transmat[state_map["T1"], state_map["T2"]] = 1
initial_transmat[state_map["T2"], state_map["T3"]] = 1
initial_transmat[state_map["T3"], state_map["T1"]] = p_elong_cycle
initial_transmat[state_map["T3"], state_map["Pre-TERM"]] = p_elong_to_pre_term
initial_transmat[state_map["Pre-TERM"], state_map["Pre-TERM"]] = 0.99
initial_transmat[state_map["Pre-TERM"], state_map["TERM"]] = 0.01
initial_transmat[state_map["TERM"], state_map["3'UTR"]] = 1.0
initial_transmat[state_map["3'UTR"], state_map["3'UTR"]] = 1.0
final_model.transmat_ = initial_transmat

# --- Erstelle die Signaturen als erwartete Zählungen (wie bei Poisson) ---
initial_unnormalized_emissions = np.full((n_components, n_features), 0.1)
center_idx = window_size // 2
initial_unnormalized_emissions[state_map["5'UTR"], :] = 0.2
initial_unnormalized_emissions[state_map["3'UTR"], :] = 0.2
initial_unnormalized_emissions[state_map["Pre-INIT"], :] = 0.3
initial_unnormalized_emissions[state_map["Pre-INIT"], center_idx-15:center_idx-10] = 0.01
initial_unnormalized_emissions[state_map["Pre-INIT"], center_idx-10:center_idx+16] = 5.0
initial_unnormalized_emissions[state_map["INIT"], :] = 0.1
initial_unnormalized_emissions[state_map["INIT"], center_idx-14:center_idx-11] = [30, 80, 30]
for i in range(center_idx-10, window_size):
    phase = (i - (center_idx-13)) % 3
    if phase == 0: initial_unnormalized_emissions[state_map["INIT"], i] = 10.0
    elif phase == 1: initial_unnormalized_emissions[state_map["INIT"], i] = 5.0
    else: initial_unnormalized_emissions[state_map["INIT"], i] = 2.0
initial_unnormalized_emissions[state_map["T1"], :] = np.random.uniform(8.0, 12.0, window_size)
initial_unnormalized_emissions[state_map["T2"], :] = np.random.uniform(4.0, 6.0, window_size)
initial_unnormalized_emissions[state_map["T3"], :] = np.random.uniform(1.0, 3.0, window_size)
initial_unnormalized_emissions[state_map["Pre-TERM"], :] = 0.1
for i in range(0, center_idx):
    phase = (i - (center_idx-13)) % 3
    if phase == 0: initial_unnormalized_emissions[state_map["Pre-TERM"], i] = 10.0
    elif phase == 1: initial_unnormalized_emissions[state_map["Pre-TERM"], i] = 5.0
    else: initial_unnormalized_emissions[state_map["Pre-TERM"], i] = 2.0
initial_unnormalized_emissions[state_map["Pre-TERM"], center_idx-1:center_idx+2] = [20, 60, 20]

initial_unnormalized_emissions[state_map["TERM"], :] = 0.1
initial_unnormalized_emissions[state_map["TERM"], center_idx-14:center_idx-11] = [20, 60, 20]
initial_unnormalized_emissions[state_map["TERM"], center_idx+1:] = 0.1


# <<<<<<< NEU: Funktion für hybride Normalisierung >>>>>>>>>
def hybrid_normalize(unnormalized_emissions):
    """
    Normalisiert Emissionen, indem die Form mit einem uniformen Hintergrund gemischt wird.
    Das Mischverhältnis basiert auf der Magnitude (durchschnittliche Zählrate) jedes Zustands.
    """
    mean_magnitudes = unnormalized_emissions.mean(axis=1)
    
    learnable_magnitudes = np.delete(mean_magnitudes, fixed_emission_states)
    max_mean_magnitude = learnable_magnitudes.max()
    if max_mean_magnitude == 0:
        max_mean_magnitude = 1.0

    row_sums = unnormalized_emissions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    shape_component = unnormalized_emissions / row_sums
    
    background_component = np.full_like(unnormalized_emissions, 1.0 / n_features)
    
    mixing_weights = (mean_magnitudes / max_mean_magnitude)[:, np.newaxis]
    mixing_weights = np.clip(mixing_weights, 0, 1) 
    
    hybrid_emissions = mixing_weights * shape_component + (1 - mixing_weights) * background_component
    
    final_sums = hybrid_emissions.sum(axis=1, keepdims=True)
    return hybrid_emissions / final_sums

# Wende die neue Normalisierungs-Funktion an
initial_emissionprob = hybrid_normalize(initial_unnormalized_emissions)

# Setze die initialen Emissionswahrscheinlichkeiten für das Modell
final_model.emissionprob_ = initial_emissionprob.copy()
initial_emissionprob_snapshot = initial_emissionprob.copy()


# --- Training des Modells ---
print("Starte Training des MultinomialHMM mit zwei Sequenzen...")
final_model.fit(X_combined, lengths_combined) # Hier werden die kombinierten Daten und Längen übergeben

# --- Zurücksetzen der fixierten Zustände ---
print(f"Setze Emissionen für fixierte Zustände {fixed_emission_states} zurück...")
for state_idx in fixed_emission_states:
    final_model.emissionprob_[state_idx] = initial_emissionprob_snapshot[state_idx]

print("Training abgeschlossen.")


# --- Inferenz und Visualisierung für read_counts_1 (wie zuvor) ---
print("\nInferenz für die erste Sequenz:")
logprob_1, state_sequence_1 = final_model.decode(X_full_1, algorithm="viterbi")
print(f"Log-Wahrscheinlichkeit des dekodierten Pfades für Sequenz 1: {logprob_1}")

state_labels = list(state_map.keys())
state_colors = plt.cm.turbo(np.linspace(0, 1, n_components))
fig1, (ax1_1, ax1_2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
fig1.suptitle("Ein-Stufen HMM (Multinomial - Hybride Norm.) - Sequenz 1", fontsize=16)
ax1_1.set_title("Simulierte Read Counts und wahre Regionen (Sequenz 1)")
ax1_1.plot(read_counts_1, label="Read Counts", color='gray', alpha=0.7)
ax1_1.axvspan(0, cds_start_pos_1-1, color='lightblue', alpha=0.2, label='Simulierte 5\'UTR')
ax1_1.axvspan(cds_start_pos_1, cds_end_pos_1-1, color='lightcoral', alpha=0.15, label='Simulierte CDS')
ax1_1.axvspan(cds_end_pos_1, seq_length_1-1, color='lightgreen', alpha=0.2, label='Simulierte 3\'UTR')
ax1_1.legend(loc='upper right')
ax1_1.set_ylabel("Anzahl Reads")
ax1_1.set_xlim(0, len(read_counts_1))

ax1_2.set_title("Vorhergesagter biologischer Pfad (Sequenz 1)")
for i, label in enumerate(state_labels):
    ax1_2.fill_between(np.arange(len(state_sequence_1)), 0, 1, where=(state_sequence_1 == i), color=state_colors[i], label=f'{label}', step='post')
ax1_2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1_2.set_ylabel("Zustand")
ax1_2.set_xlabel("Nukleotid-Position")
ax1_2.set_yticks([])
ax1_2.set_xlim(0, len(state_sequence_1))
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()

# --- Inferenz und Visualisierung für read_counts_2 (NEU) ---
print("\nInferenz für die zweite Sequenz:")
logprob_2, state_sequence_2 = final_model.decode(X_full_2, algorithm="viterbi")
print(f"Log-Wahrscheinlichkeit des dekodierten Pfades für Sequenz 2: {logprob_2}")

fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
fig2.suptitle("Ein-Stufen HMM (Multinomial - Hybride Norm.) - Sequenz 2", fontsize=16)
ax2_1.set_title("Simulierte Read Counts und wahre Regionen (Sequenz 2)")
ax2_1.plot(read_counts_2, label="Read Counts", color='gray', alpha=0.7)
ax2_1.axvspan(0, cds_start_pos_2-1, color='lightblue', alpha=0.2, label='Simulierte 5\'UTR')
ax2_1.axvspan(cds_start_pos_2, cds_end_pos_2-1, color='lightcoral', alpha=0.15, label='Simulierte CDS')
ax2_1.axvspan(cds_end_pos_2, seq_length_2-1, color='lightgreen', alpha=0.2, label='Simulierte 3\'UTR')
ax2_1.legend(loc='upper right')
ax2_1.set_ylabel("Anzahl Reads")
ax2_1.set_xlim(0, len(read_counts_2))

ax2_2.set_title("Vorhergesagter biologischer Pfad (Sequenz 2)")
for i, label in enumerate(state_labels):
    ax2_2.fill_between(np.arange(len(state_sequence_2)), 0, 1, where=(state_sequence_2 == i), color=state_colors[i], label=f'{label}', step='post')
ax2_2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2_2.set_ylabel("Zustand")
ax2_2.set_xlabel("Nukleotid-Position")
ax2_2.set_yticks([])
ax2_2.set_xlim(0, len(state_sequence_2))
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()

# --- Emissions-Signaturen Plot (unverändert) ---
fig_emissions, axs_emissions = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
fig_emissions.suptitle("Emissions-Signaturen (Hybride Wahrscheinlichkeiten): Initial und gelernt", fontsize=16)
groups = {
    "Initiation": ["5'UTR", "Pre-INIT", "INIT"],
    "Elongation (gelernt vs. initial)": ["T1", "T2", "T3"],
    "Termination & UTRs": ["Pre-TERM", "TERM", "3'UTR"]
}
for i, (group_name, members) in enumerate(groups.items()):
    axs_emissions[i].set_title(f"Gruppe: {group_name}")
    for member_name in members:
        state_idx = state_map[member_name]
        is_fixed = state_idx in fixed_emission_states
        label_suffix = " (Gelernt/Fix)" if is_fixed else " (Gelernt)"
        axs_emissions[i].plot(final_model.emissionprob_[state_idx], 'o-', color=state_colors[state_idx], label=f'{member_name}{label_suffix}', markersize=4)
        if not is_fixed:
            axs_emissions[i].plot(initial_emissionprob_snapshot[state_idx], 'x--', color=state_colors[state_idx], alpha=0.6, label=f'{member_name} (Initial)', markersize=6, linewidth=1)
    
    axs_emissions[i].legend()
    axs_emissions[i].grid(True, linestyle='--', alpha=0.6)
    axs_emissions[i].set_ylabel("Emissions-Wahrscheinlichkeit")

axs_emissions[2].set_xlabel(f"Position im {window_size}-nt-Fenster")
plt.tight_layout()
plt.show()