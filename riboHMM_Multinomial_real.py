import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# --- 1. Konfiguration ---
# Eingabedatei für das Training
INPUT_NPZ_FILE = '/home/kai/Documents/00_RNA/00_riboHMM_simple/01_Proccesed_Data/top_1000_transcripts.npz'

# Parameter für die Fenstererstellung
window_size = 31
p_site_offset = 13 # Dieser Wert wird hier nicht direkt verwendet, aber ist Teil des Konzepts

# --- 2. Hilfsfunktionen ---
def rolling_window(a, window_size):
    """ Erstellt eine gleitende Fensteransicht eines Arrays. """
    shape = (a.shape[0] - window_size + 1, window_size)
    strides = (a.strides[0], a.strides[0])
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def load_processed_data(file_path):
    """Lädt die verarbeiteten Daten aus einer .npz-Datei."""
    print(f"Lade verarbeitete Daten von {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ Fehler: Datendatei nicht gefunden: {file_path}")
        return None
    try:
        data_loader = np.load(file_path, allow_pickle=True)
        transcript_data = {key: data_loader[key] for key in data_loader.files}
        print(f"✅ {len(transcript_data)} Transkripte erfolgreich geladen.")
        return transcript_data
    except Exception as e:
        print(f"❌ Fehler beim Laden der .npz-Datei: {e}")
        return None

# --- 3. Daten laden und für HMM vorbereiten ---
transcript_data = load_processed_data(INPUT_NPZ_FILE)
if not transcript_data:
    exit()

print("Bereite Trainingsdaten für das HMM vor (Padding und Rolling Window)...")
all_X_windows = []
lengths_of_windows = []
padding = window_size // 2

for tid, read_counts in tqdm(transcript_data.items(), desc="Verarbeite Transkripte"):
    # Ignoriere sehr kurze Transkripte, die kein volles Fenster ergeben
    if len(read_counts) < window_size:
        continue
    
    padded_counts = np.pad(read_counts, (padding, padding), 'constant', constant_values=0)
    X_single_transcript = rolling_window(padded_counts, window_size)
    
    all_X_windows.append(X_single_transcript)
    lengths_of_windows.append(len(X_single_transcript))

# Füge alle Fenster-Matrizen zu einem großen Trainings-Set zusammen
X_train = np.concatenate(all_X_windows)
print(f"✅ Trainings-Set erstellt mit {X_train.shape[0]} Fenstern aus {len(lengths_of_windows)} Transkripten.")


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# HMM-Modell: Initialisierung und Training (Logik von dir übernommen)
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

# --- Manuelle Initialisierung der Parameter ---
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

# --- Erstelle die Signaturen als erwartete Zählungen ---
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

# --- Funktion für hybride Normalisierung ---
def hybrid_normalize(unnormalized_emissions):
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

initial_emissionprob = hybrid_normalize(initial_unnormalized_emissions)
final_model.emissionprob_ = initial_emissionprob.copy()
initial_emissionprob_snapshot = initial_emissionprob.copy()

# --- 4. Training des Modells ---
print("Starte Training des MultinomialHMM mit echten Daten...")
# Das Training erfolgt auf allen Transkripten, die Längen der Sequenzen werden übergeben.
final_model.fit(X_train, lengths=lengths_of_windows)

# --- Zurücksetzen der fixierten Zustände ---
print(f"Setze Emissionen für fixierte Zustände {fixed_emission_states} zurück...")
for state_idx in fixed_emission_states:
    final_model.emissionprob_[state_idx] = initial_emissionprob_snapshot[state_idx]
print("Training abgeschlossen.")

# --- 5. Inferenz und Visualisierung für ein Beispiel-Transkript ---
print("\nFühre Inferenz und Visualisierung für ein Beispiel-Transkript durch...")
example_tid = list(transcript_data.keys())[0] # Nimm das erste Transkript als Beispiel
example_read_counts = transcript_data[example_tid]

# Bereite die Daten für das Beispiel-Transkript vor
padded_example_counts = np.pad(example_read_counts, (padding, padding), 'constant', constant_values=0)
X_example = rolling_window(padded_example_counts, window_size)

# Dekodiere die Zustandssequenz für das Beispiel
logprob, state_sequence = final_model.decode(X_example, algorithm="viterbi")
print(f"Log-Wahrscheinlichkeit des dekodierten Pfades für {example_tid}: {logprob}")

# --- Visualisierung ---
state_labels = list(state_map.keys())
state_colors = plt.cm.turbo(np.linspace(0, 1, n_components))

# Plot 1: Reads und dekodierter Pfad
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
fig.suptitle(f"HMM-Analyse für Transkript: {example_tid}", fontsize=16)
ax1.set_title("Read Counts")
ax1.plot(example_read_counts, label="Read Counts", color='gray', alpha=0.7)
ax1.legend(loc='upper right')
ax1.set_ylabel("Anzahl Reads")
ax1.set_xlim(0, len(example_read_counts))

ax2.set_title("Vorhergesagter biologischer Pfad")
for i, label in enumerate(state_labels):
    ax2.fill_between(np.arange(len(state_sequence)), 0, 1, where=(state_sequence == i), color=state_colors[i], label=f'{label}', step='post')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2.set_ylabel("Zustand")
ax2.set_xlabel("Nukleotid-Position")
ax2.set_yticks([])
ax2.set_xlim(0, len(state_sequence))
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig("hmm_path_visualization.png")
print("✅ Pfad-Visualisierung gespeichert als 'hmm_path_visualization.png'")

# Plot 2: Emissions-Signaturen
fig_emissions, axs_emissions = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
fig_emissions.suptitle("Emissions-Signaturen: Initial vs. Gelernt", fontsize=16)
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
plt.savefig("hmm_emissions_visualization.png")
print("✅ Emissions-Visualisierung gespeichert als 'hmm_emissions_visualization.png'")