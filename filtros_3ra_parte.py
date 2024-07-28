import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import neurokit2 as nk


# Cargar el archivo .mat
file_path = "ECG_normal.mat"
mat = scipy.io.loadmat(file_path)

# Extraer la señal del ECG
ecg_signal = mat["val"].flatten()

# Transformar la señal a un formato adecuado para procesar
ecg_signal = nk.ecg_process(ecg_signal, sampling_rate=250)[0]["ECG_Raw"]
# Extract R-peaks locations
_, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=250)

# Visualize R-peaks in ECG signal
plot = nk.events_plot(rpeaks["ECG_R_Peaks"], ecg_signal)

# Delineate the ECG signal
_, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=250, method="peak")

# Zooming into the first 3 R-peaks, with focus on T_peaks, P-peaks, Q-peaks and S-peaks
plot = nk.events_plot(
    [
        waves_peak["ECG_T_Peaks"][:3],
        waves_peak["ECG_P_Peaks"][:3],
        waves_peak["ECG_Q_Peaks"][:3],
        waves_peak["ECG_S_Peaks"][:3],
    ],
    ecg_signal[:4000],
)

# Delineate the ECG signal and visualizing all peaks of ECG complexes
_, waves_peak = nk.ecg_delineate(
    ecg_signal, rpeaks, sampling_rate=250, method="peak", show=True, show_type="peaks"
)


plt.show()
