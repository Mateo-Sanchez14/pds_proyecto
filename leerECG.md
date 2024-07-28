# Cargar el archivo .mat

file_path = "ECG_normal.mat"
mat = scipy.io.loadmat(file_path)

# Extraer la señal del ECG

ecg_signal = mat["val"].flatten()
