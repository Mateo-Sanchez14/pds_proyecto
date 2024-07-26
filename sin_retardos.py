import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt

# Cargar la señal ECG desde el archivo .mat
mat_contents = sio.loadmat("ECG_senal_3a.mat")
# print(mat_contents.keys())  # Add this line to see all available keys

s = mat_contents["s"].flatten()

# Parámetros de la señal y del procesamiento
Fs = 800  # Frecuencia de muestreo en Hz
Fc_PA = 0.5  # Frecuencia de corte del filtro pasa alto en Hz
Fn_PA = Fc_PA / (Fs / 2)  # Frecuencia normalizada filtro pasa alto
Fc_PB = 80  # Frecuencia de corte del filtro pasa bajo en Hz
Fn_PB = Fc_PB / (Fs / 2)  # Frecuencia normalizada filtro pasa bajo
n = np.arange(len(s))  # Barrido temporal igual al numero de datos adquiridos
n = n * (1 / Fs)  # Afectamos por el valor del muestreo T=1/Fs

# Filtrar la señal con el filtro pasa alto
PA = signal.firwin(101, Fn_PA, pass_zero=False)
senal_filtrada_alto = signal.filtfilt(PA, 1, s)

# Filtrar la señal con el filtro notch
freq_notch = 50.0  # Frecuencia a eliminar (50 Hz)
quality_factor = 30.0  # Factor de calidad del filtro notch
b_notch, a_notch = signal.iirnotch(freq_notch, quality_factor, Fs)
senal_filtrada_notch = signal.filtfilt(b_notch, a_notch, senal_filtrada_alto)

# Filtrar la señal con el filtro pasa bajo
PB = signal.firwin(101, Fn_PB)
senal_filtrada_bajo = signal.filtfilt(PB, 1, senal_filtrada_notch)

# Veamos la respuesta en magnitud y frecuencia de cada filtro
N = 1024  # Número de puntos para la respuesta

plt.figure()
w, h = signal.freqz(PA, worN=N, fs=Fs)
plt.plot(w, 20 * np.log10(abs(h)))
plt.title("Filtro Pasa Alto")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Ganancia (dB)")
plt.xlim([0, 5])
plt.grid()

plt.figure()
w, h = signal.freqz(b_notch, a_notch, worN=N, fs=Fs)
plt.plot(w, 20 * np.log10(abs(h)))
plt.title("Filtro Notch")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Ganancia (dB)")
plt.xlim([30, 80])
plt.grid()

plt.figure()
w, h = signal.freqz(PB, worN=N, fs=Fs)
plt.plot(w, 20 * np.log10(abs(h)))
plt.title("Filtro Pasa Bajo")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Ganancia (dB)")
plt.xlim([50, 100])
plt.grid()

# Comparamos la evolución del filtrado
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(n, s)
plt.title("Señal de entrada - ECG original")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.xlim([3, 5])

plt.subplot(4, 1, 2)
plt.plot(n, senal_filtrada_alto)
plt.title("Señal después del filtro pasa alto")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.xlim([3, 5])

plt.subplot(4, 1, 3)
plt.plot(n, senal_filtrada_notch)
plt.title("Señal después del filtro notch")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.xlim([3, 5])

plt.subplot(4, 1, 4)
plt.plot(n, senal_filtrada_bajo)
plt.title("Señal después del filtro pasa bajo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.xlim([3, 5])

# Análisis de espectro de frecuencia antes y después del filtrado
h = np.fft.fft(s)
f = np.arange(len(h))  # Barrido en frecuencia para graficar el espectro original
H = np.fft.fft(senal_filtrada_bajo)
F = np.arange(len(H))  # Barrido en frecuencia para graficar el espectro obtenido

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(f * Fs / max(f), np.abs(h))  # Grafica del espectro de s escalado para FS
plt.title("Espectro de Frecuencia de la Señal Original")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Energía")
plt.xlim([0, 80])
plt.ylim([0, 800])

plt.subplot(2, 1, 2)
plt.plot(F * Fs / max(F), np.abs(H))  # Grafica del espectro de s escalado para FS
plt.title("Espectro de Frecuencia del Resultado del Filtrado")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Energía")
plt.xlim([0, 80])
plt.ylim([0, 800])

# Calcular la autocorrelación de la señal filtrada
autocorrelacion = np.correlate(senal_filtrada_bajo, senal_filtrada_bajo, mode="full")

# Visualizar la autocorrelación
plt.figure()
plt.plot(autocorrelacion)
plt.title("Autocorrelación de la señal filtrada")
plt.xlabel("Retardo")
plt.ylabel("Amplitud")

plt.show()
