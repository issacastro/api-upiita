'''
Se trata de una serie de funciones para transformar la forma del habla del dominio del tiempo y la amplitud (forma de onda) al dominio de la frecuencia y la potencia (MFCC, FBAN, STFT).

Nota: estas funciones estarían mejor configuradas dentro de una Clase.
Para estas pruebas pensé que dejarlas como funciones individuales sería más sencillo, pero no creo que ese sea el caso :P 
'''

#save info
import csv
import sys
from pathlib import Path

#audio 
import librosa
import librosa.display
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#data prep
import numpy as np
import random

#Detección de actividad de voz
import functions.feature_extraction_scripts.prep_noise as prep_data_vad_noise
from functions.feature_extraction_scripts.errors import NoSpeechDetected, LimitTooSmall,FeatureExtractionFail



 
#Delta
def get_change_acceleration_rate(spectro_data):
    #primera derivada = delta (tasa de cambio)
    delta = librosa.feature.delta(spectro_data)
    #segunda derivada = delta delta (cambios de aceleración)
    delta_delta = librosa.feature.delta(spectro_data,order=2)

    return delta, delta_delta

    
#Cargar el archivo wav, establecer los ajustes
def get_samps(wavefile,sr=None,high_quality=None):
    if sr is None:
        sr = 16000
    if high_quality:
        quality = "kaiser_high"
    else:
        quality = "kaiser_fast"
    y, sr = librosa.load(wavefile,sr=sr,res_type=quality) 
    
    return y, sr

#Establecer los ajustes para la extracción de mfcc
def get_mfcc(y,sr,num_mfcc=None,window_size=None, window_shift=None):
    '''
    Valores de ajuste: por defecto para la extracción de los MFCC:
    - 40 MFCCs
    - ventanas de 25ms 
    - desplazamientos de la ventana de 10ms
    '''
    if num_mfcc is None:
        num_mfcc = 40
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
    mfccs = librosa.feature.mfcc(y,sr,n_mfcc=num_mfcc,hop_length=hop_length,n_fft=n_fft)
    mfccs = np.transpose(mfccs)
    mfccs -= (np.mean(mfccs, axis=0) + 1e-8)
    
    return mfccs

#Obtener el banco, y establecer la configuración
def get_mel_spectrogram(y,sr,num_mels = None,window_size=None, window_shift=None):
    '''
    Valores de ajuste: por defecto para el cálculo del espectrograma de mel (FBANK)
    - ventanas de 25ms 
    - desplazamientos de la ventana de 10ms
    '''
    if num_mels is None:
        num_mels = 40
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
        
    fbank = librosa.feature.melspectrogram(y,sr,n_fft=n_fft,hop_length=hop_length,n_mels=num_mels)
    fbank = np.transpose(fbank)
    fbank -= (np.mean(fbank, axis=0) + 1e-8)
    
    return fbank

#Consigue el stft y ajusta la configuración si quieres# 
#Nota: No me he metido con el tamaño de la ventana o el desplazamiento aquí.
#Si cambias esto, puede que tengas que ajustar el número de características por defecto 
#columnas asignadas a stft en el módulo principal (ver a la derecha abajo def main())
def get_stft(y,sr,window_size=None, window_shift=None):
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
    stft = np.abs(librosa.stft(y,n_fft=n_fft,hop_length=hop_length)) #viene en números complejos... tiene que tomar valor absoluto...
    stft = np.transpose(stft)
    stft -= (np.mean(stft, axis=0) + 1e-8)
    
    return stft

#super experimental. Quería la frecuencia fundamental pero esto era más fácil
def get_domfreq(y,sr):
    '''
    recogiendo las frecuencias de mayor magnitud
    '''
    frequencies, magnitudes = get_freq_mag(y,sr)
    #seleccionar sólo las frecuencias de mayor magnitud, es decir, la frecuencia dominante
    dom_freq_index = [np.argmax(item) for item in magnitudes]
    dom_freq = np.array([frequencies[i][item] for i,item in enumerate(dom_freq_index)])
    #dom_freq -= (np.mean(dom_freq, axis=0) + 1e-8)
    
    return np.array(dom_freq)

#Obtener una colección de frecuencias en las mismas ventanas que otras técnicas de extracción, es decir, 25ms con desplazamientos de 10ms (lo cual es estándar para mucha investigación)
#Esto se puede ajustar aquí .. este guión está preparado para estos ajustes de la ventana
#Puede que funcione con otros, pero aún no lo he probado..
def get_freq_mag(y,sr,window_size=None, window_shift=None):
    '''
    valores por defecto:
    - ventanas de 25ms 
    - desplazamientos de la ventana de 10ms
    '''
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
    #Recoger las frecuencias presentes y sus magnitudes
    frequencies,magnitudes = librosa.piptrack(y,sr,hop_length=hop_length,n_fft=n_fft)
    frequencies = np.transpose(frequencies)
    magnitudes = np.transpose(magnitudes)
    
    return frequencies, magnitudes

#Guardar un montón de características en la forma exacta que quería era más fácil con los archivos .npy. Es rápido de guardar y rápido de cargar.
def save_feats2npy(labels_class,dict_labels_encoded,data_filename4saving,max_num_samples,dict_class_dataset_index_list,paths_list,labels_list,feature_type,num_filters,num_feature_columns,time_step,frame_width,head_folder,limit=None,delta=False,dom_freq=False,noise_wavefile=None,vad=False,dataset_index=None):
    if dataset_index is None:
        dataset_index = 0
    #dataset_index representa los conjuntos de datos de tren (0), val (1) o prueba (2)

    #crear una matriz vacía para llenarla con valores
    if limit:
        max_num_samples = int(max_num_samples*limit)
        expected_rows = max_num_samples*len(labels_class)*frame_width*time_step
    else:
        expected_rows = max_num_samples*len(labels_class)*frame_width*time_step
    feats_matrix = np.zeros((expected_rows,num_feature_columns+1)) # +1 for the label
    
    #actualizar al usuario lo que está pasando:
    msg = "\nExtracción de características: Sección {} de 3\nExtrayendo característica de: {} archivos wav por clase.\nCon {} clases, Procesando.. {} archivos wav.\nLas características se guardarán en el archivo {}.npy\n\n".format(dataset_index+1,max_num_samples,len(labels_class),len(labels_class)*max_num_samples,data_filename4saving)
    print(msg)
    
    #Revisa todos los datos del conjunto de datos y rellena la matriz
    row = 0
    #Esta fila indica cuán lejos a lo largo de la matriz vacía se está llenando
    completed = False
    # si las funciones terminan antes de tiempo, volverá que no se completó.
    
    try:
        if expected_rows < 1*frame_width*time_step:
            #Una vez puse el límite en 0 por accidente... lo cual no tiene sentido.
            raise LimitTooSmall("\nAumentar el límite: El límite en '{}' es demasiado pequeño.".upper().format(limit))
        
        #Poner los caminos de las ondas y sus etiquetas juntas en una lista de tuplas# ¡Asegúrate de que no se separen!
        #Esta lista será iterada, y cada par de archivo de ondas/etiquetas será procesado conjuntamente#
        paths_labels_list_dataset = []
        for i, label in enumerate(labels_class):
            '''
            Nota: He equilibrado los datos en base a la clase/etiqueta. Por lo tanto,
            los archivos de onda en cada clase han sido asignados, por igual y a 
            al azar, para entrenar, validar y probar los conjuntos de datos. 
            Sé que esto es un poco confuso pero....
            aquí estoy recogiendo todos los pares de archivos de onda y etiquetas 
            de cada clase, para cada sección (tren, val, prueba).
            Esta función recoge esos pares sólo para una de esas secciones 
            a la vez: la variable 'dataset_index' aquí representa el 
            sección actual (es decir, 0 == tren, 1 == validación, 2 == prueba)
            '''
            train_val_test_index_list = dict_class_dataset_index_list[label]
            
            for k in train_val_test_index_list[dataset_index]:
                paths_labels_list_dataset.append((paths_list[k],labels_list[k]))
        
        #shuffle indices:
        #¡Esto es importante! De lo contrario el algoritmo aprenderá basado en 
        #etiqueta/orden de clase (como ordené la lista anterior por clase)
        random.shuffle(paths_labels_list_dataset)
        
        for wav_label in paths_labels_list_dataset:

            if row >= feats_matrix.shape[0]:
                # Esto significa que hemos llenado la matriz! ¡Yaay!
                break
            else:
                wav_curr = wav_label[0]
                label_curr = wav_label[1]
                #integros codifican la etiqueta:
                label_encoded = dict_labels_encoded[label_curr]
                
                #La función abajo básicamente extrae las características y se asegura de que las características de cada muestra son del mismo tamaño: se cortan.
                #si es demasiado largo y cero acolchado si es demasiado corto
                feats = coll_feats_manage_timestep(time_step,frame_width,wav_curr,feature_type,num_filters,num_feature_columns,head_folder,delta=delta,dom_freq=dom_freq, noise_wavefile=noise_wavefile,vad = vad)
                
                #Añadir columna de etiquetas - ¡Necesito etiquetas para mantener las características!
                label_col = np.full((feats.shape[0],1),label_encoded)
                feats = np.concatenate((feats,label_col),axis=1)
                
                #Llenar la matriz con los rasgos que se acaban de recoger#
                feats_matrix[row:row+feats.shape[0]] = feats
                
                #actualizar la fila para el siguiente conjunto de características para llenarla con
                row += feats.shape[0]
                
                #imprime en la pantalla el progreso
                progress = row / expected_rows * 100
                sys.stdout.write("\r%d%% completo de progreso de la sección actual" % progress)
                sys.stdout.flush()
        print("\nFila alcanzada: {}\nTamaño de la matriz: {}\n".format(row,feats_matrix.shape))
        completed = True
    
    except LimitTooSmall as e:
        print(e)

    finally:
        np.save(data_filename4saving+".npy",feats_matrix)
        
    return completed


#Esta función alimenta con variables a la función de extracción de características 'get_feats' (y da forma a los datos al mismo tamaño). 
#También es un hermoso ejemplo de por qué las clases son geniales. Elegí no hacer una clase para estas funciones porque pensé que sería más sencillo, para un taller....
def coll_feats_manage_timestep(time_step,frame_width,wav,feature_type,num_filters,num_feature_columns,head_folder,delta=False,dom_freq=False,noise_wavefile=None,vad = True):
    feats = get_feats(wav,feature_type,num_filters,num_feature_columns,head_folder,delta=delta,dom_freq=dom_freq,noise_wavefile=noise_wavefile,vad = vad)
    max_len = frame_width*time_step
    if feats.shape[0] < max_len:
        diff = max_len - feats.shape[0]
        feats = np.concatenate((feats,np.zeros((diff,feats.shape[1]))),axis=0)
    else:
        feats = feats[:max_len,:]
    
    return feats

#Ruido
#por defecto lo aplica con diferentes fuerzas. Puedes fijarlo a un cierto nivel aquí:
def apply_noise(y,sr,wavefile):
    # al azar aplicar cantidades variables de ruido ambiental
    rand_scale = random.choice([0.0,0.25,0.5,0.75])
    #rand_scale = 0.75
    if rand_scale > 0.0:
        total_length = len(y)/sr
        y_noise,sr = librosa.load(wavefile,sr=16000)
        envnoise_normalized = prep_data_vad_noise.normalize(y_noise)
        envnoise_scaled = prep_data_vad_noise.scale_noise(envnoise_normalized,rand_scale)
        envnoise_matched = prep_data_vad_noise.match_length(envnoise_scaled,sr,total_length)
        if len(envnoise_matched) != len(y):
            diff = int(len(y) - len(envnoise_matched))
            if diff < 0:
                envnoise_matched = envnoise_matched[:diff]
            else:
                envnoise_matched = np.append(envnoise_matched,np.zeros(diff,))
        y += envnoise_matched

    return y

#colecciona las características reales, de acuerdo con la configuración asignada
#como con el ruido, la detección de actividad de la voz/eliminación del silencio inicial, etc.
#mfcc, fbank, stft, delta, dom_freq
def get_feats(wavefile,feature_type,num_features,num_feature_columns,head_folder,delta=False,dom_freq=False,noise_wavefile = None,vad = False):
    y, sr = get_samps(wavefile)

    if vad:
        try:
            y, speech = prep_data_vad_noise.get_speech_samples(y,sr)
            if speech:
                pass
            else:
                raise NoSpeechDetected("\n!!! FYI: No se detectó ningún discurso en el archivo: {} !!!\n".format(wavefile))
        except NoSpeechDetected as e:
            print("\n{}".format(e))
            filename = '{}/no_speech_detected.csv'.format(head_folder)
            with open(filename,'a') as f:
                w = csv.writer(f)
                w.writerow([wavefile])
            
    if noise_wavefile:
        y = apply_noise(y,sr,noise_wavefile)
        
    extracted = []
    if "mfcc" in feature_type.lower():
        extracted.append("mfcc")
        features = get_mfcc(y,sr,num_mfcc=num_features)
        if delta:
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    elif "fbank" in feature_type.lower():
        extracted.append("fbank")
        features = get_mel_spectrogram(y,sr,num_mels = num_features)
        if delta:
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    elif "stft" in feature_type.lower():
        extracted.append("stft")
        features = get_stft(y,sr)
        if delta:
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    if dom_freq:
        dom_freq = get_domfreq(y,sr)
        dom_freq = dom_freq.reshape((dom_freq.shape+(1,)))
        features = np.concatenate((features,dom_freq),axis=1)
    if features.shape[1] != num_feature_columns: 
        raise FeatureExtractionFail("The file '{}' results in the incorrect  number of columns (should be {} columns): shape {}".format(wavefile,num_feature_columns,features.shape))
    
    return features


#Sólo para propósitos de visualización: guardar en el png cómo se ven los rasgos. Usado en el script 'visualize_features.py'.
def save2png(time_step,frame_width,wav,feature_type,num_filters,num_feature_columns,head_folder,delta=False,dom_freq=False,noise_wavefile=None,vad = True):
    feats = coll_feats_manage_timestep(time_step,frame_width,wav,feature_type,num_filters,num_feature_columns,head_folder,delta=delta,dom_freq=dom_freq,noise_wavefile=noise_wavefile,vad = vad)
    
    #transponer los rasgos para ir de izquierda a derecha en el tiempo:
    feats = np.transpose(feats)
    
    #Crea un gráfico y guárdalo en el png
    plt.clf()
    librosa.display.specshow(feats)
    if noise_wavefile:
        noise = True
    else:
        noise = False
    plt.title("{}: {} pasos de tiempo, ancho de cuadro de {}".format(wav,time_step,frame_width))
    plt.tight_layout(pad=0)
    pic_path = "{}{}_vad{}_noise{}_delta{}_domfreq{}".format(feature_type,num_feature_columns,vad,noise,delta,dom_freq)
    path = unique_path(Path(head_folder), pic_path+"{:03d}.png")
    plt.savefig(path)

    return True

def unique_path(directory, name_pattern):
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path

