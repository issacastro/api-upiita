import os
import csv
import datetime
import sys
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf
import functions.feature_extraction_scripts.feature_extraction_functions as featfun
import functions.feature_extraction_scripts.prep_noise as pn
from keras.models import load_model
from keras import backend as K
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def str2bool(bool_string):
    bool_string = bool_string=="True"
    return bool_string

def Predict(feature,ann,file,file_noise,id):

    head_folder_beg = "resources/"
    head_folder_curr_project = head_folder_beg+feature
    
    #cargar la información relacionada con las características y el modelo de interés
    features_info_path = head_folder_curr_project+"/features_log.csv"
    encoded_label_path = head_folder_curr_project+"/labels_encoded.csv"
    model_path =  head_folder_curr_project+"/models/{}.h5".format(ann)
    model_log_path = head_folder_curr_project+"/model_logs/{}.csv".format(ann)
    
    #averiguar los ajustes para la extracción de características
    with open(features_info_path, mode='r') as infile:
        reader = csv.reader(infile)            
        feats_dict = {rows[0]:rows[1] for rows in reader}
    feature_type = feats_dict['features']
    num_filters = int(feats_dict['num original features'])
    num_feature_columns = int(feats_dict['num total features'])
    delta = str2bool(feats_dict["delta"])
    dom_freq = str2bool(feats_dict["dominant frequency"])
    noise = str2bool(feats_dict["noise"])
    vad = str2bool(feats_dict["beginning silence removal"])
    timesteps = int(feats_dict['timesteps'])
    context_window = int(feats_dict['context window'])
    frame_width = context_window*2+1
    
    
    #preparar el diccionario para averiguar la etiqueta asignada 
    with open(encoded_label_path, mode='r') as infile:
        reader = csv.reader(infile)            
        dict_labels_encoded = {rows[0]:rows[1] for rows in reader}
        
    #Inicializando datos
    recording_folder ="test"
    sr=16000
    speech_filename = file
    if file_noise =="":
        noise_filename="test/noise.wav"
    else:
        noise_filename = file_noise
  
    y_speech, sr = librosa.load(speech_filename,sr=sr)
    y_noise, sr = librosa.load(noise_filename,sr=sr)
    
    speech_rd = pn.rednoise(y_speech,y_noise,sr)
    speech_rd_filename = "{}/{}/NoiseReduction.wav".format(recording_folder,id)
    if not os.path.isfile(speech_rd_filename):
        sf.write(speech_rd_filename,speech_rd,sr)
    
    
    features = featfun.coll_feats_manage_timestep(timesteps,frame_width,speech_filename,feature_type,num_filters,num_feature_columns,recording_folder,delta=delta,dom_freq=dom_freq,noise_wavefile=None,vad=vad)
    
    features2 = featfun.coll_feats_manage_timestep(timesteps,frame_width,speech_rd_filename,feature_type,num_filters,num_feature_columns,recording_folder,delta=delta,dom_freq=dom_freq,noise_wavefile=None,vad=vad)
    
    
    #Necesitamos remodelar los datos de varios modelos...
    #...para averiguar qué modelos:
    with open(model_log_path, mode='r') as infile:
        reader = csv.reader(infile)            
        dict_model_settings = {rows[0]:rows[1] for rows in reader}
        
    model_type = dict_model_settings["model type"]
    activation_output = dict_model_settings["activation output"]
    
    

    X = features
    if model_type == "lstm" or model_type == "bilstm":
        X = X.reshape((timesteps,frame_width,X.shape[1]))
    elif model_type == "cnn":
        X = X.reshape((X.shape[0],X.shape[1],1))
        X = X.reshape((1,)+X.shape)
    elif model_type == "cnnlstm" or model_type == "cnnbilstm":
        X = X.reshape((timesteps,frame_width,X.shape[1],1))
        X = X.reshape((1,)+X.shape)        
    
    
    #load model
    with tf.Session(graph=K.get_session().graph) as session:
        session.run(tf.global_variables_initializer())
        model = load_model(model_path)
        results = []
        print("Proccessing file....")
        prediction = model.predict(X)
        pred = str(np.argmax(prediction[0]))
        WN = prediction[0][int(pred)]*100
        
        label = dict_labels_encoded[pred]
        print("Without NR {} {}%".format(label,WN))
        results.append(label)        
        X = features2
        if model_type == "lstm" or model_type == "bilstm":
            X = X.reshape((timesteps,frame_width,X.shape[1]))
        elif model_type == "cnn":
            X = X.reshape((X.shape[0],X.shape[1],1))
            X = X.reshape((1,)+X.shape)
        elif model_type == "cnnlstm" or model_type == "cnnbilstm":
            X = X.reshape((timesteps,frame_width,X.shape[1],1))
            X = X.reshape((1,)+X.shape)
            
        print("Proccessing file....")
        prediction = model.predict(X)
        # mostrar las entradas y salidas previstas
        pred = str(np.argmax(prediction[0]))
        WNR = prediction[0][int(pred)]*100
        label = dict_labels_encoded[pred]
        print("With WNR {} {}%".format(label,WNR))
        results.append(label)   
   
    return results,WN,WNR




def Procesar(speech_filename,noise_filename,path):
    plt.rcParams["figure.figsize"] = (10,5)
    sr=16000
    y_speech, sr = librosa.load(speech_filename,sr=sr)  
    y_noise, sr = librosa.load(noise_filename,sr=sr)
    t= np.arange(0,len(y_speech)/sr,1/sr)

    #Señal sin procesar
    plot1 =plt.figure(1)
    plt.title("Señal de voz")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.grid()
    plt.plot(t,y_speech)
    plt.savefig('{}/voz.png'.format(path))
    Max_sample = max(y_speech)
    Min_sample = min(y_speech)
    #Señal de ruido
    plot2 =plt.figure(2)
    plt.title("Señal de ruido")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.grid()
    plt.plot(t,y_noise)

    #Relacion SNR priori
    ps_pri = np.sum(np.power(y_speech,2))
    pn_pri = np.sum(np.power(y_noise,2))
    SNR1 = 10*np.log10(ps_pri/pn_pri)


    #Reduccion de rudio
    y_red = pn.rednoise(y_speech,y_noise,sr)
    for i in range(len(y_red)):  
        if y_red[i] > Max_sample:
            y_red[i] = 1e-3
        elif y_red[i] < Min_sample:
            y_red[i] = 1e-3


    #Relacion SNR posteriori
    y_red_noise = pn.get_noise_samples(y_red,sr)[0]
    ps_post = np.sum(np.power(y_red,2))
    pn_post = np.sum(np.power(y_red_noise,2))
    SNR2 = 10*np.log10(ps_post/pn_post)



    plot3 =plt.figure(3)
    plt.title("Señal con reducción de ruido")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.grid()
    plt.plot(t,y_red)
    #sf.write("{}/NoiseReduction.wav".format(path),y_red,sr)

    #Señal con eliminacion de silencios
    y_sil = pn.get_speech_samples(y_red,sr)[0]

    plot4 =plt.figure(4)
    plt.title("Señal de voz procesada")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.grid()
    t_sil = np.arange(0,len(y_sil)/sr,1/sr)
    plt.plot(t_sil,y_sil)
    plt.savefig('{}/procesada.png'.format(path))
    sf.write("{}/procesada.wav".format(path),y_sil,sr)

    t=t_sil
    pre_emphasis = 0.97
    #Filtro preenfasis
    y_filt = np.append(y_sil[0], y_sil[1:] - pre_emphasis * y_sil[:-1])
    #sf.write("{}/emphasized.wav".format(path),y_filt,sr)

    plot5 =plt.figure(5)
    plt.title("Señal de voz con filtro preenfasis")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.grid()
    plt.plot(t,y_filt)

    #Comparacion de filtro preenfasis
    plot6 =plt.figure(6)
    plt.title("Señal de voz con filtro preenfasis")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.grid()
    plt.plot(t,y_filt,label="Preénfasis")
    plt.plot(t,y_sil,label="Original")
    plt.legend(loc="upper left")


    #Señal normalizada
    y_normal = pn.normalize(y_filt)

    plot7 =plt.figure(7)
    plt.title("Señal de voz Normalizada")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.grid()
    plt.plot(t,y_normal)



    #Señal fragmentada
    window = 0.025
    length_frame = int(window*sr)
    n_frame =  int(np.floor(len(y_normal)/length_frame))
    samples_frame = n_frame*length_frame
    y_fragment = y_normal[0:samples_frame]
    t_fragment = np.arange(0,len(y_fragment)/sr,1/sr)
    Frames = np.split(y_fragment,n_frame)
    t_frame = np.split(t_fragment,n_frame)

    plot8 =plt.figure(8)
    plt.title("Frames de la señal")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.grid()
    for index in range(len(Frames)):
        plt.plot(t_frame[index],Frames[index])


    plot9 =plt.figure(9)
    plt.title("Frame de la señal")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.grid()
    plt.plot(t_frame[10],Frames[10])

    #Ventana de Hamming

    hamming = np.hamming(length_frame)

    plot10 =plt.figure(10)
    plt.title("Venta de Hamming")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.grid()
    plt.plot(t_frame[1],hamming)


    #Apliacion de la ventana de hamming
    Frames_hamming = Frames
    y_hamming = np.array([0.0])
    for index in range(len(Frames)):    
        Frames_hamming[index] = Frames[index]*hamming


    plot11 =plt.figure(11)
    plt.title("Frames de la señal con hamming")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.grid()
    for index in range(len(Frames_hamming)):
        plt.plot(t_frame[index],Frames_hamming[index])


    fig,(ax1,ax2) = plt.subplots(2)
    plt.tight_layout()
    ax1.plot(t_frame[12],Frames[12])
    ax1.set_title("Sin ventana de Hamming")
    plt.tight_layout()
    ax2.plot(t_frame[12],Frames_hamming[12])
    ax2.set_title("Con ventana de Hamming")
    plt.tight_layout()    
    plt.clf()

    return SNR1, SNR2
