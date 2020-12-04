import os
import pymongo 
import shutil
import uuid 
import json
from bson import json_util
from dotenv import load_dotenv
from flask import Flask, flash, request, redirect, url_for
from flask_cors import  CORS
from werkzeug.utils import secure_filename
from datauri import DataURI
from bson.objectid import ObjectId
from gevent.pywsgi import WSGIServer

import functions.Drive as Drive
import functions.Resampling as R
import functions.Analysis as A

#Cargamos las variabes de entorno
load_dotenv()
URI = os.getenv('MONGODB_URI')

#Configuramos los folders y archivmos permitidos
UPLOAD_FOLDER = 'audios'
TEST_FOLDER = 'test'
ALLOWED_EXTENSIONS = {'wav'}

#Inicializamos la base de datos 
client = pymongo.MongoClient(URI)
db = client['upiita'] 

#Iniciamos la aplicacion
app =Flask(__name__)
#Configuramos los CORS para permitir informacion de otros servidores
CORS(app)
app.secret_key = "secret key" 

#Seteamo las configuraciones de la aplicacion

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
#app.config["DEBUG"]= True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEST_FOLDER'] = TEST_FOLDER

#Funciones del funcionamiento de la aplicacion
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_json(data):
    return json.loads(json_util.dumps(data))


#Rutas de la aplicacion para hacer requesst
@app.route('/',methods=['GET'])
def home():
    return "<h1>Hello,World!"


@app.route('/analisis/<id>',methods=['GET'])
def page(id):
    pageid = ObjectId(id)
    query = db['test']
    register = query.find_one({"_id": pageid})
    return  parse_json(register)
    

@app.route('/analisis',methods=['GET','POST'])
def analisis():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        #Caching the file and data form for processing   
        data = request.form  
        file = request.files['file']
        print("{} - {} - {}".format(data['name'],data['gender'],data['type']))
        #Creating register
        id = uuid.uuid4()
        #Saving the file in the Test Directory
        
        path = os.path.join(app.config['TEST_FOLDER'],str(id))
        os.mkdir(path)
        path_files=[]
        if data['type'] =="Adjuntar":
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path_file = os.path.join(path, filename)
                file.save(path_file)
                R.resamplig(path_file)
                path_files.append(path_file)
        if data['type']=="Grabar":
            files = request.files.getlist('file') 
            for index,file in enumerate(files,start=1):
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    path_file = os.path.join(path, filename)
                    file.save(path_file)
                    R.resamplig(path_file)
                    path_files.append(path_file)            
        results = []
        print(path_files[0])
        for feature_extraction in os.listdir("resources"):
            models = os.path.join("resources",feature_extraction,"models")
            for ANN in os.listdir(models):
                result,WN,WNR = A.Predict(feature_extraction,ANN.split(".")[0],path_files[0],path_files[1],id)
                Test = {"ANN":ANN.split("_")[0],"Feature":feature_extraction.split("_")[0],"countryN":result[0],"N":WN,"countryWNR":result[1],"WNR":WNR,}
                results.append(Test)
        
        #query = db['analysis']
        SNR1, SNR2 = A.Procesar(path_files[0],path_files[1],path)
        print("SNR1:{} - SNR2:{}".format(SNR1,SNR2))
        query = db['test']
        audio = DataURI.from_file(path_files[0])
        audio_noise = DataURI.from_file(os.path.join(path,"procesada.wav"))
        image = DataURI.from_file(os.path.join(path,"voz.png"))
        image_noise = DataURI.from_file(os.path.join(path,"procesada.png"))
        result = {"Test":data['name'],"Data":results,"Audio":audio,"NoiseReduction":audio_noise,"Voice":image,"VoiceProcessing":image_noise,"SNR1":SNR1,"SNR2":SNR2}
        db_id = query.insert_one(result).inserted_id
        #for index,file in enumerate(os.listdir(path),start=1):
        #    if file and allowed_file(file):
        #        path_file = os.path.join(path, file)
        #        Drive.Upload(index,path_file,'Otro',db_id)
        register = query.find_one({"_id":db_id})
        shutil.rmtree(path)
        return  parse_json(register)
    return "Analisis de Auidos..."


@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        #Catching the files and form
        files = request.files.getlist('file')
        data = request.form
        #Insert the data into the databases
        print("{} - {} - {} - {}".format(data['name'],data['gender'],data['country'],data['old']))
        User = {"name":data['name'],"gender":data['gender'],"country":data['country'],"old":data['old']}
        query = db['users']

        #Getting the id from databases
        id = query.insert_one(User).inserted_id
        
        #Creating the temporary directoryto upload the corresponding files to Drive
        path = os.path.join(app.config['UPLOAD_FOLDER'],str(id))
        os.mkdir(path)

        #Iterating each file to resampling and upload to Drive
        for index,file in enumerate(files,start=1):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path_file = os.path.join(path, filename)
                file.save(path_file)
                R.resamplig(path_file)
                Drive.Upload(index,path_file,User['country'],id)

        shutil.rmtree(path)
        register = query.find_one({"_id":id})
        return  parse_json(register)
    return "Upload Files..."

if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
