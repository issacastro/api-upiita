from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth('settings/Drive.yaml')
drive = GoogleDrive(gauth)

def Upload(index,file,folder,id):
    if folder == 'México':
        folder_id= "1XnIjkzbYKxHNuiDxfuyCTZbcVfAqVvW_"
    if folder == 'Argentina':
        folder_id= "1mVO0brmw16xr6LAamazRa8xaWUSgSEfW"
    if folder == 'Colombia':
        folder_id= "1H6FJnbKlFiijRSo5ZSsUdWmUSiujLw8n"
    if folder == 'Otro':
        #folder_id= "1x_lhZed8s7jQi7uj-H8J79cFmFckdvCy" 
        folder_id= "1jeJF1AS_F8ZLXrLPta7Zkj2GAycl1DOk"   
    Audio = drive.CreateFile({'parents': [{'id': folder_id}]})
    Audio.SetContentFile(file)
    #Set the name 
    Audio['title'] = "{}-{}.wav".format(id,index)
    Audio.Upload()
    print('Created file {} with mimeType {}'.format(Audio['title'],Audio['mimeType']))
