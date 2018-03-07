# -*- coding: utf-8 -*-
import os,base64,json,datetime
import datetime
from flask import Flask, request,render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES
from CNN_Model import readModel
app = Flask(__name__)
m_predict = readModel.Predict()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        dataList = json.loads(request.data)
        imagedata = base64.b64decode(dataList['image'])
        date = datetime.datetime.now()
        datestr = date.strftime('%Y-%m-%d_%H-%M-%S')
        image_path = './ImageFiles/'+datestr+'.png'
        file=open(image_path,'wb')
        file.write(imagedata)
        file.close()
        full_path=os.getcwd()+'/ImageFiles/'+datestr+'.png'
        return str(m_predict.predict(full_path))
    return render_template('draw.html')


if __name__ == '__main__':
    app.run()
