# -*- coding: utf-8 -*-


import sys
sys.path.append('./CNN/model')

import os,base64,json,datetime
import datetime
from flask import Flask, request,render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES
import mnistLeNet
app = Flask(__name__)
m_predict = mnistLeNet.LeNetModel_io()

@app.route('/mnist', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        dataList = json.loads(request.data.decode('utf-8'))
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

""" @app.route('/fastrcnn',method=['GET','POST'])
def fastrcnn_recoginze():
    if request.method == 'POST':
        #读取图片将其保存到服务器，然后调用识别的代码进行识别，返回识别结果包括：
        #1. 识别范围；2.识别类别名称；3.结果的可靠性
        #前端的代码让文素帮个忙好了
    return render_template('draw.html') """


if __name__ == '__main__':
    app.run()
    
