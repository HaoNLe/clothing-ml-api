from flask import Flask, request, jsonify
#from models import classifier
from io import BytesIO
from Pillow import Image
import torchvision.transforms as transforms

from flask_uploads import UploadSet, IMAGES, configure_uploads, ALL
import time
#import cv2
import numpy as np
#import config

import torch
from torchvision import models

device = torch.device("cpu")
classifier = models.resnet18(num_classes=6).to(device)
classifier.load_state_dict(torch.load('resnet18_1st', map_location='cpu'))
classifier.eval()

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/api/classify', methods=['POST'])
def predict():
    data = {'state': False}
    print(request)
    app.logger.info('FILE RECEIVED: %s', request.files)

    img = request.files['fileupload'].read()
    #try:
    #    topk = request.form['topk']
    #except:
    #    topk = 1
    #img = np.fromstring(img, np.uint8)
    #img = cv2.imdecode(img, flags=1)
    #img = cv2.resize(img, (224, 224))
    imarr = np.uint8(np.asarray(Image.open(BytesIO(img)).convert('RGB').resize((224,224))))

    #imarr = np.uint8(np.asarray(im.convert('RGB').resize((224,224))))
    trans = transforms.ToTensor()
    imarr = trans(imarr)
    imarr = imarr.unsqueeze(0)
    
    data = predict_img(imarr)#, is_numpy=True, topk=topk)
    return jsonify(data)

def predict_img(img): #, is_numpy=False, topk=1):
    data = dict()
    start = time.time()
    result = classifier(img) #model.predict(img, is_numpy=is_numpy, topk=int(topk))
    result = result.clone().detach()[0]
    nparr = np.float64(result.numpy())

    cost_time = time.time() - start
    data['predictions'] = list()
    
    for prob in nparr:
        data['predictions'].append(prob)
    #    m_predict = {'label': label, 'probability': ("%.4f" % prob)}
    #    data['predictions'].append(m_predict)
    
    data['state'] = True
    data['time'] = cost_time
    return data

if __name__ == '__main__':
    app.run(debug=True)