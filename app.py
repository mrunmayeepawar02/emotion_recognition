from flask import Flask,render_template,request
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT']=1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET','POST'])
def after():
    img=request.files['file1']
    img.save('D:\web_project\env\static\emot.jpg')
    image=cv2.imread('D:\\web_project\\env\\static\emot.jpg',0)
    image=cv2.resize(image, dsize=[48,48])
    image=np.reshape(image,(1,48,48,1))
    image=image/255
    model=load_model('ResNet50.h5')
    prediction=model.predict(image)
    EMOTIONS = ["angry" ,"disgust","fear", "happy", "sad", "surprised",
 "neutral"]
    prediction=np.argmax(prediction)
    final_prediction=EMOTIONS[prediction]
    return render_template('after.html', data=final_prediction)

if __name__ == '_main_':
    app.run(debug=True)