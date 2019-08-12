from flask import Flask, request
from fastai import *
from fastai.vision import *
from gevent.pywsgi import WSGIServer

app = Flask(__name__, static_url_path="/static")

# path = Path("/home/omm/WorkStuff/herokuDeploy/ML-apps-on-Heroku/app/models")
# imgPath = Path("/home/omm/Downloads/wall-1564639504.png")

try:
	learn = load_learner(path_to_export)
	print('yes')
except RuntimeError as e:
	print(e)

@app.route("/", methods=['GET'])
def index():
	return render_template('index.html')


@app.route("/analyze", methods=['POST']) 	
def predictCar(learn):
	img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    predictionSet = learn.predict(img)
    prediction = predictionSet(img)[0] 
    return jsonify('result': str(prediction))

# def homepage(request):
	# return render_template("index.html")
# defaults.device = torch.device('cpu')
# learn = imagePredictorModel(path)
# x = predictCar(learn, imgPath)
# print(x[2])
# print(type(x[2]))

if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	http_server = WSGIServer(('',port),app)
	http_server.serve_forever()