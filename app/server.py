from fastai import *
from fastai.vision import *

learn = None

def imagePredictorModel(path_to_export, export_file_name):
	learn = load_learner(path_to_export, export_file_name)
	

def predictCar(request):
	img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)
    return prediction 

def homepage(request):
	return render_template("index.html")