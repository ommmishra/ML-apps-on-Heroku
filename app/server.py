from fastai import *
from fastai.vision import *

learner = None

def imagePredictorModel(path_to_export, export_file_name):
	learner = load_learner(path_to_export, export_file_name)
	

def predict(request):
	img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))

