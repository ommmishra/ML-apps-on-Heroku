from fastai import *
from fastai.vision import *

path = Path("/home/omm/WorkStuff/herokuDeploy/ML-apps-on-Heroku/app/models")
imgPath = Path("/home/omm/Downloads/wall-1564639504.png")

learn = None
def imagePredictorModel(path_to_export):
	learn = load_learner(path_to_export)
	print('yes')
	return learn

def predictCar(learn, imgPath):
	# img_data = await request.form()
 #    img_bytes = await (img_data['file'].read())
 #    img = open_image(BytesIO(img_bytes))
    img = open_image(imgPath)
    pred_class, pred_tensor, predictionSet = learn.predict(img)
    return predictionSet 

# def homepage(request):
	# return render_template("index.html")
defaults.device = torch.device('cpu')
learn = imagePredictorModel(path)
x = predictCar(learn, imgPath)
print(x[2])
print(type(x[2]))