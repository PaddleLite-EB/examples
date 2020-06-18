
from paddlelite import *
import numpy as np
import time, os, json
import cv2
import struct

from collections import namedtuple

def test(predictor):
	i = predictor.get_input(0);
	i.resize((1, 3, 224, 224));
	z = np.zeros((1, 224, 224, 3)).astype(np.float32)
	z = z.reshape(1, 3, 224, 224);
	i.set_data(z)

	predictor.run();

def read_labels(j):
	f = open(j["labels"], 'r+')
	lines = [];
	for line in f.readlines():
		line = line.strip();
		lines.append(line)
	f.close()
	return lines;

def post_process(j, t, out):
	data = out.data()[0];
	index = np.argmax(data)
	score = data[index]
	labels = read_labels(j);
	print("index = {} lable = {} score = {}".format(index, labels[index], score))


def preprocess(j, t):
	image_path = t["image"]
	shape = t["input_shape"]
	img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
	img = cv2.resize(img, (shape[2], shape[1]))
	if j["colorFormat"] == "RGB":
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BGR -> RGB
	img = np.array(img).astype(np.float32)

	img -= np.array(j["img_mean"]).reshape((1, 1, 3))
	img *= np.array(j["scale"]).reshape((1, 1, 3))
	return img;

def load_model(model_dir):
	valid_places = (
		Place(TargetType.kFPGA, PrecisionType.kFP16, DataLayoutType.kNHWC),
		Place(TargetType.kHost, PrecisionType.kFloat),
		Place(TargetType.kARM, PrecisionType.kFloat),
	);
	config = CxxConfig();
	config.set_model_file(model_dir + "/model")
	config.set_param_file(model_dir + "/params")
	config.set_valid_places(valid_places);
	predictor = CreatePaddlePredictor(config);
	test(predictor);
	return predictor;

def predict(predictor, j, t):
	img = preprocess(j, t);
	i = predictor.get_input(0);
	shape = shape = t["input_shape"]
	i.resize((1, 3, shape[2], shape[1]));

	z = np.zeros((1, shape[1], shape[2], 3)).astype(np.float32)
	z[0, 0:img.shape[0], 0:img.shape[1] + 0, 0:img.shape[2]] = img
	z = z.reshape(1, 3, shape[1], shape[2]);
	i.set_data(z)

	predictor.run();
	out = predictor.get_output(0);
	return out;

def test_model(j):
	model_dir = j["model"]
	predictor = load_model(model_dir)
	for t in j["tests"]:
		out = predict(predictor, j, t)
		post_process(j, t, out)
	
def main(config_file):
	with open(config_file) as f:
		j = json.load(f)
		test_model(j)

config_file = "vega/preprocessor.json"

main(config_file)




