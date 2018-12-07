'''
Copyright 2016 Satish Palaniappan, Siddharth G, Vijayalakshimi MLS

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
from PIL import Image
import time
import sys
import os
import caffe
from SelectiveSearch import Pokemon_Segmenter
from collections import defaultdict
import numpy as np
import pickle
import random

python3 = "/home/satish/anaconda3/bin/python3.4"

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def clean():
	import os, shutil
	for folder in ["Output/Stage1_SelectiveSearch/","Output/Stage1_SelectiveSearch/Clips","Output/Stage2_PokeNoPoke/Other","Output/Stage2_PokeNoPoke/Partly_Pokemon","Output/Stage2_PokeNoPoke/Pokemon","Output/Stage3_Merged_Filtered/", "Output/Stage3_Merged_Filtered/Clips", "Output/Stage3_Merged_Filtered/Flitered_Final", "Output/Stage4_Pokemon_Label"]:
		for the_file in os.listdir(folder):
			file_path = os.path.join(folder, the_file)
			try:
				if os.path.isfile(file_path):
					os.unlink(file_path)
			except Exception, e:
				print e

def getPokemonClass():
	FileList = ["Output/Stage3_Merged_Filtered/Flitered_Final/"+i for i in os.listdir("Output/Stage3_Merged_Filtered/Flitered_Final/")]
	if FileList == []:
		FileList = ["Sample_Images/"+sys.argv[2]]

	caffe.set_mode_gpu()
	classifier = caffe.Classifier("Models/Classify_Pokemon/deploy.prototxt","Models/Classify_Pokemon/bvlc_googlenet_pokenet_iter_64000.caffemodel" ,
	        image_dims=[224,224], raw_scale=255.0, channel_swap=[2,1,0])

	inputs = [caffe.io.load_image(im_f) for im_f in FileList]
	predictions = classifier.predict(inputs)

	pokelist = []
	score = defaultdict(int)
	index = 0
	for prediction in predictions:
		l = np.array(sorted([i.strip() for i in open("Kanto.txt").readlines()]))
		idx = list((-prediction).argsort())[:10]
		print("\n The top 10 predictions for " + FileList[index])
		print(l[np.array(idx)])
		print("\n Thier Scores:")
		for li,pj in zip(l[np.array(idx)],prediction[np.array(idx)]):
			print(li + ": " + str(pj))
		if prediction[np.array(idx)][0] > 0.50:
			pokelist.append(l[np.array(idx)][0])
			imgt = Image.open(FileList[index])
			imgt.save("Output/Stage4_Pokemon_Label/" + l[np.array(idx)][0] + ".png")
			score[l[np.array(idx)][0]]= max(score[l[np.array(idx)][0]], prediction[np.array(idx)][0])
		index+=1
	if len(pokelist) == 0:
		index = 0
		print("\n Considering low scored results too.\n")
		for prediction in predictions:
			l = np.array(sorted([i.strip() for i in open("Kanto.txt").readlines()]))
			idx = list((-prediction).argsort())[:10]
			print("\n The top 10 predictions for " + FileList[index])
			print(l[np.array(idx)])
			print("\n Thier Scores:")
			for li,pj in zip(l[np.array(idx)],prediction[np.array(idx)]):
				print(li + ": " + str(pj))
			# print(prediction[np.array(idx)][0])
			pokelist.append(l[np.array(idx)][0])
			imgt = Image.open(FileList[index])
			imgt.save("Output/Stage4_Pokemon_Label/" + l[np.array(idx)][0] + ".png")
			score[l[np.array(idx)][0]]= max(score[l[np.array(idx)][0]], prediction[np.array(idx)][0])

			index+=1

	pokelist = list(set(pokelist))

	# pokelist = []
	# score = sorted(score.items(), key = lambda k: k[1], reverse = True)
	# for i, j in score:
	# 	pokelist.append(i)

	# clean
	# for f in FileList:
	# 	os.remove(f)

	# full_pred = np.zeros(prediction.shape[0])
	# for prediction in predictions:
	# 	idx = list((-prediction).argsort())[:10]
	# 	if prediction[np.array(idx)][0] > 0.70:
	# 		full_pred += prediction
	# idx = list((-full_pred).argsort())[:10]
	# pokelist = l[np.array(idx)]

	return pokelist

def getAttackClass(pokelist):
	FileList = ["Sample_Images/"+sys.argv[2]]

	caffe.set_mode_gpu()
	classifier = caffe.Classifier("Models/Classify_Attacks/deploy.prototxt","Models/Classify_Attacks/bvlc_googlenet_pokenet_attacks_iter_132000.caffemodel" ,
	        image_dims=[224,224], raw_scale=255.0, channel_swap=[2,1,0])

	inputs = [caffe.io.load_image(im_f) for im_f in FileList]
	predictions = classifier.predict(inputs)

	attacklist = []
	l = np.array([i.lower().replace(" ","") for i in sorted(load_obj("Models/Classify_Attacks/kanto_attacks_list"))])
	attack_map = load_obj("Models/Classify_Attacks/pokemon_attack_map")
	attacks = []
	for poke in pokelist:
		attacks.extend(list(attack_map[poke]))
	attacks = np.array([i.lower().replace(" ","") for i in sorted(attacks)])
	idx = l.searchsorted(attacks)

	for prediction in predictions:
		ids = list((-prediction[idx]).argsort())[:10]

		print("\n The top 10 attacks:")
		print(attacks[np.array(ids)])
		print("\n Thier Scores:")
		for li,pj in zip(attacks[np.array(ids)],sorted(prediction[idx], reverse = True)):
			print(li + ": " + str(pj))
		#print(prediction[np.array(idx)][0])
		attacklist.append(str(attacks[np.array(ids)][0]))

	return attacklist

def getCaption(pokelist, attacklist):
	generator = load_obj("Caption_Structures/random_templates_generator")
	sentence = ""
	attack = attacklist[0]
	if len(pokelist) >= 2:
		# find attacker
		sentence = str(random.choice(list(generator["two"])))
		PA = random.choice(pokelist)
		pokelist.remove(PA)
		PB = random.choice(pokelist)

		attacker = ""
		target = ""
		attack_map = load_obj("Models/Classify_Attacks/pokemon_attack_map")
		if attack in attack_map[PA]:
			attacker = PA
			target = PB
		if attack in attack_map[PB]:
			attacker = PB
			target = PA
		print("The Attacker: "+ attacker)
		print("The Traget: "+ target)
		print("The Attack: "+ attack.capitalize())

		sentence = sentence.replace("$PA",attacker).replace("$PB",target).replace("$A",attacklist[0].capitalize())

	else:
		sentence = str(random.choice(list(generator["one"])))
		PA = random.choice(pokelist)
		print("The Attacker: "+ PA)
		print("The Attack: "+ attacklist[0])
		sentence = sentence.replace("$PA",PA).replace("$A",attacklist[0].capitalize())

	return sentence

if __name__ == '__main__':

	print("Segmenting Pokemon from the Battle Image...")
	# Pokemon_Segmenter.cropROI(sys.argv[2])
	os.system("python SelectiveSearch/Pokemon_Segmenter.py " + sys.argv[1] + " \"" + sys.argv[2]+"\"")

	print("Classifying the Pokemon...")
	time.sleep(1)
	pokelist = getPokemonClass()
	print("\nThe Pokemon are: ")
	print(pokelist)
	print("\n")

	print("Stage 4: Pokemon Classification/Identification Complete!")
	print("Results stored in Output/Stage4_Pokemon_Label/")
	print("Next: Attack Classification/Identification")
	a = raw_input("Do you want to proceed?")

	print("Identifying the Attack...")
	time.sleep(1)
	attacklist = getAttackClass(pokelist)

	print("\n")

	print("Stage 5: Attack Classification/Identification Complete!")
	print("The Results:")
	print("The Attack is: ")
	print(attacklist)
	print("The Pokemon were: ")
	print(pokelist)
	print("\n")
	print("Next: Captioning the battle image.")

	a = raw_input("Do you want to proceed?")

	print("Captioning the Battle Image...")
	sentence = getCaption(pokelist, attacklist)
	print("\nThe Caption is:")
	print(sentence)
	print("\n")
	print("Stage 6: Image Captioning Complete!")

	# Display text on image
	from PIL import Image
	from PIL import ImageFont
	from PIL import ImageDraw
	img = Image.open("Sample_Images/"+sys.argv[2])
	draw = ImageDraw.Draw(img)
	font = ImageFont.truetype("CaviarDreams.ttf", int(img.size[1]*img.size[0]*0.000040))
	draw.text((10, 10),sentence,(255,255,255),font=font)
	img.show()

	a = raw_input("Do you want to clean up?")
	clean()
	print("Previous results flushed.")
