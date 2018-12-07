# -*- coding: utf-8 -*-
### Insert Current Path
import os, sys, inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
cmd_folder+="/"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
###

from PIL import Image
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage.transform
import pickle
import os
import numpy as np
import sys
import caffe
import selectivesearch


def getClass(FileList):
    # Make classifier.
    caffe.set_mode_gpu()
    classifier = caffe.Classifier(cmd_folder+"PokeNoPoke/deploy.prototxt",cmd_folder+"PokeNoPoke/bvlc_googlenet_pokeNoPokeNet_iter_19000.caffemodel" ,
            image_dims=[224,224], raw_scale=255.0, channel_swap=[2,1,0])

    # Load numpy array (.npy), directory glob (*.jpg), or image file.

    inputs = [caffe.io.load_image(im_f) for im_f in FileList]

    print("Classifying %d inputs." % len(inputs))

    # Classify.
    predictions = classifier.predict(inputs)

    # print(predictions)

    return predictions

candidates = set()
merged_candidates = set()
refined = set()
final = set()
poke = set()
partpoke= set()
other = set()
final_extended = set()
final_extended_blow = set()

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

def overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    SA = w1*h1
    SB = w2*h2
    SI = np.max([ 0, np.min([x1+w1,x2+w2]) - np.max([x1,x2]) ]) * np.max([ 0, np.min([y1+h1,y2+h2]) - np.max([y1,y2]) ])
    SU = SA + SB - SI
    overlap_AB = float(SI) / float(SU)
    return (overlap_AB > 0.0)

def merge_poke_part():
    global poke, partpoke, final

    tempc = set()
    feb = set()
    for x, y, w, h in poke:
        if (x, y, w, h) in tempc: continue
        temp = set()
        temp.add((x, y, w, h))
        for x1, y1, w1, h1 in partpoke:
            if w1*h1 < w*h*0.90 and overlap(x,y,w,h,x1,y1,w1,h1):
                temp.add((x1, y1, w1, h1))
                tempc.add((x1, y1, w1, h1))
        feb.add(extend_rect(temp))

    tempc = set()
    for x, y, w, h in feb:
        if (x, y, w, h) in tempc: continue
        f = feb.copy()
        f.remove((x, y, w, h))
        for x1, y1, w1, h1 in f:
            if x1 <= x and x1+w1 >= x+w and y1 <= y and y1+h1 >= y+h and (x,y,w,h) in poke:
                tempc.add((x1, y1, w1, h1))
    final = feb - tempc

def blow_superbox():
    global final_extended
    thresh = width*0.10
    tempc = set()
    feb = set()
    for x, y, w, h in poke.union(partpoke):
        if (x, y, w, h) in tempc: continue
        temp = set()
        temp.add((x, y, w, h))
        f = poke.union(partpoke).copy()
        f.remove((x, y, w, h))
        for x1, y1, w1, h1 in f:
            if x1-x> -thresh and (x1+w1)-(x+w) < thresh:
                temp.add((x1, y1, w1, h1))
                tempc.add((x1, y1, w1, h1))
            elif w1*h1 < w*h*0.20 and overlap(x,y,w,h,x1,y1,w1,h1):
                temp.add((x1, y1, w1, h1))
                tempc.add((x1, y1, w1, h1))
        feb.add(extend_rect(temp))

    tempc = set()
    for x, y, w, h in feb:
        if (x, y, w, h) in tempc: continue
        f = feb.copy()
        f.remove((x, y, w, h))
        for x1, y1, w1, h1 in f:
            if x1 >= x and x1+w1 <= x+w and y1 >= y and y1+h1 <= y+h:
                tempc.add((x1, y1, w1, h1))
    final_extended = feb - tempc

def extend_superbox(param = 0.16):
    # global width, height
    # thresh = ((width+height)/2)*(param)

    # tempc = set()
    # for x, y, w, h in final:
    #     if (x, y, w, h) in tempc: continue
    #     temp = set()
    #     temp.add((x, y, w, h))
    #     f = final.copy()
    #     f.remove((x, y, w, h))
    #     for x1, y1, w1, h1 in f:
    #         if abs(y1-y) <= thresh and abs(h1-h) <= thresh:
    #             temp.add((x1, y1, w1, h1))
    #             tempc.add((x1, y1, w1, h1))
    #     final_extended.add(extend_rect(temp))
    global final_extended
    final_extended = final.copy()

def extend_rect(l):
    return (min([i[0] for i in l]), min([i[1] for i in l]), max([i[0]+i[2] for i in l]) - min([i[0] for i in l]), max([i[1]+i[3] for i in l]) - min([i[1] for i in l]))

def draw_superbox(finals=[], param = 0.01):
    noover = []
    refinedT = []

    global final
    final = set()

    # (x1,y1) top-left coord, (x2,y2) bottom-right coord, (w,h) size
    if finals != []:
        refinedT = finals
    else:
        refinedT = refined
    remp = set(refinedT)
    ref = list(refinedT)

    while len(ref) > 0:
        x1, y1, w1, h1 = ref[0]

        if len(ref) == 1: # final box
            final.add((x1, y1, w1, h1))
            ref.remove((x1, y1, w1, h1))
            remp.remove((x1, y1, w1, h1))
        else:
            ref.remove((x1, y1, w1, h1))
            remp.remove((x1, y1, w1, h1))

        over = set()
        for x2, y2, w2, h2 in remp:
            A = {'x1': x1, 'y1': y1, 'x2': x1+w1, 'y2': y1+h1, 'w': w1, 'h': h1}
            B = {'x1': x2, 'y1': y2, 'x2': x2+w2, 'y2': y2+h2, 'w': w2, 'h': h2}

            # overlap between A and B
            SA = A['w']*A['h']
            SB = B['w']*B['h']
            SI = np.max([ 0, np.min([A['x2'],B['x2']]) - np.max([A['x1'],B['x1']]) ]) * np.max([ 0, np.min([A['y2'],B['y2']]) - np.max([A['y1'],B['y1']]) ])
            SU = SA + SB - SI
            overlap_AB = float(SI) / float(SU)
            overlap_A = float(SI) / float(SA)
            overlap_B = float(SI) / float(SB)

            if overlap_A >= param or overlap_B >= param:
                over.add((B['x1'],B['y1'],B['w'],B['h']))

        if len(over) != 0: #Overlap
            remp = remp - over
            for i in over: ref.remove(i)
            over.add((A['x1'],A['y1'],A['w'],A['h']))
            # print(over)
            final.add((min([i[0] for i in over]), min([i[1] for i in over]), max([i[0]+i[2] for i in over]) - min([i[0] for i in over]), max([i[1]+i[3] for i in over]) - min([i[1] for i in over])))
            # final.add((np.mean([i[0] for i in over]), np.mean([i[1] for i in over]), np.mean([i[2] for i in over]), np.mean([i[3] for i in over])))
            noover.append(False)
        else:   #No overlap
            final.add((x1,y1,w1,h1))
            noover.append(True)

    if all(noover):
        return
    else:
        draw_superbox(final, param = param)
        return

def contains_remove():
    global merged_candidates
    temp_merged_candidates = set()
    # if len(merged_candidates) >= 5:
    # 	merged_candidates = list(merged_candidates)
    # 	merged_candidates = sorted(merged_candidates, key = lambda x : x[2]*x[3])
    # 	merged_candidates = set(merged_candidates[:int(len(merged_candidates)*0.60)])
    for x, y, w, h in merged_candidates:
        if w <= width*0.85 and h <= height*0.85:
            temp_merged_candidates.add((x,y,w,h))
    merged_candidates = temp_merged_candidates.copy()

    for x, y, w, h in merged_candidates:
        temp = set(merged_candidates)
        temp.remove((x, y, w, h))
        test = []
        for x1, y1, w1, h1 in temp:
            A = {'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h, 'w': w, 'h': h}
            B = {'x1': x1, 'y1': y1, 'x2': x1+w1, 'y2': y1+h1, 'w': w1, 'h': h1}
            # overlap between A and B
            SA = A['w']*A['h']
            SB = B['w']*B['h']
            SI = np.max([ 0, np.min([A['x2'],B['x2']]) - np.max([A['x1'],B['x1']]) ]) * np.max([ 0, np.min([A['y2'],B['y2']]) - np.max([A['y1'],B['y1']]) ])
            SU = SA + SB - SI
            overlap_AB = float(SI) / float(SU)
            if overlap_AB > 0.0:
                # if x1>=x and y1 >= y and x1+w1 <= x+w and y1+h1 <= y+h:
                if x1<=x and y1 <= y and x1+w1 >= x+w and y1+h1 >= y+h:
                    test.append(False)
                else:
                    test.append(True)
            else:
                test.append(True)
        if all(test):
            refined.add((x, y, w, h))

def mean_rect(l):
    return (min([i[0] for i in l]), min([i[1] for i in l]), max([i[0]+i[2] for i in l]) - min([i[0] for i in l]), max([i[1]+i[3] for i in l]) - min([i[1] for i in l]))

def merge():
    global width, height
    thresh = int(((width+height)/2)*(0.14))
    tempc = set()
    for x, y, w, h in candidates:
        if (x, y, w, h) in tempc: continue
        temp = set()
        temp.add((x, y, w, h))
        for x1, y1, w1, h1 in candidates:
            if abs(x1-x) <= thresh and abs(y1-y) <= thresh and abs(w1-w) <= thresh and abs(h1-h) <= thresh:
                temp.add((x1, y1, w1, h1))
                tempc.add((x1, y1, w1, h1))
        merged_candidates.add(mean_rect(temp))
    contains_remove()

if __name__ == '__main__':
    candidates = set()
    merged_candidates = set()
    refined = set()
    final = set()
    final_extended = set()
    final_extended_blow = set()
    poke = set()
    partpoke= set()
    other = set()

    name = sys.argv[2]

    img = skimage.io.imread("Sample_Images/"+name)[:,:,:3]
    width = len(img[0])
    height = len(img)

    if width*height < 256*256:
    	new_size  = 256
    	width  = int(new_size * width / height)
    	height = new_size
    	# print("Me1")
    elif width*height > 512*512:
    	new_size  = 512
    	width  = int(new_size * width / height)
    	height = new_size
    	# print("Me2")
    elif width*height < 512*512  and width*height > 512*512*0.50:
    	new_size  = 512
    	width  = int(new_size * width / height)
    	height = new_size
    	# print("Me3")
    elif width*height > 256*256  and width*height < 512*512*0.50:
    	new_size  = 256
    	width  = int(new_size * width / height)
    	height = new_size
    	# print("Me4")

    if sys.argv[1] == "0":
        num = 1
        for sc in [350,450,500]:
            for sig in [0.8]:
                for mins in [30,60,120]: # important
                    img = skimage.io.imread("Sample_Images/"+name)[:,:,:3]

                    if height == len(img) and width == len(img[0]):
                        pass
                    else:
                        img = skimage.transform.resize(img, (height, width))
                    # perform selective search
                    img_lbl, regions = selectivesearch.selective_search(
                        img, scale=sc, sigma= sig,min_size = mins)

                    for r in regions:
                        # excluding same rectangle (with different segments)
                        if r['rect'] in candidates:
                            continue
                        # excluding regions smaller than 2000 pixels
                        if r['size'] < 2000:
                            continue
                        # distorted rects
                        x, y, w, h = r['rect']
                        if w / h > 1.2 or h / w > 1.2:
                            continue
                        if w >= (img.shape[1]-1)*(0.7) and h >= (img.shape[0]-1)*(0.7):
                            continue
                        candidates.add(r['rect'])
            print("Scan " + str(num) + " Done.")
            num+=1
    elif sys.argv[1] == "1":
        candidates = load_obj("regions_pkl/candidates_"+name.replace(".jpg",".pkl").replace(".png",".pkl"))
        if height == len(img) and width == len(img[0]):
            pass
        else:
            img = skimage.transform.resize(img, (height, width))

    # print(candidates)
    merge()
    # print(refined)
    # draw_superbox()
    # print(final)
    # extend_superbox()
    # print(final_extended)
    # blow_superbox()
    # print(final_extended_blow)

    draw = merged_candidates
    fList = []
    box_list = []

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in draw:
        rect = mpatches.Rectangle(
        (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.savefig("Output/Stage1_SelectiveSearch/"+name)
    plt.close('all')

    ij = 1
    for x, y, w, h in draw:
        skimage.io.imsave("Output/Stage1_SelectiveSearch/Clips/"+name.rsplit(".",1)[0]+"_"+str(ij)+"."+name.rsplit(".",1)[1], img[y:y+h,x:x+w])
        fList.append("Output/Stage1_SelectiveSearch/Clips/"+name.rsplit(".",1)[0]+"_"+str(ij)+"."+name.rsplit(".",1)[1])
        box_list.append((x, y, w, h))
        ij+=1

    print("Stage 1: Selective Search Complete!")
    print("Results stored in Output/Stage1_SelectiveSearch/")
    print("Next: Filtering Regions Proposals with Pokemon-NoPokemon CNN Classfier.")
    a = raw_input("Do you want to proceed?")

    try:
        a = getClass(fList)
        l = np.array([0,1,2])
        i = 0
        for pred in a:
            idx = list((-pred).argsort())
            pred = l[np.array(idx)]
            if pred[0] == 0 and pred[1] == 2:
                other.add(box_list[i])
                imgt = Image.open(fList[i])
                imgt.save("Output/Stage2_PokeNoPoke/Other/" + fList[i].replace("Output/Stage1_SelectiveSearch/Clips/","").rsplit(".",1)[0] +"_"+str(i)+ "." + fList[i].replace("Output/Stage1_SelectiveSearch/Clips/","").rsplit(".",1)[1])
            elif pred[0] == 1:
                poke.add(box_list[i])
                imgt = Image.open(fList[i])
                imgt.save("Output/Stage2_PokeNoPoke/Pokemon/" + fList[i].replace("Output/Stage1_SelectiveSearch/Clips/","").rsplit(".",1)[0] +"_"+str(i)+ "." + fList[i].replace("Output/Stage1_SelectiveSearch/Clips/","").rsplit(".",1)[1])
            elif pred[0] == 2 and pred[1] == 0:
                partpoke.add(box_list[i])
                imgt = Image.open(fList[i])
                imgt.save("Output/Stage2_PokeNoPoke/Partly_Pokemon/" + fList[i].replace("Output/Stage1_SelectiveSearch/Clips/","").rsplit(".",1)[0] +"_"+str(i)+ "." + fList[i].replace("Output/Stage1_SelectiveSearch/Clips/","").rsplit(".",1)[1])
            i+=1
    except:
        print("No Pokemon Regions")

    print("Stage 2: Filtering Regions Proposals with Pokemon-NoPokemon CNN Classfier Complete!")
    print("Results stored in Output/Stage2_PokeNoPoke/")
    print("Next: Merging the Pokemon regions with Partly Pokemon regions and Final Filtering.")
    a = raw_input("Do you want to proceed?")

    blow_superbox()
    print(final_extended)

    # draw rectangles on the original image
    img = skimage.io.imread("Sample_Images/"+name)[:,:,:3]
    if height == len(img) and width == len(img[0]):
        pass
    else:
        img = skimage.transform.resize(img, (height, width))

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in final_extended:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.savefig("Output/Stage3_Merged_Filtered/final_"+name)
    plt.close('all')

    fList = []
    box_list = []
    ij = 1
    for x, y, w, h in final_extended:
        skimage.io.imsave("Output/Stage3_Merged_Filtered/Clips/"+name.rsplit(".",1)[0]+"_"+str(ij)+"."+name.rsplit(".",1)[1], img[y:y+h,x:x+w])
        fList.append("Output/Stage3_Merged_Filtered/Clips/"+name.rsplit(".",1)[0]+"_"+str(ij)+"."+name.rsplit(".",1)[1])
        box_list.append((x, y, w, h))
        ij+=1

    try:
        i = 0
        path = "Output/Stage3_Merged_Filtered/Flitered_Final/"
        if len(fList) >= 2:
            pred_list = []
            a = getClass(fList)
            l = np.array([0,1,2])
            for pred in a:
                idx = list((-pred).argsort())
                pred = l[np.array(idx)]
                if pred[0] == 0:
                    pred_list.append(0)
                elif pred[0] == 1:
                    pred_list.append(1)
                elif pred[0] == 2:
                    pred_list.append(2)
                print(pred)
                i+=1
            if 1 in pred_list:
                for index,j in enumerate(pred_list):
                    if j == 1:
                        img = Image.open(fList[index])
                        img.save(path + fList[index].replace("Output/Stage3_Merged_Filtered/Clips/","").rsplit(".",1)[0] +"_"+str(index)+ "." + fList[index].replace("Output/Stage3_Merged_Filtered/Clips/","").rsplit(".",1)[1])
            else:
                for index,j in enumerate(pred_list):
                    img = Image.open(fList[index])
                    img.save(path + fList[index].replace("Output/Stage3_Merged_Filtered/Clips/","").rsplit(".",1)[0] +"_"+str(index)+ "." + fList[index].replace("Output/Stage3_Merged_Filtered/Clips/","").rsplit(".",1)[1])

        elif len(fList) == 1:
            img = Image.open(fList[i])
            img.save(path + fList[i].replace("Output/Stage3_Merged_Filtered/Clips/","").rsplit(".",1)[0] +"_"+str(i)+ "." + fList[i].replace("Output/Stage3_Merged_Filtered/Clips/","").rsplit(".",1)[1])
        else:
            img = Image.open("Sample_Images/"+name)
            img.save(path + name.rsplit(".",1)[0] +"_"+str(i)+ "." + name.rsplit(".",1)[1])
    except:
        print("No Pokemon Regions")


    print("Stage 3: Merging the Pokemon regions with Partly Pokemon regions and Final Filtering, Complete!")
    print("Results stored in Output/Stage3_Merged_Filtered/")
    print("Next: Pokemon Classification/Identification")
    a = raw_input("Do you want to proceed?")
