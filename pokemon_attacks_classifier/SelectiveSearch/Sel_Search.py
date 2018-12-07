# -*- coding: utf-8 -*-
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import numpy as np
import skimage.transform
import os
import sys
import pickle

candidates = set()
merged_candidates = set()
refined = set()
final = set()
final_extended = set()
final_extended_blow = set()

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol = 2)

def overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    SA = w1*h1
    SB = w2*h2
    SI = np.max([ 0, np.min([x1+w1,x2+w2]) - np.max([x1,x2]) ]) * np.max([ 0, np.min([y1+h1,y2+h2]) - np.max([y1,y2]) ])
    SU = SA + SB - SI
    overlap_AB = float(SI) / float(SU)
    return (overlap_AB > 0.0)

def blow_superbox():
    global final_extended_blow
    thresh = width*0.10
    tempc = set()
    feb = set()
    for x, y, w, h in final_extended:
        if (x, y, w, h) in tempc: continue
        temp = set()
        temp.add((x, y, w, h))
        f = final_extended.copy()
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
    final_extended_blow = feb - tempc

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

def draw_superbox(finals=[], param = 0.40):
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
    if len(merged_candidates) >= 5:
    	merged_candidates = list(merged_candidates)
    	merged_candidates = sorted(merged_candidates, key = lambda x : x[2]*x[3])
    	merged_candidates = set(merged_candidates[:int(len(merged_candidates)*0.60)])
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

ijk = 1
for name in os.listdir("Raw_Data"):
    candidates = set()
    merged_candidates = set()
    refined = set()
    final = set()
    final_extended = set()
    final_extended_blow = set()

    print(name)
    print((float(ijk)/len(os.listdir("Raw_Data")))*100)
    img = skimage.io.imread("Raw_Data/"+name)
    width = len(img[0])
    height = len(img)

    if width*height < 256*256:
        new_size  = 256
        width  = int(new_size * width / height)
        height = new_size
        print("Me1")
    elif width*height > 512*512:
        new_size  = 512
        width  = int(new_size * width / height)
        height = new_size
        print("Me2")
    elif width*height < 512*512  and width*height > 512*512*0.50:
        new_size  = 512
        width  = int(new_size * width / height)
        height = new_size
        print("Me3")
    elif width*height > 256*256  and width*height < 512*512*0.50:
        new_size  = 256
        width  = int(new_size * width / height)
        height = new_size
        print("Me4")

    for sc in [350,450,500]:
        for sig in [0.8]:
            for mins in [30,60,120]: # important
                img = skimage.io.imread("Raw_Data/"+name)[:,:,:3]

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
                print(str(sc)+"_"+str(sig)+"_"+str(mins))

    # img = skimage.io.imread("Raw_Data/"+name)[:,:,:3]
    # if height == len(img) and width == len(img[0]):
    #     pass
    # else:
    #     img = skimage.transform.resize(img, (height, width))

    # candidates = set([(383, 246, 103, 167), (61, 72, 596, 381), (61, 297, 174, 105), (485, 103, 163, 175), (543, 243, 138, 147), (374, 351, 132, 102), (61, 243, 215, 159), (518, 262, 93, 126), (401, 263, 67, 114), (385, 351, 72, 84), (480, 99, 178, 215), (483, 62, 160, 196), (262, 72, 395, 377), (143, 150, 187, 178), (480, 62, 178, 252), (61, 150, 314, 252), (215, 243, 61, 119), (96, 150, 280, 178), (61, 199, 215, 203), (61, 150, 269, 252), (385, 351, 73, 84), (0, 72, 681, 439), (143, 150, 233, 178), (143, 53, 538, 396), (518, 261, 82, 83), (61, 150, 341, 252), (134, 347, 129, 102), (61, 72, 596, 377), (354, 42, 146, 91), (483, 62, 198, 196), (0, 202, 220, 174), (61, 297, 194, 105), (0, 53, 681, 458), (258, 53, 423, 396)])


    repeat = 0
    while True:
        if repeat == 0:
            merged_candidates = set()
            refined = set()
            final = set()
            final_extended = set()
            final_extended_blow = set()

            print(candidates)
            merge()
            print(refined)
            draw_superbox()
            print(final)
            extend_superbox()
            print(final_extended)
            blow_superbox()
            print(final_extended_blow)

        elif repeat == 1:
            merged_candidates = set()
            refined = set()
            final = set()
            final_extended = set()
            final_extended_blow = set()

            print(candidates)
            merge()
            print(refined)
            draw_superbox(param = 0.90)
            print(final)
            extend_superbox(param = 0.06)
            print(final_extended)
            blow_superbox()
            print(final_extended_blow)

        img = skimage.io.imread("Raw_Data/"+name)[:,:,:3]
        if height == len(img) and width == len(img[0]):
            pass
        else:
            img = skimage.transform.resize(img, (height, width))
        # draw rectangles on the original image
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(img)
        for x, y, w, h in final:
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)

        img1 = skimage.io.imread("Raw_Data/"+name)[:,:,:3]
        if height == len(img1) and width == len(img1[0]):
            pass
        else:
            img1 = skimage.transform.resize(img1, (height, width))
        ij = 1
        for x, y, w, h in final:
            skimage.io.imsave("regions/"+name.rsplit(".",1)[0]+"_"+str(ij)+"."+name.rsplit(".",1)[1], img1[y:y+h,x:x+w])
            ij+=1
        plt.savefig("final_Regions/"+name)
        plt.close('all')
        save_obj(final, "final_Regions_pkl/final_"+name.rsplit(".",1)[0])
        save_obj(candidates, "final_Regions_pkl/candidates_"+name.rsplit(".",1)[0])

        if len(final) == 0:
            skimage.io.imsave("regions/"+name.rsplit(".",1)[0]+"."+name.rsplit(".",1)[1], img1)
            final.add((0,0,img1.shape[1]-1,img1.shape[0]-1))
            if len(candidates) == 0:
                candidates.add((0,0,img1.shape[1]-1,img1.shape[0]-1))
            save_obj(final, "final_Regions_pkl/final_"+name.rsplit(".",1)[0])
            save_obj(candidates, "final_Regions_pkl/candidates_"+name.rsplit(".",1)[0])
            break

        test = list(final)[0]
        w = test[2]
        h = test[3]
        if len(final) <= 2 and w*h >= width*height*0.90:
            if repeat == 1:
                break
            repeat = 1
            continue
        else:
            repeat = 0
            break

    ijk+=1
