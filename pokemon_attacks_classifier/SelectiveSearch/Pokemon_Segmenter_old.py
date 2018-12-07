# -*- coding: utf-8 -*-
### Insert Current Path
import os, sys, inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
cmd_folder+="/"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
###

import skimage.data
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import numpy as np

# 512x512 (resolution to be set if input image is too low or too high)
# dont force a rectangular image to 512x512 sqaure
# works best without resizing
# if resoltuion is too low like 100x100, then maybe resize (but the performance will decrease)

candidates = set()
merged_candidates = set()
refined = set()
final = set()
width = 0
height = 0

def contains():
    x1, y1, w1, h1 = p
    for x, y, w, h in candidates:
        if x1>=x and y1 >= y and x1+w1 <= x+w and y1+h1 <= y+h:
            return True
        if x1<=x and y1 <= y and x1+w1 >= x+w and y1+h1 >= y+h:
            candidates.remove((x, y, w, h))
            return False
    return False

def draw_superbox():
    # (x1,y1) top-left coord, (x2,y2) bottom-right coord, (w,h) size
    remp = set(refined)
    ref = list(refined)
    while len(ref) > 0:
        x1, y1, w1, h1 = ref[0]

        if len(ref) == 1: # final box
            final.add((x1, y1, w1, h1))
        else:
            ref.remove((x1, y1, w1, h1))
            remp.remove((x1, y1, w1, h1))

        over = set()
        nonover = set()
        for x2, y2, w2, h2 in remp:
            A = {'x1': x1, 'y1': y1, 'x2': x1+w1, 'y2': y1+h1, 'w': w1, 'h': h1}
            B = {'x1': x2, 'y1': y2, 'x2': x2+w2, 'y2': y2+h2, 'w': w2, 'h': h2}

            # overlap between A and B
            SA = A['w']*A['h']
            SB = B['w']*B['h']
            SI = np.max([ 0, np.min([A['x2'],B['x2']]) - np.max([A['x1'],B['x1']]) ]) * np.max([ 0, np.min([A['y2'],B['y2']]) - np.max([A['y1'],B['y1']]) ])
            SU = SA + SB - SI
            overlap_AB = float(SI) / float(SU)

            if overlap_AB >= 0.20:
                over.add((B['x1'],B['y1'],B['w'],B['h']))
        if len(over) != 0: #Overlap
            remp = remp - over
            for i in over: ref.remove(i)
            over.add((A['x1'],A['y1'],A['w'],A['h']))
            final.add((min([i[0] for i in over]), min([i[1] for i in over]), max([i[0]+i[2] for i in over]) - min([i[0] for i in over]), max([i[1]+i[3] for i in over]) - min([i[1] for i in over])))
        else:   #No Overlap
            final.add((A['x1'],A['y1'],A['w'],A['h']))

# def contains_remove():

#     for x, y, w, h in merged_candidates:
#         f = False
#         temp = set(merged_candidates)
#         temp.remove((x, y, w, h))
#         for x1, y1, w1, h1 in temp:
#             if x1>=x and y1 >= y and x1+w1 <= x+w and y1+h1 <= y+h:
#                 f = False
#                 break
#             else:
#                 f = True
#         if f == True:
#             refined.add((x, y, w, h))

def contains_remove():
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
    thresh = int(((width+height)/2)*(0.14)) #0.01

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



def cropROI(filename):
    global width, height
    image = skimage.io.imread(filename)

    width = len(image[0])
    height = len(image)
    # area = width * height

    # if (area < 256*256 or area > 512*512) and ((height > 512 and width > 512) or (height < 256 and width <256)):
    #     height = int(512 * height/width)
    #     width = 512

    num = 1
    for sc in [350,450,500]:
        for sig in [0.8]:
            for mins in [30,60,120]:
                img = skimage.io.imread(filename)[:,:,:3]
                # if height == len(img) and width == len(img[0]):
                #     pass
                # else:
                #     img = skimage.transform.resize(img, (height, width))
                # perform selective search
                img_lbl, regions = selectivesearch.selective_search(img, scale=sc, sigma= sig, min_size = mins)

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
                    if w >= (img.shape[0]-1)*(0.9) and h >= (img.shape[1]-1)*(0.9):
                        continue
                    candidates.add(r['rect'])
        print("Stage " + str(num) + " Done.")
        num+=1

    merge()
    draw_superbox()

    img = skimage.io.imread(filename)
    # if height == len(img) and width == len(img[0]):
    #     pass
    # else:
    #     img = skimage.transform.resize(img, (height, width))
    i=1
    for x, y, w, h in refined:
        if str(sys.version)[:3] == "3.4":
            skimage.io.imsave(cmd_folder+"Temp/" + str(i)+".jpg", img[y:y+h, x:x+w]) # crop
        else:
            skimage.io.imsave(cmd_folder+"Temp/" + str(i)+".jpg", img[y:y+h, x:x+w]) # crop
        i+=1
if __name__ == '__main__':
    cropROI(sys.argv[1])
