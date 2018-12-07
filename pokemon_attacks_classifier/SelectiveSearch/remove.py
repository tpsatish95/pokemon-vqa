import numpy as np
from scipy.ndimage import gaussian_filter

from skimage import img_as_float
from skimage.morphology import reconstruction
import skimage.io
import os
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, threshold_adaptive, threshold_li, threshold_yen, threshold_isodata, gaussian_filter
import skimage.io
import skimage.color
import skimage.morphology

candidates = set()
refined = set()
final = set()
final_extended = set()
ref = set()

def extend_rect(l):
    return (min([i[0] for i in l]), min([i[1] for i in l]), max([i[0]+i[2] for i in l]) - min([i[0] for i in l]), max([i[1]+i[3] for i in l]) - min([i[1] for i in l]))

def extend_superbox():
    global width, height
    thresh = ((width+height)/2)*(0.22)

    tempc = set()
    for x, y, w, h in final:
        if (x, y, w, h) in tempc: continue
        temp = set()
        temp.add((x, y, w, h))
        for x1, y1, w1, h1 in final:
            # if abs(x1-x) <= thresh and abs(w1-w) <= thresh:
            if x1 >= x and (w1+x1) <= w+x:
                temp.add((x1, y1, w1, h1))
                tempc.add((x1, y1, w1, h1))
        final_extended.add(extend_rect(temp))
    contains_remove(final_extended)

def draw_superbox(finals=[]):
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
            # print(overlap_AB)
            #

            if overlap_A >= 0.15 or overlap_B >= 0.15:
                over.add((B['x1'],B['y1'],B['w'],B['h']))
        # print(len(over))
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
        draw_superbox(final)
        return

def contains_remove(param = []):
	if param != []:
		can = set(param)
	else:
		can = set(candidates)
	for x, y, w, h in can:
		temp = set(can)
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
		if all(test) and param == []:
		    refined.add((x, y, w, h))
		if all(test) and param != []:
			ref.add((x, y, w, h))

# Convert to float: Important for subtraction later which won't work with uint8
ij = 10
for i in os.listdir("."):
	if ".py" not in i:
		# image = img_as_float(skimage.io.imread(i)[:,:,:3])
		# image = gaussian_filter(image, 1)

		# seed = np.copy(image)
		# seed[1:-1, 1:-1] = image.min()
		# mask = image

		# dilated = reconstruction(seed, mask, method='dilation')
		# image = image - dilated
		image = skimage.io.imread(i)
		image = skimage.color.rgb2gray(image)
		thresh = threshold_otsu(image)
		binary = image <= thresh

		# skimage.io.imsave("binary_"+i.split(".")[0]+".png", binary)
		# bin = skimage.io.imread("binary_"+i.split(".")[0]+".png")
		bin = binary
		# im = gaussian_filter(bin, sigma=0.1)
		# blobs = im > im.mean()
		blobs = bin
		# skimage.io.imsave("h"+i, blobs)

		# blobs_labels = measure.label(blobs, background=0)
		labels = skimage.morphology.label(blobs, neighbors = 4)


		candidates = set()
		blobs = ndimage.find_objects(labels)
		image1 = skimage.io.imread(i)
		width = len(image1[0])
		height = len(image1)
		for c1, c2 in blobs:
			if (c2.stop - c2.start) * c1.stop - c1.start > (image1.shape[0]*image1.shape[1])*(0.026):
				if (c2.stop - c2.start) * c1.stop - c1.start < (image1.shape[0]*image1.shape[1])*(0.90):
					candidates.add((c2.start, c1.start,c2.stop - c2.start, c1.stop - c1.start))

		print(candidates)
		contains_remove()
		print(refined)
		draw_superbox()
		print(final)
		extend_superbox()
		print(final_extended)
		print(ref)

		image1 = skimage.io.imread(i)
		# draw rectangles on the original image
		fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
		ax.imshow(image1)

		for x, y, w, h in ref:
			skimage.io.imsave(str(ij)+".jpg", image1[x:x+w,y:y+h])
			rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
			ax.add_patch(rect)
			ij+=1
		plt.savefig("patch_"+i.split(".")[0]+".png")
		plt.close('all')
