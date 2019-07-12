####################################################################################################
## A program producing pencil drawing from natural images (https://github.com/LarsPh/Pencil-Sketch).
## Copyright (c) 2019 Zhaorong Wang.
##  
## This program is free software: you can redistribute it and/or modify  
## it under the terms of the GNU General Public License as published by  
## the Free Software Foundation, version 3.
## 
## This program is distributed in the hope that it will be useful, but 
## WITHOUT ANY WARRANTY; without even the implied warranty of 
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License 
## along with this program. If not, see <http://www.gnu.org/licenses/>.
####################################################################################################


import numpy as np
import cv2
from scipy.ndimage.interpolation import rotate
from scipy.interpolate import UnivariateSpline
from scipy.signal import fftconvolve
from scipy.sparse import spdiags
from scipy.sparse.linalg import cg
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

#np.set_printoptions(threshold=sys.maxsize)
#np.set_printoptions(**opt)

#normalize the pixel valute to the range 0-255
def to255Int(A, c=255):
    return np.uint8((A - np.amin(A))/(np.amax(A)-np.amin(A))*c)

#edge detection
def gradEdges(S, method):
    #gradient edge detection, works poorly even after denoising
    #S = cv2.GaussianBlur(S, (7,7), 0)
    #S = cv2.bilateralFilter(S, 30, 30, 3)
    #Dx = np.column_stack((S[ : ,1: ], S[ : , -1]))-S
    #Dy = np.row_stack((S[1: , : ], S[-1, : ]))-S
    #G = np.sqrt(Dx**2+Dy**2)
    if(method == 1):
        #Laplacian edge detection, works better after bilateral filter
        #S = cv2.GaussianBlur(S, (7,7), 0)
        S = cv2.bilateralFilter(S, 30, 30, 3)
        G = cv2.Laplacian(S, cv2.CV_8U, ksize=5)
    else:
        #Canny edge detection, good performance on denoising, but more and thinner strokes
        #works better with ligher strokes, i.e. smaller value for darkerStroke
        (r, c) = np.shape(S)
        mean = np.sum(S)/(r*c)
        lower = mean*0.75
        upper = mean*1.25
        G = cv2.Canny(S, lower, upper)
    return G

#Draw strokes. Generate cross-like effect at junctions of strokes
def drawLines(G, nDirec, kernR, darkerStroke):
    (r, c) = np.shape(G)
    kernScale = int(min(r, c)/kernR)
    #generate line segments 
    L = np.zeros((kernScale*2+1, kernScale*2+1, nDirec))
    L[ : , kernScale+1, 0] = 1
    for i in range(nDirec):
        L[ : , : ,i] = np.round(rotate(L[ : , : ,0], i/nDirec*180, reshape=False))
    #classification, fftconvolve save mem compared with convolve when N is large
    Gc = np.zeros((r, c, nDirec))
    for i in range(nDirec):
        Gc[ : , : ,i] = fftconvolve(G, L[ : , : ,i], mode="same")
    Index = np.argmax(Gc, axis=2)
    C = np.zeros((r, c, nDirec))
    for i in range(nDirec):
        C[ : , : ,i] = G*(Index == i)

    Sp = np.zeros((r, c))
    for i in range(nDirec):
        Sp += fftconvolve(C[ : , : ,i], L[ : , : ,i], mode="same")

    Sp = 1-(Sp - np.amin(Sp))/(np.amax(Sp)-np.amin(Sp))
    Sp = Sp ** darkerStroke
    return Sp

#Tone transformation
def transferTone(A, w1, w2, w3):
    A = to255Int(A)
    (r, c) = np.shape(A)
    p = np.zeros(256)
    inten = np.arange(0, 256)
    p1 = (1/9)*np.exp(-(255-inten)/9)
    p2 = np.zeros(256)
    p2[105:225] = 1/(225-105)
    p3 = (1/np.sqrt(2*np.pi*11))*np.exp(-(inten-90)**2/(11*11*2))
    p = w1*p1+w2*p2+w3*p3
    p /= np.sum(p)
    #plt.plot(inten, p, 'ro')
    #smoothen the curve
    fp = UnivariateSpline(inten, p)
    fp.set_smoothing_factor(0.000035)
    p = fp(inten)
    plt.plot(inten, p, 'go')
    p /= np.sum(p)
    cp = np.zeros(256)
    #calculate cunmulative density function
    for i in range(256):
        cp[i] = np.sum(p[0:i])
    cp /= cp[255]
    s = np.zeros(256)
    for i in range(r):
        for j in range(c):
            s[A[i, j]] += 1
    s /= r*c
    cs = np.zeros(256)
    for i in range(256):
        cs[i] = np.sum(s[0:i])
    #plt.plot(inten, s, 'o')
    
    Diff = np.zeros((256, 256))
    for i in range(256):
        for j in range(256):
            Diff[i, j] = np.abs(cs[i]-cp[j])
    #tonal mapping
    matching = np.zeros(256)
    for i in range(256):
        minIdx = 0
        minVal = Diff[i, 0]
        for j in range(256):
            if (Diff[i, j] < minVal):
                minVal = Diff[i, j]
                minIdx = j
        matching[i] = minIdx
    J = np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            J[i, j] = matching[A[i, j]]

    t = np.zeros(256)
    for i in range(r):
        for j in range(c):
            t[np.uint8(J[i, j])] += 1
    t /= r*c
    plt.plot(inten, t, 'bo')
    #uncommnet to see the comparison on target(p) and mapped(t) histogram
    #plt.show()
    return to255Int(J)

#Transfer the texture so that it matches the tone image by sovling linear equation
def renderTexture(J, H1, lam):
    (r, c) = np.shape(J)
    (n, m) = np.shape(H1)
    #adjust the resolution of texture
    x = int(np.ceil(r/n))
    y = int(np.ceil(c/m))
    H2 = H1
    for i in range(y):
        H2 = np.hstack((H2, H1))
    H = H2
    for i in range(x):
        H = np.vstack((H, H2))
    H = H[ :r, :c]
    Image.fromarray(to255Int(H), "L").save("temp/texture_test1.jpg")

    epsilon = 0.0001
    H = np.float64(H)/255
    #prevent log(0)
    J = np.float64(J+epsilon)/255
    #transform to vector
    h = np.reshape(H, r*c)
    j = np.reshape(J, r*c)
    logh = np.log(h)
    logj = np.log(j)
    i = np.ones(r*c)
    #use methods for sparse matrix to prevent mem problems
    Dx = spdiags(np.array([-i,i]), np.array([0,1]), r*c, r*c)
    Dy = spdiags(np.array([-i,i]), np.array([0,c]), r*c, r*c)
    A = spdiags(logh*logh, 0, r*c, r*c)+lam*(Dx.transpose().dot(Dx)+Dy.transpose().dot(Dy))
    b = logh*logj
    #conjugate gradient
    beta, _ = cg(A, b)
    Beta = beta.reshape(r, c)
    T = H**Beta
    #elements of T have values between 0 to 1 
    return T

#Combine the pencil strokes and tonal texture
def combineST(S, T):
    return to255Int(S) * T

#edgeDetctMthd: choose method for edge detection. 1 for bilateral filter + Laplacian, 2 for Canny
#kerndirN: value for line segments (kernal) with different directions
#kernScale: the scale for kernal would be min(w, h)/kerScale. w and h are the height and width of input image
#darkerStroke: larger value would make the stroke darker
#w1, w2, w3: weight of three distribution
#lam: lambda multiplied by the gradient of beta in formula (8) in final step (check the paper for details) 
def pencilSketch(srcDir="source_images", txPath="textures/texture1.jpg", edgeDetctMthd=1, kerdireN=8, kernScale=200, darkerStroke=2, w1=0.52, w2=0.37, w3=0.11, lam=0.2):
    print("Generating images...")

    for fname in os.listdir(srcDir):
        im = Image.open(srcDir+"/"+fname)
        ImRGB = np.array(im)
        ImLUV = cv2.cvtColor(ImRGB, cv2.COLOR_RGB2Luv)
        #transform on L channal
        ImGs = ImLUV[ : , : ,0]
        if not os.path.exists("temp"):
            os.mkdir("temp")
        if not os.path.exists("output"):
            os.mkdir("output")
        Image.fromarray(ImGs, "L").save("temp/grayscale_"+fname) 
        #print(np.shape(ImGs))

        G = gradEdges(ImGs, 1)
        Image.fromarray(to255Int(G), "L").save("temp/gradient_"+fname)
        Sp = drawLines(G, kerdireN, kernScale, darkerStroke)
        Image.fromarray(to255Int(Sp), "L").save("temp/lines_"+fname)
        J = transferTone(ImGs, w1, w2, w3)
        Image.fromarray(to255Int(J), "L").save("temp/tone_map_"+fname)
        texIm = Image.open(txPath).convert("L")
        H = np.array(texIm)
        T = renderTexture(J, H, lam)
        Image.fromarray(to255Int(T), "L").save("temp/pencil_texture_"+fname)
        R = combineST(Sp, T)
        Image.fromarray(to255Int(R), "L").save("output/pencil_sketch_"+fname)
        ImLUV[ : , : ,0] = to255Int(R)
        ImRGB = cv2.cvtColor(ImLUV, cv2.COLOR_Luv2RGB)
        Image.fromarray(ImRGB, "RGB").save("output/colored_pencil_sketch_"+fname)

    print("Finished generation")

pencilSketch()