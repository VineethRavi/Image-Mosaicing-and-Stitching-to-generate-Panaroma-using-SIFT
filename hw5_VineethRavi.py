import numpy as np
import cv2 as cv
import math
from scipy import optimize

"""
Importing the neccessary Libraries required for HW5
"""

# The function below is used for drawing lines, between points.
# The line is drawn between the two images, where corresponding points are matched.
def draw_lines(image1,sift_pts1,image2,sift_pts2):
    
    s1=np.shape(image1)
    s2=np.shape(image2)
    # Combining Both the images, to be placed adjacent to each other
    Siftimage=np.zeros((s1[0],s1[1]+s2[1],3)) 
    Siftimage[:,0:s1[1]]=image1
    Siftimage[:,s1[1]:s1[1]+s2[1]]=image2
    # Drawing Lines for SIFT points matched
    for i in range(0,len(sift_pts1)):
        cv.line(Siftimage,(int(sift_pts1[i][1]),int(sift_pts1[i][0])),(s1[1]+int(sift_pts2[i][1]),int(sift_pts2[i][0])),(0,0,255),thickness=2)

    return Siftimage


# Matching the corresponding interest points, between the two images
def match_sift(image1,Sift_Corners1,des1,image2,Sift_Corners2,des2):
    
    l1=len(Sift_Corners1)
    l2=len(Sift_Corners2)
    euclidian=np.zeros((l1,l2)) # Similarity metric for feature vectors
    
    for i in range(0,l1):
        for j in range(0,l2): # Normalizing Lengths to make it invariant to change in illumniation
            euclidian[i][j]=np.linalg.norm(des1[i]/np.linalg.norm(des1[i])-des2[j]/np.linalg.norm(des2[j]))
    # Computing the Euclidean distance between the 128 bit feature vectors        
    sift_threshold=np.min(euclidian)*(10)
    sift_pts1=[]
    sift_pts2=[]
    array_pts2=[]
    # Finding similar interest points and mapping them with each other
    for i in range(0,len(euclidian)):
        val=np.min(euclidian[i])    # Computing the minimum value in each row
        idx=np.argmin(euclidian[i])
        if((val<=sift_threshold)and(idx not in array_pts2)):
            array_pts2.append(idx)
            sift_pts1.append([Sift_Corners1[i][0],Sift_Corners1[i][1]])
            sift_pts2.append([Sift_Corners2[idx][0],Sift_Corners2[idx][1]])    
    # Adding similar and corresponding interest points to the list        
    
    return sift_pts1,sift_pts2


"""
Implementing the SIFT Algorithm from CV library
This is used to find the interest points in the images
"""

def SIFT(image1clr,image2clr):
    
    """
    Converting to Gray-Scale values to perfrom the SIFT operator, to find interest points
    """
    image1=cv.cvtColor(image1clr,cv.COLOR_RGB2GRAY)
    image2=cv.cvtColor(image2clr,cv.COLOR_RGB2GRAY)

    sift = cv.xfeatures2d.SIFT_create()
    
    # Finding the interest points and the descriptor vectors for every interest point in the image
    kp1,des1 = sift.detectAndCompute(image1,None)
    kp2,des2 = sift.detectAndCompute(image2,None)
    
    Sift_1=[]
    Sift_2=[]
    # Extracting the interest points for both images
    for i in kp1:
        Sift_1.append([i.pt[1],i.pt[0]])
        
    for i in kp2:
        Sift_2.append([i.pt[1],i.pt[0]])
        
    print('d1',len(des1))
    print('d2',len(des2))
        
    # Finding the similarity metric and mapping similar interest points
    sift_pts1,sift_pts2=match_sift(image1,Sift_1,des1,image2,Sift_2,des2)    
    # Drawing Lines between the corresponding interest points in the image
    Siftimage=draw_lines(image1clr,sift_pts1,image2clr,sift_pts2)  
    
    cv.imwrite('SIFTimage.jpg',Siftimage)
    # Writing the Output corresponding interest points in both images, with lines drawn
    
    return sift_pts1,sift_pts2

"""
Estimating Homography using Linear Least Squares Algorithm
"""

def HomographyMatrix(points1,points2):
    
    ta=[]
    H = np.zeros((3,3))
    
    for i in range(0,len(points1)):
        tmp=[points1[i][0],points1[i][1],1,0,0,0,-points2[i][0]*points1[i][0],-points2[i][0]*points1[i][1],-points2[i][0]]            
        ta.append(tmp)
        tmp=[0,0,0,points1[i][0],points1[i][1],1,-points2[i][1]*points1[i][0],-points2[i][1]*points1[i][1],-points2[i][1]]
        ta.append(tmp)
 
    A=np.asarray(ta)         # Computing the A matrix, which is used to find the Homography
    
    u,s,vh=np.linalg.svd(A)    # Linear Least Squares Estimate
    
    h=np.transpose(vh[-1])    # Initial guess or solution obtained by using the Linear Least Squares Algorithm

    H[0][0] = h[0]
    H[0][1] = h[1]
    H[0][2] = h[2]
    H[1][0] = h[3]
    H[1][1] = h[4]
    H[1][2] = h[5]
    H[2][0] = h[6]
    H[2][1] = h[7]
    H[2][2] = h[8]
    
    return A,h,H             # Returning the values

"""
Implementing the RANSAC Algorithm
"""

def RANSAC(image1clr,image2clr,sift_pts1,sift_pts2):

    sift_pts1=np.asarray(sift_pts1).astype('float')
    sift_pts2=np.asarray(sift_pts2).astype('float')
    
    p=0.99              # The probability that at least one of the N trials will be free of outliers in the calculation of H
    n=6                 # The number of correspondences chosen each trial to compute the Homography
    epsilon=0.70        # The Probability that a correspondence is an outlier
    delta=30            # The threshold for the inlier set
    n_total=len(sift_pts1)     # The total number of points available for choosing n correspondences
    
    M = int(np.ceil(n_total*(1.0-epsilon)))     # Minimum value for the size of the inlier set, for it to be acceptable
    N = int(np.ceil(np.log(1.0-p)/(np.log(1.0-((1.0-epsilon)**n)))))     # The total number of Trials
    
    print('N',N)
    print('M',M)

    inliers_pts=[]                     # Finding the corresponding inlier points in a given trial
    outliers_pts=[]                    # Finding the corresponding outlier points in a given trial
    count_inliers=[]                   # Count for the Number of inliers / Support in each trial
    
    for k in range(0,N):   
        c = np.random.choice(n_total, n)    # Randomly choose n correspondences from the interest points vectors
        A,h,H=HomographyMatrix(sift_pts1[c],sift_pts2[c])    # Compute the Homography matrix for these correspondences
        
        inliers_pts.append([])
        outliers_pts.append([])
        count=0
        for i in range(0,n_total):
            new_pt2=np.matmul(H,[sift_pts1[i][0],sift_pts1[i][1],1.0])   # Using Homography to compute x' mapping in image
            new_pt2=new_pt2/new_pt2[2]
            if(np.linalg.norm(sift_pts2[i]-new_pt2[0:2])<=delta):        # Verifying if the error is minimal in computing x'
                inliers_pts[k].append([sift_pts1[i],sift_pts2[i] ])      # Appending list of inliers correspondences
                count=count+1                                            # Increasing Count
            else:
                outliers_pts[k].append([sift_pts1[i],sift_pts2[i] ])     # Appending list of inliers correspondences
        
        count_inliers.append(count)                                 # The Support for each pair of inliers set
        
        
    val=np.argmax(count_inliers)            # Finding the Maximum number of inliers set support
        
    print('val',val)
                              # Verifying if the number of inliers is > M, for the inliers set to be accepted
    if(val>=M):
        print("The size of the inlier set is acceptable")
    else:
        print("The size of the inlier set is NOT acceptable")
        
    max_inliers_pts = inliers_pts[val]       # Best inliers correspondences set with highest support
    min_outliers_pts = outliers_pts[val]
    
    max_inliers_pts=np.asarray(max_inliers_pts)
    min_outliers_pts=np.asarray(min_outliers_pts)
 
    final_inlier_points1 = max_inliers_pts[:,0]          # Inliers in image 1
    final_inlier_points2 = max_inliers_pts[:,1]          # Inliers in image 2
    
    final_outlier_points1 = min_outliers_pts[:,0]        # Outliers in image 1
    final_outlier_points2 = min_outliers_pts[:,1]        # Outliers in image 2
         
    inliers_image=draw_lines(image1clr,final_inlier_points1,image2clr,final_inlier_points2)  
    cv.imwrite('inliers_image.jpg',inliers_image)
    
    outliers_image=draw_lines(image1clr,final_outlier_points1,image2clr,final_outlier_points2)  
    cv.imwrite('outliers_image.jpg',outliers_image)
    
    """
    Linear Least Squares for obtaining Homography
    """
                                                         # Computing Initial guess of Homography to be used in LM Algorithm
    A,h,H=HomographyMatrix(final_inlier_points1,final_inlier_points2)    # This is used as input in Non-Linear least squares
                                                     
    return A,h,H

"""
Implementing the Levenberg-Marquardt Algorithm (using non-linear least squares)
"""
def fun(x):
    
    val=np.matmul(B,x)                  # Function for which we are finding solution using the LM Algorithm
    
    if(np.linalg.norm(x)< 1e-50 ):  # To prevent, trivial solution (all zeros), from being approached by Gradient Descent
        return np.ones(len(B))*5000   # Steer solution to non trivial minima, where Ah=0
    
    return val

def LM(A,h):
    
    H=np.zeros((3,3))
    global B
    B=A
    
    sol=optimize.root(fun,h,method='lm')       #  Using scipy.optimize for implementing the LM algorithm
    
    H[0][0] = sol.x[0]
    H[0][1] = sol.x[1]
    H[0][2] = sol.x[2]
    H[1][0] = sol.x[3]
    H[1][1] = sol.x[4]
    H[1][2] = sol.x[5]
    H[2][0] = sol.x[6]
    H[2][1] = sol.x[7]
    H[2][2] = sol.x[8]                    # Refining the initial homography guess to OBTAIN THE FINAL Homography
    
    return H          # The refined Homography


def WeightedAverageRGBPixelValue(pt, img):     # Interpolation of Pixel value using pixel values from adjacent neighbours
    
    y1=int(math.floor(pt[0]))             # Used in Weighted Average Pixel Computation
    y2=int(math.ceil(pt[0]))
    x1=int(math.floor(pt[1]))
    x2=int(math.ceil(pt[1]))
        
    Wp=1/np.linalg.norm(np.array([pt[0]-x1,pt[1]-y1]))  # Weights for adjacent pixels
    Wq=1/np.linalg.norm(np.array([pt[0]-x1,pt[1]-y2]))
    Wr=1/np.linalg.norm(np.array([pt[0]-x2,pt[1]-y1]))
    Ws=1/np.linalg.norm(np.array([pt[0]-x2,pt[1]-y2]))
                                                         # Computing the average pixel value using euclidian metric
    pixel_value = (Wp*img[y1][x1] + Wq*img[y2][x1] + Wr*img[y1][x2] + Ws*img[y2][x2])/(Wp+Wq+Wr+Ws)
    
    return pixel_value       # Return Pixel Value

"""
Computing Corners which are used in computing the Resultant stitched Mosaic / Panaroma
"""

def ComputeCorners(image,H):
    
    a=np.matmul(H,[0.0,0.0,1.0])
    b=np.matmul(H,[np.shape(image)[0],0.0,1.0])
    c=np.matmul(H,[np.shape(image)[0],np.shape(image)[1],1.0])
    d=np.matmul(H,[0.0,np.shape(image)[1],1.0])
    
    a=a/a[2]            # The 4 corners in the bounding box, in clockwise order is given here
    b=b/b[2]
    c=c/c[2]
    d=d/d[2]

    return a,b,c,d

"""
Stitching the Images together, to form the Panaroma
"""

def StitchMosaic(image1,image2,image3,image4,image5,H13,H23,H43,H53):
    
    a1,b1,c1,d1=ComputeCorners(image1,H13)           # Computing the Corners
    a2,b2,c2,d2=ComputeCorners(image2,H23)
    a3,b3,c3,d3=ComputeCorners(image3,np.eye(3))
    a4,b4,c4,d4=ComputeCorners(image4,H43)
    a5,b5,c5,d5=ComputeCorners(image5,H53)
    
    # Similar to HW3 , for computing the Projections between two images with corresponding points
    
    array=[a1[0],b1[0],c1[0],d1[0],a2[0],b2[0],c2[0],d2[0],a3[0],b3[0],c3[0],d3[0],a4[0],b4[0],c4[0],d4[0],a5[0],b5[0],c5[0],d5[0]]
    
    xmin=np.min(array)
    xmax=np.max(array)
    
    array=[a1[1],b1[1],c1[1],d1[1],a2[1],b2[1],c2[1],d2[1],a3[1],b3[1],c3[1],d3[1],a4[1],b4[1],c4[1],d4[1],a5[1],b5[1],c5[1],d5[1]]
    
    ymin=np.min(array)
    ymax=np.max(array)
        
    xlen=int(round(xmax-xmin))       # Final Dimensions of the image of the output panaroma
    ylen=int(round(ymax-ymin))
    
    final=np.zeros((xlen,ylen,3))
    print(xlen,ylen)
    
    H13=np.linalg.pinv(H13)          # Computing inverses, similar reasoning as in HW3
    H23=np.linalg.pinv(H23)
    H43=np.linalg.pinv(H43)
    H53=np.linalg.pinv(H53)
    
    """
    Stitching the Images together, by projecting images, 2,4,1,5 on image 3. This is done using Weighted Average Pixel Calculation or Bilinear interpolation
    Similar to the procedure and reasoning given in HW3
    """ 
    for i in range(0,np.shape(final)[0]):
        for j in range(0,np.shape(final)[1]):
            
            if((i+xmin)>=0 and (i+xmin)<np.shape(image3)[0] and (j+ymin)>=0 and (j+ymin)<np.shape(image3)[1]):
                final[i,j]=image3[int(i+xmin),int(j+ymin)]
            else:
                tmp=np.array([i+xmin,j+ymin,1.0])
                xp=np.array(np.dot(H23,tmp))       # Fitting the Image to World
                xp=xp/xp[2]
                if((xp[0]>0)and(xp[0]<np.shape(image2)[0]-1)and(xp[1]>0)and(xp[1]<np.shape(image2)[1]-1)):
                    final[i,j]=WeightedAverageRGBPixelValue(xp,image2)
                    
                else:
                    tmp=np.array([i+xmin,j+ymin,1.0])
                    xp=np.array(np.dot(H43,tmp))       # Fitting the Image to World
                    xp=xp/xp[2]
                    if((xp[0]>0)and(xp[0]<np.shape(image4)[0]-1)and(xp[1]>0)and(xp[1]<np.shape(image4)[1]-1)):
                        final[i,j]=WeightedAverageRGBPixelValue(xp,image4)
                    
                    else:
                        tmp=np.array([i+xmin,j+ymin,1.0])
                        xp=np.array(np.dot(H13,tmp))       # Fitting the Image to World
                        xp=xp/xp[2]
                        if((xp[0]>0)and(xp[0]<np.shape(image1)[0]-1)and(xp[1]>0)and(xp[1]<np.shape(image1)[1]-1)):
                            final[i,j]=WeightedAverageRGBPixelValue(xp,image1)
                            
                        else:
                            tmp=np.array([i+xmin,j+ymin,1.0])
                            xp=np.array(np.dot(H53,tmp))       # Fitting the Image to World
                            xp=xp/xp[2]
                            if((xp[0]>0)and(xp[0]<np.shape(image5)[0]-1)and(xp[1]>0)and(xp[1]<np.shape(image5)[1]-1)):
                                final[i,j]=WeightedAverageRGBPixelValue(xp,image5)           
    
    return final      # Final Output Panaroma

"""
MAIN FUNCTION BEGINS HERE
"""

# Reading the images for performing image mosaicing
image1clr=cv.imread("1.jpg") 
image2clr=cv.imread("2.jpg")
image3clr=cv.imread("3.jpg")
image4clr=cv.imread("4.jpg")
image5clr=cv.imread("5.jpg")

"""
Resizing the images for faster computation
"""

resize=0.3

image1clr=cv.resize(image1clr, (0,0), fx=resize, fy=resize)
image2clr=cv.resize(image2clr, (0,0), fx=resize, fy=resize)
image3clr=cv.resize(image3clr, (0,0), fx=resize, fy=resize)
image4clr=cv.resize(image4clr, (0,0), fx=resize, fy=resize)
image5clr=cv.resize(image5clr, (0,0), fx=resize, fy=resize)


"""
Finding Interest Points for two consecutive pairs of images using SIFT
We use RANSAC, to eliminate outliers and find the inital solution guess using Linear Least squares to get the Homography Matrix
We refine the guess using Levenberg-Marquardt algorithm (Non-linear least squares) to obtain the final Homography
"""

sift_pts1,sift_pts2=SIFT(image1clr,image2clr)
A,h,H12=RANSAC(image1clr,image2clr,sift_pts1,sift_pts2)
H12=LM(A,h)                                              # Homography from 1 to 2 

sift_pts1,sift_pts2=SIFT(image2clr,image3clr)
A,h,H23=RANSAC(image2clr,image3clr,sift_pts1,sift_pts2)
H23=LM(A,h)                                              # Homography from 2 to 3

H13=np.matmul(H23,H12)                                   # Homography from 1 to 3                       

sift_pts1,sift_pts2=SIFT(image4clr,image3clr)
A,h,H43=RANSAC(image4clr,image3clr,sift_pts1,sift_pts2)
H43=LM(A,h)                                              # Homography from 4 to 3 

sift_pts1,sift_pts2=SIFT(image5clr,image4clr)
A,h,H54=RANSAC(image5clr,image4clr,sift_pts1,sift_pts2)
H54=LM(A,h)                                              # Homography from 5 to 4 

H53=np.matmul(H43,H54)                                   # Homography from 5 to 3 

"""
We now stitch the images together using the Final Refined Homographies obtained to get the Panaroma
"""
panaroma=StitchMosaic(image1clr,image2clr,image3clr,image4clr,image5clr,H13,H23,H43,H53)

cv.imwrite('Panaroma.jpg',panaroma)

