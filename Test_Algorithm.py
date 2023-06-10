import numpy as np
from numpy import inf
from TAD_Laplace import Laplace
import argparse


def main():

     parser = argparse.ArgumentParser()
     parser.add_argument("heatmap", type=str, help="heatmap file")
     parser.add_argument("bed", type=str, help="bed file")
     parser.add_argument("sig0", nargs='?', default= 0.8,help="fiedler value threshold for sufficient algebraic connectedness,stop splitting")
     parser.add_argument("ms0", nargs='?', default= 3,help="minimimum bin size of TAD region")
     parser.add_argument("MOD", nargs='?', default= 1,help="toeplitz matrix form (1=y,!1=n)")
     parser.add_argument("MERGE", nargs='?', default= 0,help= "merge TAD boundaries (0=n,!0=y)" )
     parser.add_argument("NN", nargs='?', default= 0,help="scale fiedler number by matrix/submatrix size (0=n,!0=y)")
     args = parser.parse_args()

     H = np.genfromtxt(args.heatmap,delimiter=',')
     B = np.genfromtxt(args.bed,delimiter='\t')

     idx = np.where(np.sum(H,axis=0) != 0)[0]
     idx2 = np.where(np.sum(H,axis=0) == 0)[0]
     H2 = np.copy(H)
     B2 = np.copy(B)
     for i in range(0,len(idx2)):
          for j in range(0,len(idx2)):
               H2[int(idx2[i])][int(idx2[j])]=-10000
     collect = np.where(H2==-10000)
     H3 = np.delete(H2,collect[0],0) # delete row
     H3 = np.delete(H3,collect[0],1) # delete col
     B2 = np.delete(B2,collect[0],0) # delete row
     B3 = np.zeros((len(B2),len(B2[0])))
     for i in range(0,len(B3)):
          B3[i,0:len(B2[0])-1] = B2[i,1:]  
          B3[i,len(B2[0])-1] = i   

     H3 = np.dot(np.dot((1/np.sum(H3,0))*(np.eye(len(H3))),np.dot(H3,(1/np.sum(H3,0)*np.eye(len(H3))))),np.power(np.mean(np.sum(H3,0)),2))
     Ht = np.ceil(H3)
     Ht=np.log(Ht)
     Ht[Ht==-inf]=-1
     Ht[Ht>6]=6 #upperthreshold
     Ht = Ht + 1.001

     data = Laplace(Ht,args.sig0,args.ms0,args.MOD,args.MERGE,args.NN,captureFile=0)    
     data.TadBoundaries()
     data.save("TADindex.csv")
 

     TAD_boundaries = np.zeros((len(data.TAD_boundaries)-1,2))     
     for i in range(1,len(data.TAD_boundaries)):
          TAD_boundaries[i-1][0]= B3[int(data.TAD_boundaries[i-1])][0]
          TAD_boundaries[i-1][1]= B3[int(data.TAD_boundaries[i])][0]

 #    np.savetxt("B3.csv", B3, delimiter=',')
     np.savetxt("TAD_boundaries.csv",TAD_boundaries,delimiter=',',fmt='%d')

if __name__ == "__main__":
     main()

