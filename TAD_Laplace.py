from scipy import linalg as SLA
import numpy as np
import copy
import math
import argparse

class Laplace():

     def __init__(self,filename,sig0,ms0,MOD,MERG,NN,captureFile=1):
          self.sig0 = float(sig0)
          self.ms0 = int(ms0)
          self.MOD = int(MOD)
          self.MERG = int(MERG)
          self.NN = int(NN)
          if bool(captureFile):
               self.Ho = np.genfromtxt(filename,delimiter=',')
          else:
               self.Ho = filename #user input is already array

     def TadBoundaries(self):

          # Remove unmappable region if they are included
          idx0 = np.sum(self.Ho,axis=0)==0;
          if np.any(idx0 == True):
                vals = np.where(idx0 == True)[0]
                #print("vals = ",vals)
                for i in range(0,len(vals)):
                    self.Ho = np.delete(self.Ho,vals[i],0) #remove col
                    self.Ho = np.delete(self.Ho,vals[i],1) #remove row
                    vals = vals - 1 # removing column and row causes successive val position to shift by one

          # Remove the diagonal
          dH = np.diagonal(self.Ho)
          MM = np.zeros((len(dH), len(dH)), float)
          np.fill_diagonal(MM, 1)
          dH = MM*dH
          H = self.Ho - dH
          print("H = ",H)
          L = len(self.Ho)

          # Splitting on Toeplitz normalized matrix
          if int(self.MOD) == 1:
                # First split is performed on Toeplitz normalized matrix
                Hn=self.ToepNorm(self.Ho)  # Toeplitz normalization
                print("Hn = ",Hn)
                evec,eval = self.Fdvectors(Hn)

                # Positions of starting domain
                first = np.sign(evec)
                first = np.delete(first,0)

                second = np.sign(evec)
                second = np.delete(second,-1)

                pos = np.nonzero(first - second)[0] + 1 #find boundaries (jump between 1 and -1)
                pos = np.insert(pos,0,0)
                pos = np.append(pos,len(self.Ho)-1)
          elif int(self.MOD) == 2:
                # Splitting directly on the data matrix
                pos = np.array([0,L-1])

          # Recursive splitting 
          spa = np.zeros(L)
          spa[pos] = 1   

          #splitting on subblocks and runnning Fiedler Vector on each

          for i in range(0,len(pos)-1):
               # Block range 
               idx = range(pos[i],pos[i+1])

               # If block size <= ms0, we will not split again
               if (len(idx)>self.ms0):
                    #Sub-matrix
                    SH = H[idx[0]:idx[-1]+1,idx[0]:idx[-1]+1]

                    #Fiedler number and vector
                    evec,eval = self.Fdvectors(SH)

                    #If the Fiedler number of the block is small
                    if eval <= self.sig0:
                         sp1 = self.SubSplit(SH)
                          # Mark boundaries
                         bindex = np.nonzero(sp1)[0]
                         spa[pos[i]+bindex]=1

          posn = np.where(spa>0)[0]

          if (self.MERG != 0):
               posn=self.MergeSmall(posn,H)

          self.TAD_boundaries = posn
    
          return(self.TAD_boundaries)

     def SubSplit(self,SH):
          # Recursively splitting a connection matrix via Fiedler value and vector
          # returns boundaries after positional split

          # Fiedler vector
          evec,eval = self.Fdvectors(SH)

          first = np.sign(evec)
          first = np.delete(first,0)

          second = np.sign(evec)
          second = np.delete(second,-1)

          pos = np.nonzero(first - second)[0] + 1 #find boundaries (jump between 1 and -1)
          pos = np.insert(pos,0,0)
          pos = np.append(pos,len(SH)-1)

          sp = np.zeros(len(SH))
          sp[pos] = 1

          # If Fiedler value is high enough
          if  (eval > self.sig0+1E-5):  #   +1e-5 for numerical stability
               sp = 1
               return(sp) 

          # For each sub-block
          for i in range(0, len(pos)-1):
               idx = range(pos[i],pos[i+1])

               #minimum sub-block size
               if (len(idx)>self.ms0):
                    # Continue to split
                    sp1 = self.SubSplit(SH[idx[0]:idx[-1]+1,idx[0]:idx[-1]+1]) #needed tp add 1 because idx[-1] is excluded
                    # Mark bock boundary
                    bindex = np.nonzero(sp1)[0]
                    sp[bindex + pos[i]]=1

          return(sp)

     # Fiedler vector calculation
     def Fdvectors(self,H):
          
          H = (H + np.transpose(H))/2
          N = len(H)
          dgr = np.sum(H,axis=0)
          dgr[dgr==0] = 1
          dgr = np.absolute(dgr)  #decided to treat negatives as pos, previously zeroed out after sqrt

          DN = np.zeros((len(dgr), len(dgr)), float)
          DN2 = np.zeros((len(dgr), len(dgr)), float)
          np.fill_diagonal(DN, 1)
          np.fill_diagonal(DN2, 1)
          DN = DN*(1/np.sqrt(dgr))
          
          DN2 = (DN2*dgr) - H

          L1 = np.dot(DN2,DN)
          L = np.dot(DN,L1)

          if (self.NN == 1):
               L = DN2
          
          L = (L + np.transpose(L))/2
          seek = np.isnan(L)
          if np.any(seek == True):
                L[seek == True] = 0
          seek = np.isneginf(L)
          if np.any(seek == True):  #adjust negative infinite values 
                L[seek == True] = -1000000
          seek = np.isposinf(L)
          if np.any(seek == True):  #adjust positive infinite values
                L[seek == True] = 1000000

          print("L = ",L)
          
          vals,vecs = SLA.eigh(L,eigvals = (0,1))  # output only eigenvectors up to Fiedler

          Fdv = vecs[:,-1]
          Fdvl = vals[-1]


          if (self.NN == 1):
               Fdvl = math.pow((Fdvl/(len(L))),0.3)

          return(Fdv,Fdvl)

     #Merge small regions
     def MergeSmall(self,posn,H):
          
          Pos = posn
          Posr= copy.deepcopy(Pos)  
          
          # Find region only with 1 bin size
          first = np.delete(Pos,0)
          second = np.delete(Pos,-1)
          combine = first - second
          idx1 = np.where(combine == 1)[0] 

          for i in range(0,len(idx1)):
               cond1 = idx1[i]+1 <= len(Pos)-1 
               cond2 = idx1[i]-1 >= 0

               if idx1[i]+1 <= len(Pos)-1:
                    vrp = np.mean(H[Pos[idx1[i]],Pos[idx1[i]]+1:Pos[idx1[i]+1]+1]) #TAD to the right (need extra one - original code in matlab where last Pos[idx1[i]+1] is inclusive (python is exclusive)
               if idx1[i]-1 >= 0:
                    vrm = np.mean(H[Pos[idx1[i]],Pos[idx1[i]-1]:Pos[idx1[i]]-1+1]) #TAD to the left (add one because original matlab code end value is inclusive (Python exclusive)
               if ((cond1 and cond2)  and (vrm >= vrp)):
                    Posr[idx1[i]] = -100
               elif  ((cond1 and cond2)  and (vrm < vrp)):
                    Posr[idx1[i]+1] = -100
               if idx1[i] == 0:
                    Posr[1] = -100  # remove second index if boundary every one at start
               if idx1[i] == Pos[-1]-1:
                    Posr[-2]=-100  # remove second to last index if boundary every one at end

          Pos = np.delete(Pos,np.where(Posr==-100)[0])
          return(Pos)

     #Toeplitz normalization
     def ToepNorm(self,X):
          X = np.array(X,dtype='f')
          L = len(X)
          
          # Diagonal summation
          ds =self.sumDiag(X)
          #np.savetxt("ds_sumdiag.csv", ds, delimiter=',') 

          #Diagonal meaan value
          ds1 = ds[L-1:-1]  #does not copy end value
          ds2 = np.append(ds1, ds[-1])   # attach end

          B = range(L)[::-1]
          B = [x+1 for x in B]

          mds = np.zeros(L)
          for i in range(0,L):
               mds[i] = float(ds2[i])/float(B[i]) #mean diagonal value

          #np.savetxt("mds.csv", mds, delimiter=',')

          #Normalization matrix
          Tp = SLA.toeplitz(mds)


          NX = np.true_divide(X,Tp)

          seek = np.isnan(NX)
          if np.any(seek == True): #adjust nans
                NX[seek == True] = 0
          seek = np.isneginf(NX)
          if np.any(seek == True):  #adjust negative infinite values 
                NX[seek == True] = 0
          seek = np.isposinf(NX)
          if np.any(seek == True):  #adjust positive infinite values
                NX[seek == True] = 0

          #np.savetxt("toepH.csv", NX, delimiter=',')
          return(NX)         
          

     def sumDiag(self,X,antiD=0):
   
          N = len(X) #rows
          M = len(X[0]) #columns

          endflip = 0
          
          #Make sure matrix is not wide (reduces computaional burden for very wide matrices)
          if (M > N):
                X = np.transpose(X) 
                N = len(X)
                M = len(X[0])    
                endflip = 1

          #Xmod = np.zeros((N+M-1,M))

          if antiD == 0: #diagonals
               #Start diagonal numbering from upper right corner to lower left corner
               diags = [X.diagonal(i) for i in range(X.shape[1]-1,-X.shape[0],-1)]
          else:  #antidiagonals
               #start diagonal numbering from upper left corner to lower right corner
               X = np.transpose(X)
               X = np.flipud(X)
               X = np.transpose(X)
               diags = [X.diagonal(i) for i in range(X.shape[1]-1,-X.shape[0],-1)]
              
          #sum along each diagonal
          sdiags = np.zeros(N+M-1)
          for i in range(0,len(diags)):
               sdiags[i] = sum(diags[i])

          if endflip == 1:  #flip back to correct order if rows and columns were originally swapped
               sdiags = np.flipud(sdiags)

          return(sdiags)

     def save(self,fileName):
          np.savetxt(fileName, self.TAD_boundaries, delimiter=',',fmt='%d')

def main():
     parser = argparse.ArgumentParser()
     parser.add_argument("heatmap", type=str, help="heatmap file")
     parser.add_argument("sig0", nargs='?', default= 0.8,help="fiedler value threshold for sufficient algebraic connectedness,stop splitting")
     parser.add_argument("ms0", nargs='?', default= 3,help="minimimum bin size of TAD region")
     parser.add_argument("MOD", nargs='?', default= 1,help="toeplitz matrix form (1=y,!1=n)")
     parser.add_argument("MERGE", nargs='?', default= 0,help= "merge TAD boundaries (0=n,!0=y)" )
     parser.add_argument("NN", nargs='?', default= 0,help="scale fiedler number by matrix/submatrix size (0=n,!0=y)")
     args = parser.parse_args()

     data = Laplace(args.heatmap,args.sig0,args.ms0,args.MOD,args.MERGE,args.NN)
     data.TadBoundaries()
     data.save("TAD_boundaries.csv")
     
                   
if __name__ == "__main__":
     main()

     

