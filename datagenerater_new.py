import numpy as np
import scipy.io as sio
from random import shuffle
from numpy.matlib import repmat
import math




def readmatfile(filename):
    try:
        outdata = sio.loadmat(filename)[sio.whosmat(filename)[0][0]]
    except Exception:
        print('read erro ' + filename)
        outdata = []
    return outdata


def getlabdata(str_lab,N):
    
    dic = {'clean': [1, 0, 0, 0,0,0], 'white': [0, 1, 0, 0,0,0], 'babble': [0, 0, 1, 0,0,0 ], 'airplane': [0, 0, 0, 1,0,0],
           'cantine': [0, 0, 0, 0,1,0], 'market': [0, 0, 0, 0,0,1]}
    tt=repmat(dic[str_lab],N,1)
    return tt


def getlab_spk(str_lab,N):

    spk_id=int(str_lab[1:4])-51
    lab=np.zeros(50)
    lab.put(spk_id,1)
    lab=lab.tolist()
    tt=repmat(lab,N,1)
    return tt





def enframe(x,inc,w_len, isframe=1):
    """
    :param x:
    :param inc:
    :param w_len:
    :param isframe:
    :return:
    """

    nx, dx = x.shape
    if nx<5:
        return np.zeros([1,1])


    if isframe:

        zz=[]
        
        NN=math.ceil((w_len-1)/2)
        # print(NN)
        
        temp_before=repmat(x[0,:],int(NN),1)

        temp_end = repmat(x[-1, :], int(NN), 1)

        zz.append(temp_before)

        zz.append(x)

        zz.append(temp_end)

        x= np.concatenate(zz)

    nx,dx=x.shape

    length = w_len

    nf = int(math.ceil((nx - length + inc) // inc))

    # f = np.zeros((nf, length))

    indf = inc * np.arange(nf)

    inds = np.arange(length) + 1

    f = x[(np.transpose(np.vstack([indf] * length)) +

           np.vstack([inds] * nf)) - 1]

    a,b,c=f.shape

    f=f.reshape(a,b*c)

    return f

    
def readmat_lab(filename,str_lab_noise, str_lab_spk):
    """
    :param filename: mat file
    :param str_lab_noise: noise type
    :param str_lab_spk: spk label
    :return:
    """

    try:
        outdata = sio.loadmat(filename)[sio.whosmat(filename)[0][0]]
           
    except Exception:
        print(filename)
        outdata = []
        outlab_spk  = []
        outlab_noise= []
        return outdata,outlab_spk,outlab_noise
    
    data_framed = enframe(outdata, 1, 11, isframe=1)
    outlab_noise = getlabdata(str_lab_noise, np.shape(data_framed)[0])
    outlab_spk  =getlab_spk(str_lab_spk,np.shape(data_framed)[0])
    
    return data_framed, outlab_noise, outlab_spk    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

def loaddata(scp_file,savepath):
    d = np.genfromtxt(scp_file, dtype=str)

    N = np.shape(d)[0]

    print(N)

    filelist = d[:, 1]

    lablist = d[:, 2]
    lab_spklist=d[:,0]

    # files = [None]*N
    # labs = [None]*N
    files = []
    labs = []
    labs_spk=[]

    for i in range(N):

        if i % 100 == 0:
            print([i, N])

        try:
            
            file_data = readmatfile(savepath+'/'+filelist[i])
            # print(filelist[i])
            # print(np.shape(file_data))
        except Exception:
            print(filelist[i])
            # del files[i]
            # del labs[i]
            continue

        data_framed = enframe(file_data, 1, 11, isframe=1)
        if np.shape(file_data)[0] < 3:
            print(filelist[i])
            # del files[i]
            # del labs[i]
            continue

        file_lab = getlabdata(lablist[i], np.shape(data_framed)[0])
        spk_lab=getlab_spk(lab_spklist[i],np.shape(data_framed)[0])

        files.append(data_framed)
        labs.append(file_lab)
        labs_spk.append(spk_lab)

    N_file = np.shape(files)[0]

    if N_file < N:
        print('error files')
        print(N - N_file)





    return files, labs, labs_spk, N_file


def loadfilename(scp_file):
    d = np.genfromtxt(scp_file, dtype=str)

    N = np.shape(d)[0]

    print(N)

    filelist = d[:, 1]
    return filelist, N

class DataSet():



    def __init__(self,scp_file,savepath,batch_size, is_train=1):

        self.batch_size=batch_size
        self.scp_file=scp_file
        self.is_train=is_train
        self.savepath=savepath
        if is_train:
            print('--------------start load data--------------')
            self.datas, self.labs, self.labs_spk, self.size_epoch=loaddata(scp_file,savepath)
            print('--------------end load data--------------')
        else:
            self.files,self.size_epoch=loadfilename(scp_file)

        self._epoch_complate=0
        self._index_in_epoch=0
        self._n_batch=0

    def get_epoch_complate(self):
        return  self._epoch_complate

    def get_batch_num(self):
        return self._n_batch

    def next_batch(self):

        if self.is_train==0:
            print('erro')

            return [], []
        start=self._index_in_epoch
        self._index_in_epoch=self._index_in_epoch + self.batch_size

        if self._index_in_epoch> self.size_epoch:
            self._epoch_complate +=1
            perm=np.arange(self.size_epoch)
            shuffle(perm)

            print(perm)

            self.datas=[self.datas[i] for i in perm]
            self.labs=[self.labs[i] for i in perm]
            self.labs_spk=[self.labs_spk[i] for i in perm]
            start=0
            self._index_in_epoch=self.batch_size
            self._n_batch=0

        end=self._index_in_epoch
        self._n_batch += 1
        batch_datas=self.datas[start:end]
        batch_labs=self.labs[start:end]
        batch_labs_spk=self.labs_spk[start:end]

        return np.concatenate(batch_datas), np.concatenate(batch_labs), np.concatenate(batch_labs_spk)

    def next_file(self):

        if self.is_train==1:
            print ('erro data')
            return np.zeros([1,1]), 'erro'


        start = self._index_in_epoch

        if self._index_in_epoch >= self.size_epoch:
            self._epoch_complate += 1
            data_framed =np.zeros([1,1])
            tt=''
            return data_framed, tt
        print(start)    
        tt = self.files[start]
        filename=self.savepath+'/'+self.files[start]

        self._index_in_epoch = self._index_in_epoch + 1

        try:
            file_data = readmatfile(filename)
        except Exception:
            data_framed =np.zeros([1,1])
            print(filename)
            return data_framed, tt


        data_framed = enframe(file_data, 1, 11, isframe=1)


        return data_framed, tt












