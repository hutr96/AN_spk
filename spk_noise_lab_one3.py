
from __future__ import print_function
import tensorflow as tf

import numpy as np

import datagenerater as dg

from datagenerater_new import *

import scipy.io as sio

from numpy.matlib import repmat

import os


import logging











# a linear layer

def linear(input, output_dim, scope=None, stddev=1.0):
    norm = tf.random_normal_initializer(stddev=stddev)

    const = tf.constant_initializer(0.0)

    with tf.variable_scope(scope):
        w = tf.get_variable("w", [input.get_shape()[1], output_dim], initializer=norm)

        b = tf.get_variable("b", [output_dim], initializer=const)

        return tf.matmul(input, w) + b


def batch_norm_wrapper(inputs, is_training, scope=None, decay=0.999):
    epsilon = 0.0001

    with tf.variable_scope(scope):

        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))

        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))

        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)

        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

        if is_training:

            batch_mean, batch_var = tf.nn.moments(inputs, [0])

            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))

            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

            with tf.control_dependencies([train_mean, train_var]):

                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)

        else:

            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)




class GAN(object):
    def __init__(self,model_path,logname):


        # Generate Layer
        self.g_h1_dim = 1024
        self.g_h2_dim = 1024
        self.g_h3_dim = 128




        # Discriminate speaker layer
        self.d_spk_h1 = 1024
        self.d_spk_h2 = 1024
        self.spk_lab_dim=50

        # Discriminate noise layer

        self.d_noise_h1=1024
        self.d_noise_h2=1024
        self.noise_lab_dim=7



        #
        self.input_dim = 627



        self.is_training = True





        self.model_path = model_path

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.logger = logging.getLogger('GAN-' + logname)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(logname + '.log')
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)

    def generator(self, input):

        # batch normallized input

        input_BN = batch_norm_wrapper(input, self.is_training, scope='in_G_BN')

        # hidden layer 0 on G

        h0 = linear(input_BN, self.g_h1_dim, 'g0')

        h0_BN = batch_norm_wrapper(h0, self.is_training, scope='g0_BN')

        h0_out = tf.nn.softplus(h0_BN, name='g0_out')

        # hiddenlayer 1 on G

        h1 = linear(h0_out, self.g_h2_dim, 'g1')

        h1_BN = batch_norm_wrapper(h1, self.is_training, scope='g1_BN')

        h1_out = tf.nn.softplus(h1_BN, name='g1_out')

        # hidden layer 2

        h2 = linear(h1_out, self.g_h3_dim, 'g2')

        h2_BN = batch_norm_wrapper(h2, self.is_training, scope='g2_BN')

        h2_out = tf.nn.tanh(h2_BN, name='g2_out')

        return h2_out

    def discriminator_spk(self, input):

        input_BN = batch_norm_wrapper(input, self.is_training, scope='in_D_BN')

        # hidden layer 0

        h0 = linear(input_BN, self.d_spk_h1, 'd0_spk')

        h0_BN = batch_norm_wrapper(h0, self.is_training, scope='d0_BN_spk')

        h0_out = tf.nn.sigmoid(h0_BN, name='d0_out_spk')

        # hiddenlayer 1

        h1 = linear(h0_out, self.d_spk_h2, 'd1_spk')

        h1_BN = batch_norm_wrapper(h1, self.is_training, scope='d1_BN_spk')

        h1_out = tf.nn.sigmoid(h1_BN, name='d1_out_spk')

        # output layer

        h2 = linear(h1_out, self.spk_lab_dim, 'd2_spk')

        h2_out = tf.nn.softmax(h2, name='d2_out_spk')

        return h2_out

    def discriminator_noise(self, input):

        input_BN = batch_norm_wrapper(input, self.is_training, scope='in_D_BN')

        # hidden layer 0

        h0 = linear(input_BN, self.d_noise_h1, 'd0_noise')

        h0_BN = batch_norm_wrapper(h0, self.is_training, scope='d0_BN_noise')

        h0_out = tf.nn.sigmoid(h0_BN, name='d0_out_noise')

        # hiddenlayer 1

        h1 = linear(h0_out, self.d_noise_h2, 'd1_noise')

        h1_BN = batch_norm_wrapper(h1, self.is_training, scope='d1_BN_noise')

        h1_out = tf.nn.sigmoid(h1_BN, name='d1_out_noise')

        # output layer

        h2 = linear(h1_out, self.noise_lab_dim, 'd2_noise')

        h2_out = tf.nn.softmax(h2, name='d2_out_noise')

        return h2_out


    def optimizer(self, loss, var_list, lab):

        initial_learning_rate = 0.005

        decay = 0.9

        num_decay_step = 150

        batch = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(

            initial_learning_rate,

            batch,

            num_decay_step,

            decay,

            staircase=True

        )
        if lab == 'G':
            learning_rate = learning_rate
        else:
            learning_rate = learning_rate

        m_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(

            loss,

            global_step=batch,

            var_list=var_list

        )

        return m_optimizer

    def creat_model(self):

        with tf.variable_scope('D_pre'):
            self.pre_input = tf.placeholder(tf.float32, shape=(None, self.input_dim))

            self.pre_labs_spk = tf.placeholder(tf.float32, shape=(None, self.spk_lab_dim))

            self.pre_labs_noise=tf.placeholder(tf.float32, shape=(None, self.noise_lab_dim))

            g_out = self.generator(self.pre_input)

            d_spk_out = self.discriminator_spk(g_out)

            d_noise_out=self.discriminator_noise(g_out)


            loss_spk=tf.reduce_mean(-tf.reduce_sum(self.pre_labs_spk * tf.log(d_spk_out + 1e-10), 1), 0)

            loss_noise=tf.reduce_mean(-tf.reduce_sum(self.pre_labs_noise * tf.log(d_noise_out + 1e-10), 1), 0)





            self.pre_loss = 0.7*loss_spk + 0.3*loss_noise
            #self.pre_loss = loss_noise
            self.pre_opt = self.optimizer(self.pre_loss, None, 'd')

        with tf.variable_scope('G'):
            self.G_input = tf.placeholder(tf.float32, shape=(None, self.input_dim))

            self.G = self.generator(self.G_input)

        with tf.variable_scope('D') as scope:
            self.labs_spk = tf.placeholder(tf.float32, shape=(None, self.spk_lab_dim))

            self.D1_spk = self.discriminator_spk(self.G)



            self.labs_noise = tf.placeholder(tf.float32, shape=(None, self.noise_lab_dim))

            self.D1_noise = self.discriminator_noise(self.G)

        loss_d_noise= tf.reduce_mean(-tf.reduce_sum(self.labs_noise * tf.log(self.D1_noise + 1e-10), 1), 0)

        loss_d_spk= tf.reduce_mean(-tf.reduce_sum(self.labs_spk * tf.log(self.D1_spk + 1e-10), 1), 0)
        
        loss_d_noise_ng= tf.reduce_mean(-tf.reduce_sum((1-self.labs_noise) * tf.log((self.D1_noise) + 1e-10), 1), 0)
        
        
        self.loss_d=0.7*loss_d_spk+ 0.3*loss_d_noise
            
        self.loss_g=loss_d_noise
        #self.loss_g=loss_d_noise_ng


        vars = tf.trainable_variables()

        self.d_pre_params_g = [v for v in vars if v.name.startswith('D_pre/g')]

        self.d_pre_params_d = [v for v in vars if v.name.startswith('D_pre/d')]

        self.d_params = [v for v in vars if v.name.startswith('D/')]

        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.opt_d = self.optimizer(self.loss_d, self.d_params, 'D')

        self.opt_g = self.optimizer(self.loss_g, self.g_params, 'G')

    def test_model(self, scp_file, model_name, savepath_in, savepath_out):

        self.is_training = False

        saver = tf.train.Saver()

        with tf.Session() as  session:

            saver.restore(session, model_name)

            m_testdata=dg.DataSet(scp_file,savepath_in,1,0)


            while 1:
                if m_testdata.get_epoch_complate()==1:
                    break

                data,filename=m_testdata.next_file()

                n, d = np.shape(data)

                if n < 3:
                    print(filename + 'ERRO\n')
                    self.logger.info(filename + 'ERRO')
                    continue

                m_G = session.run([self.G], {self.G_input: data})
                savename = savepath_out + '/' + filename

                pathname = os.path.dirname(savename)

                if not os.path.exists(pathname):
                    os.makedirs(pathname)
                sio.savemat(savename, {'data': m_G})



    def train_model(self,train_scp,num_pertrain,num_GAN):



        print('read data')

        # m_traindata=dg.DataSet(train_scp,'',64,1)

        saver = tf.train.Saver(max_to_keep=10000)

        with tf.Session() as session:

            # tf.global_variables_initializer().run()
            tf.initialize_all_variables().run()

            # pre train discriminator
            d = np.genfromtxt(train_scp, dtype=str)

            N = np.shape(d)[0]

            files_all=d[:, 1]
            labs_noise=d[:,2]
            labs_spk=d[:,0]
            self.batch_size=64
            for epoch in range(num_GAN):
                perm = np.arange(N)
                shuffle(perm)

                rand_train_files = [files_all[i] for i in perm]
                rand_train_labs_noise = [labs_noise[i] for i in perm]
                rand_train_labs_spk =[labs_spk[i] for i in perm]
                loss_g1=0;
                batch_idxs = N // self.batch_size
                for idx in range(0, batch_idxs):
                    
                    
                    zz= 'Epoch: ' + str(epoch) +' ' + str(idx) + '---' + str(batch_idxs)
                    print(zz)
                    batch_train_files = rand_train_files[idx*self.batch_size:(idx+1)*self.batch_size]
                    batch_train_labs_noise=rand_train_labs_noise[idx*self.batch_size:(idx+1)*self.batch_size]
                    batch_train_labs_spk = rand_train_labs_spk[idx*self.batch_size:(idx+1)*self.batch_size]
                    
                    batch_train=[readmat_lab(x,y,z) for (x,y,z) in zip(batch_train_files,batch_train_labs_noise,batch_train_labs_spk)]
                    
                    data=[x[0] for x in batch_train]
                    lab_spk=[x[2] for x in batch_train]
                    lab_noise=[x[1] for x in batch_train]
                    
                    data=np.concatenate(data)
                    lab_spk=np.concatenate(lab_spk)
                    lab_noise=np.concatenate(lab_noise)
                    
                    
                    N = np.shape(lab_noise)[0];
                    lab_G_noise = repmat([1, 0, 0, 0, 0, 0,0], N, 1)
                    #lab_G_noise = lab_noise;
                    tt = np.random.rand()
                    loss_d=0.0
                    #if tt<0.5:
                    loss_d1, _ = session.run([self.loss_d, self.opt_d], {self.G_input: data, self.labs_spk: lab_spk, self.labs_noise: lab_noise})
                    loss_d2, _ = session.run([self.loss_d, self.opt_d], {self.G_input: data, self.labs_spk: lab_spk, self.labs_noise: lab_noise})
                    if tt<0.5:
                        loss_g1, _ = session.run([self.loss_g, self.opt_g], {self.G_input: data, self.labs_spk: lab_spk, self.labs_noise: lab_G_noise})

                    #loss_g2, _ = session.run([self.loss_g, self.opt_g], {self.G_input: data, self.labs_spk: lab_spk, self.labs_noise: lab_G_noise})

                    #loss_g3, _ = session.run([self.loss_g, self.opt_g], {self.G_input: data, self.labs_spk: lab_spk, self.labs_noise: lab_G_noise})
                    
                    self.logger.info('step ' + str(epoch)+' Batch ' + str(idx) + 'loss_d1 ' + str(loss_d1) + ' loss_g1 ' + str(loss_g1) + ' loss_d2 ' + str(loss_d2) )#+ ' loss_g3 ' + str(loss_g3))
                
                
                
                
                temp_savename = self.model_path + '/yu_GAN' + str(epoch) + '.ckpt'
                saver.save(session, save_path=temp_savename)
            saver.save(session, save_path=self.model_path + '/yu_GAN.ckpt')         
            
            
            
            
            
result_path='/gpfs/gss1/work/aaudeeplearning/hongyu/GAN_DATA/result'            

extname='spk_noise_lab_one3'
my_modelpath =result_path+'/'+extname+'/model_temp'
my_logname =result_path+'/'+extname+'/log_temp'
print(my_modelpath)
print(my_logname)
print('creat1')
mymodel=GAN(my_modelpath,my_logname)
print('creat 2')
mymodel.creat_model()
print('train 3')
mymodel.train_model('scp/rand_all.scp',0,60)
print('end')

database_path='/gpfs/gss1/work/aaudeeplearning/hongyu/GAN_DATA' 
          
inpath = database_path+'/UBM/mfcc'
outpath = database_path+'/UBM/GAN_' + extname
scp_UBM='scp/UBM.scp'
my_model_name = my_modelpath+'/yu_GAN.ckpt'
mymodel.test_model(scp_file=scp_UBM, model_name=my_model_name, savepath_in=inpath, savepath_out=outpath)


#train clean

inpath = database_path+'/train_spk/mfcc/train_spk_clean'
outpath= database_path+'/train_spk/GAN_'+extname + '/train_spk_clean'
scp_train_spk='scp/train_spk.scp'
mymodel.test_model(scp_file=scp_train_spk, model_name=my_model_name, savepath_in=inpath,savepath_out=outpath)

#train noise
noises=['mok_ssn','mok_ped','mok_str', 'mok_caf','mok_bus','mok_bbl'];
snrs=['10','20']

for noise in noises:
    for snr in snrs:
        inpath = database_path+'/train_spk/mfcc/train_spk_'+noise+'/SNR_'+snr
        outpath= database_path+'/train_spk/GAN_'+extname + '/train_spk_' + noise + '/SNR_'+ snr
        mymodel.test_model(scp_file=scp_train_spk, model_name=my_model_name, savepath_in=inpath,savepath_out=outpath)




#test clean
inpath = database_path+'/test_spk/mfcc/test_spk_clean'
outpath= database_path+'/test_spk/GAN_'+extname + '/test_spk_clean'
scp_test_spk='scp/test_spk.scp'
mymodel.test_model(scp_file=scp_test_spk, model_name=my_model_name, savepath_in=inpath,savepath_out=outpath)

#test noise
snrs=['-5','00','05','10','15','20']
for noise in noises:
    for snr in snrs:
        inpath = database_path+'/test_spk/mfcc/test_spk_'+noise+'/SNR_'+snr
        outpath= database_path+'/test_spk/GAN_'+extname + '/test_spk_' + noise + '/SNR_'+ snr
        mymodel.test_model(scp_file=scp_test_spk, model_name=my_model_name, savepath_in=inpath,savepath_out=outpath)

# os.chdir("/gpfs/gss1/home/hongyu/GAN_denoise/matlabcode")

# cmd="matlab -nodisplay -r "+ ''' "doall('GAN_spk_noise_ngout')" '''

# os.system(cmd)











