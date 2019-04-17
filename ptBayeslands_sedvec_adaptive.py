

#Main Contributers:   Rohitash Chandra and Ratneel Deo  Email: c.rohitash@gmail.com, deo.ratneel@gmail.com

# Bayeslands II: Parallel tempering for multi-core systems - Badlands

from __future__ import print_function, division

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import random
import time
import operator
import math
import cmocean as cmo
from pylab import rcParams
import copy
from copy import deepcopy
import cmocean as cmo
from pylab import rcParams
import collections
from scipy import special
import fnmatch
import shutil
from PIL import Image
from io import StringIO
from cycler import cycler
import os
import sys
import matplotlib.mlab as mlab
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree
from scipy import stats 
from pyBadlands.model import Model as badlandsModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import itertools
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()
from plotly.offline.offline import _plot_html
import pandas
import argparse


#Initialise and parse inputs
parser=argparse.ArgumentParser(description='PTBayeslands modelling')

parser.add_argument('-p','--problem', help='Problem Number 1-crater-fast,2-crater,3-etopo-fast,4-etopo,5-null,6-mountain', required=True, dest="problem",type=int)
parser.add_argument('-s','--samples', help='Number of samples', default=10000, dest="samples",type=int)
parser.add_argument('-r','--replicas', help='Number of chains/replicas, best to have one per availble core/cpu', default=10,dest="num_chains",type=int)
parser.add_argument('-t','--temperature', help='Demoninator to determine Max Temperature of chains (MT=no.chains*t) ', default=10,dest="mt_val",type=int)
parser.add_argument('-swap','--swap', help='Swap Ratio', dest="swap_ratio",default=0.02,type=float)
parser.add_argument('-b','--burn', help='How many samples to discard before determing posteriors', dest="burn_in",default=0.25,type=float)
parser.add_argument('-pt','--ptsamples', help='Ratio of PT vs straight MCMC samples to run', dest="pt_samples",default=0.5,type=float)

args = parser.parse_args()
    
#parameters for Parallel Tempering
problem = args.problem
samples = args.samples #10000  # total number of samples by all the chains (replicas) in parallel tempering
num_chains = args.num_chains
swap_ratio = args.swap_ratio
burn_in=args.burn_in
#maxtemp = int(num_chains * 5)/args.mt_val
maxtemp =  num_chains*args.mt_val 
swap_interval = int(swap_ratio * (samples/num_chains)) #how ofen you swap neighbours
num_successive_topo = 4
pt_samples = samples*args.pt_samples




class ptReplica(multiprocessing.Process):
    def __init__(self,  num_param, vec_parameters, minlimits_vec, maxlimits_vec, stepratio_vec,   check_likelihood_sed ,  swap_interval, sim_interval, simtime, samples, real_elev,  real_erodep_pts, erodep_coords, filename, xmlinput,  run_nb, tempr, parameter_queue,event , main_proc,   burn_in):

        multiprocessing.Process.__init__(self)
        self.processID = tempr      
        self.parameter_queue = parameter_queue
        self.event = event
        self.signal_main = main_proc
        self.temperature = tempr
        self.swap_interval = swap_interval
        self.filename = filename
        self.input = xmlinput  
        self.simtime = simtime
        self.samples = samples
        self.run_nb = run_nb 
        self.num_param =  num_param
        self.font = 9
        self.width = 1 
        self.vec_parameters = np.asarray(vec_parameters)
        self.minlimits_vec = np.asarray(minlimits_vec)
        self.maxlimits_vec = np.asarray(maxlimits_vec)
        self.stepratio_vec = np.asarray(stepratio_vec)
        self.check_likelihood_sed =  check_likelihood_sed
        self.real_erodep_pts = real_erodep_pts
        self.erodep_coords = erodep_coords
        self.real_elev = real_elev
        self.runninghisto = True  
        self.burn_in = burn_in
        self.sim_interval = sim_interval
        self.sedscalingfactor = 50 # this is to ensure that the sediment likelihood is given more emphasis as it considers fewer points (dozens of points) when compared to elev liklihood (thousands of points)
        self.adapttemp =  self.temperature

        self.adapt_cov = 15
        self.cholesky = [] 
        self.cov_init = False
        self.use_cov = False
        self.cov_counter = 0 

    def interpolateArray(self, coords=None, z=None, dz=None):
        """
        Interpolate the irregular spaced dataset from badlands on a regular grid.
        """
        x, y = np.hsplit(coords, 2)
        dx = (x[1]-x[0])[0]

        nx = int((x.max() - x.min())/dx+1)
        ny = int((y.max() - y.min())/dx+1)
        xi = np.linspace(x.min(), x.max(), nx)
        yi = np.linspace(y.min(), y.max(), ny)

        xi, yi = np.meshgrid(xi, yi)
        xyi = np.dstack([xi.flatten(), yi.flatten()])[0]
        XY = np.column_stack((x,y))

        tree = cKDTree(XY)
        distances, indices = tree.query(xyi, k=3)
        if len(z[indices].shape) == 3:
            z_vals = z[indices][:,:,0]
            dz_vals = dz[indices][:,:,0]
        else:
            z_vals = z[indices]
            dz_vals = dz[indices]

        zi = np.average(z_vals,weights=(1./distances), axis=1)
        dzi = np.average(dz_vals,weights=(1./distances), axis=1)
        onIDs = np.where(distances[:,0] == 0)[0]
        if len(onIDs) > 0:
            zi[onIDs] = z[indices[onIDs,0]]
            dzi[onIDs] = dz[indices[onIDs,0]]
        zreg = np.reshape(zi,(ny,nx))
        dzreg = np.reshape(dzi,(ny,nx))
        return zreg,dzreg

    def run_badlands(self, input_vector):
        #Runs a badlands model with the specified inputs

        #Create a badlands model instance
        model = badlandsModel()

        # Load the XmL input file
        model.load_xml(str(self.run_nb), self.input, muted=True)

        # Adjust erodibility based on given parameter
        model.input.SPLero = input_vector[1] 
        model.flow.erodibility.fill(input_vector[1] )

        # Adjust precipitation values based on given parameter
        model.force.rainVal[:] = input_vector[0] 

        # Adjust m and n values
        model.input.SPLm = input_vector[2] 
        model.input.SPLn = input_vector[3] 

        #Check if it is the etopo extended problem
        if problem == 4 or problem == 3:  # will work for more parameters
            model.input.CDm = input_vector[4] # submarine diffusion
            model.input.CDa = input_vector[5] # aerial diffusion


        #Check if it is the mountain problem
        if problem==6:
            #Round the input vector 
            k=round(input_vector[4],1) #to closest 0.1

            #Load the current tectonic uplift parameters
            tectonicValues=pandas.read_csv(str(model.input.tectFile[0]),sep=r'\s+',header=None,dtype=np.float).values
        
            #Adjust the parameters by our value k, and save them out
            newFile = "Examples/mountain/tect/uplift"+str(self.temperature)+"_"+str(k)+".csv"
            newtect = pandas.DataFrame(tectonicValues*k)
            newtect.to_csv(newFile,index=False,header=False)
            
            #Update the model uplift tectonic values
            model.input.tectFile[0]=newFile

        elev_vec = collections.OrderedDict()
        erodep_vec = collections.OrderedDict()
        erodep_pts_vec = collections.OrderedDict()

        for x in range(len(self.sim_interval)):
            self.simtime = self.sim_interval[x]
            
            model.run_to_time(self.simtime, muted=True)

            elev, erodep = self.interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)

            erodep_pts = np.zeros((self.erodep_coords.shape[0]))

            for count, val in enumerate(self.erodep_coords):
                erodep_pts[count] = erodep[val[0], val[1]]

            elev_vec[self.simtime] = elev
            erodep_vec[self.simtime] = erodep
            erodep_pts_vec[self.simtime] = erodep_pts

        return elev_vec, erodep_vec, erodep_pts_vec


    def likelihood_func(self,input_vector ):
        #print("Running likelihood function: ", input_vector)
        
        pred_elev_vec, pred_erodep_vec, pred_erodep_pts_vec = self.run_badlands(input_vector )
        tausq = np.sum(np.square(pred_elev_vec[self.simtime] - self.real_elev))/self.real_elev.size 
        tau_erodep =  np.zeros(self.sim_interval.size) 
        #print(self.sim_interval.size, self.real_erodep_pts.shape)
        for i in range(  self.sim_interval.size):
            tau_erodep[i]  =  np.sum(np.square(pred_erodep_pts_vec[self.sim_interval[i]] - self.real_erodep_pts[i]))/ self.real_erodep_pts.shape[1]

        likelihood_elev = - 0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(pred_elev_vec[self.simtime] - self.real_elev) / tausq 
        likelihood_erodep = 0 
        
        if self.check_likelihood_sed  == True: 

            for i in range(1, self.sim_interval.size):
                likelihood_erodep  += np.sum(-0.5 * np.log(2 * math.pi * tau_erodep[i]) - 0.5 * np.square(pred_erodep_pts_vec[self.sim_interval[i]] - self.real_erodep_pts[i]) / tau_erodep[i]) # only considers point or core of erodep
        
            likelihood = np.sum(likelihood_elev) +  (likelihood_erodep * self.sedscalingfactor)

        else:
            likelihood = np.sum(likelihood_elev)

        rmse_elev = np.sqrt(tausq)
        rmse_erodep = np.sqrt(tau_erodep) 
        avg_rmse_er = np.average(rmse_erodep)

        return [likelihood *(1.0/self.adapttemp), pred_elev_vec, pred_erodep_pts_vec, likelihood, rmse_elev, avg_rmse_er]

    def computeCovariance(self, i, pos_v):
        cov_mat = np.cov(pos_v[:i,].T)
        # np.savetxt('%s/cov_mat_%s.txt' %(self.filename,self.temperature), cov_mat )
        cov_noise = self.stepratio_vec*np.identity(cov_mat.shape[0], dtype = float)
        covariance = np.add(cov_mat, cov_noise)        
        L = np.linalg.cholesky(covariance)
        self.cholesky = L
        self.cov_init = True
        # self.cov_counter += 1 

    def run(self):

        #This is a chain that is distributed to many cores. AKA a 'Replica' in Parallel Tempering

        samples = self.samples
        count_list = [] 
        stepsize_vec = np.zeros(self.maxlimits_vec.size)
        span = (self.maxlimits_vec-self.minlimits_vec) 

        for i in range(stepsize_vec.size): # calculate the step size of each of the parameters
            stepsize_vec[i] = self.stepratio_vec[i] * span[i]

        v_proposal = self.vec_parameters # initial param values passed to badlands
        v_current = v_proposal # to give initial value of the chain

        #  initial predictions from Badlands model
        print("Intital parameter predictions: ", v_current)
        initial_predicted_elev, initial_predicted_erodep, init_pred_erodep_pts_vec = self.run_badlands(v_current)
        
        #calc initial likelihood with initial parameters
        [likelihood, predicted_elev,  pred_erodep_pts, likl_without_temp, avg_rmse_el, avg_rmse_er] = self.likelihood_func(v_current )

        print('\tinitial likelihood:', likelihood)

        likeh_list = np.zeros((samples,2)) # one for posterior of likelihood and the other for all proposed likelihood
        likeh_list[0,:] = [-10000, -10000] # to avoid prob in calc of 5th and 95th percentile   later

        count_list.append(0) # just to count number of accepted for each chain (replica)
        accept_list = np.zeros(samples)
        

        #---------------------------------------
        #now, create memory to save all the accepted tau proposals
        prev_accepted_elev = deepcopy(predicted_elev)
        prev_acpt_erodep_pts = deepcopy(pred_erodep_pts) 
        sum_elev = deepcopy(predicted_elev)
        sum_erodep_pts = deepcopy(pred_erodep_pts)

        #print('time to change')
        burnsamples = int(samples*self.burn_in)
        
        #---------------------------------------
        #now, create memory to save all the accepted   proposals of rain, erod, etc etc, plus likelihood
        pos_param = np.zeros((samples,v_current.size)) 
        list_yslicepred = np.zeros((samples,self.real_elev.shape[0]))  # slice mid y axis  
        list_xslicepred = np.zeros((samples,self.real_elev.shape[1])) # slice mid x axis  
        ymid = int(self.real_elev.shape[1]/2 ) 
        xmid = int(self.real_elev.shape[0]/2)
        list_erodep  = np.zeros((samples,pred_erodep_pts[self.simtime].size))
        list_erodep_time  = np.zeros((samples , self.sim_interval.size , pred_erodep_pts[self.simtime].size))

        start = time.time() 

        num_accepted = 0
        num_div = 0 

        #pt_samples = samples * 0.5 # this means that PT in canonical form with adaptive temp will work till pt  samples are reached. Set in arguments, default 0.5

        init_count = 0

        rmse_elev  = np.zeros(samples)
        rmse_erodep = np.zeros(samples)

        #save

        with file(('%s/experiment_setting.txt' % (self.filename)),'a') as outfile:
            outfile.write('\nsamples_per_chain:,{0}'.format(self.samples))
            outfile.write('\nburnin:,{0}'.format(self.burn_in))
            outfile.write('\nnum params:,{0}'.format(self.num_param))
            outfile.write('\ninitial_proposed_vec:,{0}'.format(v_proposal))
            outfile.write('\nstepsize_vec:,{0}'.format(stepsize_vec))  
            outfile.write('\nstep_ratio_vec:,{0}'.format(self.stepratio_vec)) 
            outfile.write('\nswap interval:,{0}'.format(self.swap_interval))
            outfile.write('\nsim interval:,{0}'.format(self.sim_interval))
            outfile.write('\nlikelihood_sed (T/F):,{0}'.format(self.check_likelihood_sed))
            outfile.write('\nerodep_coords:,{0}'.format(self.erodep_coords))
            outfile.write('\nsed scaling factor:,{0}'.format(self.sedscalingfactor))

        for i in range(samples-1):

            print ("Temperature: ", self.temperature, ' Sample: ', i ,"/",samples)

            if i < pt_samples:
                self.adapttemp =  self.temperature #* ratio  #

            if i == pt_samples and init_count ==0: # move to MCMC canonical
                self.adapttemp = 1
                [likelihood, predicted_elev,  pred_erodep_pts, likl_without_temp, avg_rmse_el, avg_rmse_er] = self.likelihood_func(v_proposal) 
                init_count = 1


            if self.cov_init: # and use_cov:        
                # print ('self. cholesky', self.cholesky)
                # print(' v_prop %s' %self.temperature, v_p)
                # print ('v_current shape ', v_current.shape) 
                v_p = np.random.normal(size = v_current.shape)
                # print ('\nv_proposal shapessssssss', v_proposal.shape , v_p.shape)
                # print ('\nv_proposal shapessssssss', v_proposal , v_p)

                v_proposal = v_current + np.dot(self.cholesky,v_proposal)
                # print ('\n Proposal after using v_proposal %s' % (int(self.temperature)), v_proposal)
                
                v_proposal = v_current + np.dot(self.cholesky,v_p)
                # print ('\n Proposal after using v_p %s' %(int(self.temperature)), v_proposal)

            # Update by perturbing all the  parameters via "random-walk" sampler and check limits
            else:    
                v_proposal =  np.random.normal(v_current,stepsize_vec)

            for j in range(v_current.size):
                if v_proposal[j] > self.maxlimits_vec[j]:
                    v_proposal[j] = v_current[j]
                elif v_proposal[j] < self.minlimits_vec[j]:
                    v_proposal[j] = v_current[j]

            # print('v_proposal_ %s '%self.temperature,v_proposal_)  
            # Passing paramters to calculate likelihood and rmse with new tau
            
            [likelihood_proposal, predicted_elev,  pred_erodep_pts, likl_without_temp, avg_rmse_el, avg_rmse_er] = self.likelihood_func(v_proposal)

            final_predtopo= predicted_elev[self.simtime]
            pred_erodep = pred_erodep_pts[self.simtime]

            # Difference in likelihood from previous accepted proposal
            diff_likelihood = likelihood_proposal - likelihood

            try:
                mh_prob = min(1, math.exp(diff_likelihood))
            except OverflowError as e:
                mh_prob = 1

            u = random.uniform(0,1)
            
            accept_list[i+1] = num_accepted
            likeh_list[i+1,0] = likelihood_proposal

            if u < mh_prob: # Accept sample
                # Append sample number to accepted list
                count_list.append(i)            
                
                likelihood = likelihood_proposal
                v_current = v_proposal
                pos_param[i+1,:] = v_current # features rain, erodibility and others  (random walks is only done for this vector)
                likeh_list[i + 1,1]=likelihood  # contains  all proposal liklihood (accepted and rejected ones)
                list_yslicepred[i+1,:] =  final_predtopo[:, ymid] # slice taken at mid of topography along y axis  
                list_xslicepred[i+1,:]=   final_predtopo[xmid, :]  # slice taken at mid of topography along x axis 
                list_erodep[i+1,:] = pred_erodep
                rmse_elev[i+1,] = avg_rmse_el
                rmse_erodep[i+1,] = avg_rmse_er

                for x in range(self.sim_interval.size): 
                    list_erodep_time[i+1,x, :] = pred_erodep_pts[self.sim_interval[x]]

                num_accepted = num_accepted + 1 
                prev_accepted_elev.update(predicted_elev)

                if i>burnsamples: 
                    
                    for k, v in prev_accepted_elev.items():
                        sum_elev[k] += v 

                    for k, v in pred_erodep_pts.items():
                        sum_erodep_pts[k] += v

                    num_div += 1

            else: # Reject sample
                likeh_list[i + 1, 1]=likeh_list[i,1] 
                pos_param[i+1,:] = pos_param[i,:]
                list_yslicepred[i+1,:] =  list_yslicepred[i,:] 
                list_xslicepred[i+1,:]=   list_xslicepred[i,:]
                list_erodep[i+1,:] = list_erodep[i,:]
                list_erodep_time[i+1,:, :] = list_erodep_time[i,:, :]
                rmse_elev[i+1,] = rmse_elev[i,] 
                rmse_erodep[i+1,] = rmse_erodep[i,]
            
                if i>burnsamples:

                    for k, v in prev_accepted_elev.items():
                        sum_elev[k] += v

                    for k, v in prev_acpt_erodep_pts.items():
                        sum_erodep_pts[k] += v

                    num_div += 1

            if (i >= self.adapt_cov and i % self.adapt_cov == 0) :
                print ('\ncov computed = i ',i, '\n')
                self.computeCovariance(i,pos_param)

            if ( i % self.swap_interval == 0 ):
                others = np.asarray([likelihood])
                param = np.concatenate([v_current,others,np.asarray([self.temperature])])     

                # paramater placed in queue for swapping between chains
                self.parameter_queue.put(param)
                
                #signal main process to start and start waiting for signal for main
                self.signal_main.set()              
                self.event.wait()
                
                # retrieve parametsrs fom ques if it has been swapped
                if not self.parameter_queue.empty() : 
                    try:
                        result =  self.parameter_queue.get()
                        v_current= result[0:v_current.size]     
                        likelihood = result[v_current.size]

                    except:
                        print ('error')

                else:
                    print("  ")
                    
                self.event.clear()

        #----------end for loop of samples----------------------------- 


        accepted_count =  len(count_list) 
        accept_ratio = accepted_count / (samples * 1.0) * 100
        others = np.asarray([ likelihood])
        param = np.concatenate([v_current,others,np.asarray([self.temperature])])   

        print("param first:",param)
        print("v_current",v_current)
        print("others",others)
        print("temp",np.asarray([self.temperature]))
        
        self.parameter_queue.put(param)

        #Save out the data for each chain
        file_name = self.filename+'/posterior/pos_parameters/chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name,pos_param ) 

        file_name = self.filename+'/posterior/predicted_topo/chain_xslice_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name, list_xslicepred )

        file_name = self.filename+'/posterior/predicted_topo/chain_yslice_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name, list_yslicepred )

        file_name = self.filename+'/posterior/pos_likelihood/chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name,likeh_list, fmt='%1.2f')

        file_name = self.filename + '/posterior/accept_list/chain_' + str(self.temperature) + '_accept.txt'
        np.savetxt(file_name, [accept_ratio], fmt='%1.2f')

        file_name = self.filename + '/posterior/accept_list/chain_' + str(self.temperature) + '.txt'
        np.savetxt(file_name, accept_list, fmt='%1.2f') 

        file_name = self.filename+'/posterior/rmse_elev_chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name, rmse_elev, fmt='%1.2f')       
    
        file_name = self.filename+'/posterior/rmse_erodep_chain_'+ str(self.temperature)+ '.txt'
        np.savetxt(file_name, rmse_erodep, fmt='%1.2f')


        for s in range(self.sim_interval.size):  
            file_name = self.filename + '/posterior/predicted_erodep/chain_' + str(self.sim_interval[s]) + '_' + str(self.temperature) + '.txt'
            np.savetxt(file_name, list_erodep_time[:,s, :] , fmt='%.2f') 

        for k, v in sum_elev.items():
            sum_elev[k] = np.divide(sum_elev[k], num_div)
            mean_pred_elevation = sum_elev[k]

            sum_erodep_pts[k] = np.divide(sum_erodep_pts[k], num_div)
            mean_pred_erodep_pnts = sum_erodep_pts[k]

            file_name = self.filename + '/posterior/predicted_topo/chain_' + str(k) + '_' + str(self.temperature) + '.txt'
            np.savetxt(file_name, mean_pred_elevation, fmt='%.2f')

        self.signal_main.set()


class ParallelTempering:

    def __init__(self, vec_parameters, num_chains, maxtemp,NumSample,swap_interval, fname, realvalues_vec, num_param,  real_elev, erodep_pts, erodep_coords, simtime, siminterval, resolu_factor, run_nb, inputxml):


        self.swap_interval = swap_interval
        self.folder = fname
        self.maxtemp = maxtemp
        self.num_swap = 0
        self.num_chains = num_chains
        self.chains = []
        self.temperatures = []
        self.NumSamples = int(NumSample/self.num_chains)
        self.sub_sample_size = max(1, int( 0.05* self.NumSamples))
        self.show_fulluncertainity = False # needed in cases when you reall want to see full prediction of 5th and 95th percentile of topo. takes more space 
        self.real_erodep_pts  = erodep_pts
        self.real_elev = real_elev
        self.resolu_factor =  resolu_factor
        self.num_param = num_param
        self.erodep_coords = erodep_coords
        self.simtime = simtime
        self.sim_interval = siminterval
        self.run_nb =run_nb 
        self.xmlinput = inputxml
        self.vec_parameters = vec_parameters
        self.realvalues  =  realvalues_vec 
        
        # create queues for transfer of parameters between process chain
        #self.chain_parameters = [multiprocessing.Queue() for i in range(0, self.num_chains) ]
        self.parameter_queue = [multiprocessing.Queue() for i in range(num_chains)]
        self.chain_queue = multiprocessing.JoinableQueue()  
        self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]

        # two ways events are used to synchronize chains
        self.event = [multiprocessing.Event() for i in range (self.num_chains)]
        #self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]

        self.geometric =  True
        self.total_swap_proposals = 0


    def default_beta_ladder(self, ndim, ntemps, Tmax): #https://github.com/konqr/ptemcee/blob/master/ptemcee/sampler.py
        """
        Returns a ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined by the
        arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
        Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
        this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
        <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
        ``ntemps`` is also specified.
        
        """

        if type(ndim) != int or ndim < 1:
            raise ValueError('Invalid number of dimensions specified.')
        if ntemps is None and Tmax is None:
            raise ValueError('Must specify one of ``ntemps`` and ``Tmax``.')
        if Tmax is not None and Tmax <= 1:
            raise ValueError('``Tmax`` must be greater than 1.')
        if ntemps is not None and (type(ntemps) != int or ntemps < 1):
            raise ValueError('Invalid number of temperatures specified.')

        tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                        2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                        2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                        1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                        1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                        1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                        1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                        1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                        1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                        1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                        1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                        1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                        1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                        1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                        1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                        1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                        1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                        1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                        1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                        1.26579, 1.26424, 1.26271, 1.26121,
                        1.25973])

        if ndim > tstep.shape[0]:
            # An approximation to the temperature step at large
            # dimension
            tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndim)
        else:
            tstep = tstep[ndim-1]

        appendInf = False
        if Tmax == np.inf:
            appendInf = True
            Tmax = None
            ntemps = ntemps - 1

        if ntemps is not None:
            if Tmax is None:
                # Determine Tmax from ntemps.
                Tmax = tstep ** (ntemps - 1)
        else:
            if Tmax is None:
                raise ValueError('Must specify at least one of ``ntemps'' and '
                                'finite ``Tmax``.')

            # Determine ntemps from Tmax.
            ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

        betas = np.logspace(0, -np.log10(Tmax), ntemps)
        if appendInf:
            # Use a geometric spacing, but replace the top-most temperature with
            # infinity.
            betas = np.concatenate((betas, [0]))

        return betas
        
        
    def assign_temperatures(self):
        # #Linear Spacing
        # temp = 2
        # for i in range(0,self.num_chains):
        #   self.temperatures.append(temp)
        #   temp += 2.5 #(self.maxtemp/self.num_chains)
        #   print (self.temperatures[i])
        #Geometric Spacing

        if self.geometric == True:
            betas = self.default_beta_ladder(2, ntemps=self.num_chains, Tmax=self.maxtemp)      
            for i in range(0, self.num_chains):         
                self.temperatures.append(np.inf if betas[i] is 0 else 1.0/betas[i])
                print (self.temperatures[i])
        else:

            tmpr_rate = (self.maxtemp /self.num_chains)
            temp = 1
            print("Temperatures...")
            for i in xrange(0, self.num_chains):            
                self.temperatures.append(temp)
                temp += tmpr_rate
                print(self.temperatures[i])


    
    def initialize_chains (self,     minlimits_vec, maxlimits_vec, stepratio_vec,  check_likelihood_sed,   burn_in):
        self.burn_in = burn_in
        self.vec_parameters =   np.random.uniform(minlimits_vec, maxlimits_vec) # will begin from diff position in each replica (comment if not needed)
        self.assign_temperatures()
        
        for i in xrange(0, self.num_chains):
            self.chains.append(ptReplica( self.num_param, self.vec_parameters, minlimits_vec, maxlimits_vec, stepratio_vec,  check_likelihood_sed ,self.swap_interval, self.sim_interval,   self.simtime, self.NumSamples, self.real_elev,   self.real_erodep_pts, self.erodep_coords, self.folder, self.xmlinput,  self.run_nb,self.temperatures[i], self.parameter_queue[i],self.event[i], self.wait_chain[i],burn_in))
            
            
    def swap_procedure(self, parameter_queue_1, parameter_queue_2):
        #print (parameter_queue_2, ", param1:",parameter_queue_1)
        if parameter_queue_2.empty() is False and parameter_queue_1.empty() is False:
            param1 = parameter_queue_1.get()
            param2 = parameter_queue_2.get()
            lhood1 = param1[self.num_param]
            T1 = param1[self.num_param+1]
            lhood2 = param2[self.num_param]
            T2 = param2[self.num_param+1]

            #SWAPPING PROBABILITIES
            #old method
            swap_proposal =  (lhood1/[1 if lhood2 == 0 else lhood2])*(1/T1 * 1/T2)
            u = np.random.uniform(0,1)
            
            #new method (sandbridge et al.)
            #try:
            #    swap_proposal = min(1, 0.5*math.exp(lhood1-lhood2))
            #except OverflowError as e:
            #    print("overflow for swap prop, setting to 1")
            #    swap_proposal = 1
            
            if u < swap_proposal: 
                self.total_swap_proposals += 1
                self.num_swap += 1
                param_temp =  param1
                param1 = param2
                param2 = param_temp
            return param1, param2 

        else:
            self.total_swap_proposals += 1
            return
    

    def run_chains (self ):
        
        # only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
        swap_proposal = np.ones(self.num_chains-1) 
        
        # create parameter holders for paramaters that will be swapped
        replica_param = np.zeros((self.num_chains, self.num_param))  
        lhood = np.zeros(self.num_chains)

        # Define the starting and ending of MCMC Chains
        start = 0
        end = self.NumSamples-1
        number_exchange = np.zeros(self.num_chains)

        filen = open(self.folder + '/num_exchange.txt', 'a')

        #-------------------------------------------------------------------------------------
        # run the MCMC chains
        #-------------------------------------------------------------------------------------
        for l in range(0,self.num_chains):
            self.chains[l].start_chain = start
            self.chains[l].end = end
        
        #-------------------------------------------------------------------------------------
        # run the MCMC chains
        #-------------------------------------------------------------------------------------
        for j in range(0,self.num_chains):        
            self.chains[j].start()

        
        while True:
            count = 0
            for index in range(self.num_chains):
                if not self.chains[index].is_alive():
                    count+=1
                    print(str(self.chains[index].temperature) +" Dead")

            if count == self.num_chains:
                break
            print("Waiting for chains to finish...")
            timeout_count = 0
            for index in range(0,self.num_chains):
                print("Waiting for chain: {}".format(index+1))
                flag = self.wait_chain[index].wait(timeout=5)
                if flag:
                    print("Signal from chain: {}".format(index+1))
                    timeout_count += 1

            if timeout_count != self.num_chains:
                print("Skipping the swap!")
                continue
            print("Event occured")
            for index in range(0,self.num_chains-1):
                print('starting swap')
                try:
                    param_1, param_2, swapped = self.swap_procedure(self.parameter_queue[index],self.parameter_queue[index+1])
                    self.parameter_queue[index].put(param_1)
                    self.parameter_queue[index+1].put(param_2)
                    if index == 0:
                        if swapped:
                            swaps_appected_main += 1
                        total_swaps_main += 1
                except:
                    print("Nothing Returned by swap method!")
            for index in range (self.num_chains):
                    self.event[index].set()
                    self.wait_chain[index].clear()

        print("Joining processes")

        #JOIN THEM TO MAIN PROCESS
        for index in range(0,self.num_chains):
            self.chains[index].join()
        self.chain_queue.join()

        print(number_exchange, 'num_exchange, process ended')

        pos_param, likelihood_rep, accept_list, pred_topo,  combined_erodep, accept, pred_topofinal, list_xslice, list_yslice, rmse_elev, rmse_erodep = self.show_results('chain_')

        self.view_crosssection_uncertainity(list_xslice, list_yslice)

        optimal_para, para_5thperc, para_95thperc = self.get_uncertainity(likelihood_rep, pos_param)
        np.savetxt(self.folder+'/optimal_percentile_para.txt', [optimal_para, para_5thperc, para_95thperc] )


        for s in range(self.num_param):  
            self.plot_figure(pos_param[s,:], 'pos_distri_'+str(s), self.realvalues[s]  ) 




        for i in range(self.sim_interval.size):

            self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=pred_topo[i,:,:], title='Predicted Topography ', time_frame=self.sim_interval[i],  filename= 'mean')

        self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=  self.real_elev , title='Ground truth Topography', time_frame= self.simtime, filename = 'ground_truth')
        residual_elev = self.real_elev - pred_topo[self.sim_interval.size-1,:,:] 
        self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=  residual_elev , title='Residual Topography', time_frame= self.simtime, filename = 'residual_elev')
        #self.plot_figure(residual_elev.flatten(), 'residual_elev', []  ) 





        if self.show_fulluncertainity == True: # this to be used when you need output of the topo predictions - 5th and 95th percentiles

            pred_elev5th, pred_eroddep5th, pred_erd_pts5th = self.run_badlands(np.asarray(para_5thperc)) 
        
            self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=pred_elev5th[self.simtime], title='Pred. Topo. - 5th Percentile', time_frame= self.simtime, filename= '5th')

            pred_elev95th, pred_eroddep95th, pred_erd_pts95th = self.run_badlands(para_95thperc)
        
            self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=pred_elev95th[self.simtime], title='Pred. Topo. - 95th Percentile', time_frame= self.simtime, filename = '95th')

            pred_elevoptimal, pred_eroddepoptimal, pred_erd_optimal = self.run_badlands(optimal_para)
        
            self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=pred_elevoptimal[self.simtime], title='Pred. Topo. - Optimal', time_frame= self.simtime, filename = 'optimal')

            
    

        swap_perc = self.num_swap*100/self.total_swap_proposals  
            

        return (pos_param,likelihood_rep, accept_list,   combined_erodep,  pred_topofinal, swap_perc, accept,  rmse_elev, rmse_erodep)

     

    def view_crosssection_uncertainity(self,  list_xslice, list_yslice):
        print ('list_xslice', list_xslice.shape)
        print ('list_yslice', list_yslice.shape)

        ymid = int(self.real_elev.shape[1]/2 ) #   cut the slice in the middle 
        xmid = int(self.real_elev.shape[0]/2)

        print( 'ymid',ymid)
        print( 'xmid', xmid)
        print(self.real_elev)
        print(self.real_elev.shape, ' shape')

        x_ymid_real = self.real_elev[xmid, :] 
        y_xmid_real = self.real_elev[:, ymid ] 
        x_ymid_mean = list_xslice.mean(axis=1)

        print( x_ymid_real.shape , ' x_ymid_real shape')
        print( x_ymid_mean.shape , ' x_ymid_mean shape')
        
        x_ymid_5th = np.percentile(list_xslice, 5, axis=1)
        x_ymid_95th= np.percentile(list_xslice, 95, axis=1)

        y_xmid_mean = list_yslice.mean(axis=1)
        y_xmid_5th = np.percentile(list_yslice, 5, axis=1)
        y_xmid_95th= np.percentile(list_yslice, 95, axis=1)


        x = np.linspace(0, x_ymid_mean.size * self.resolu_factor, num=x_ymid_mean.size) 
        x_ = np.linspace(0, y_xmid_mean.size * self.resolu_factor, num=y_xmid_mean.size)

        #ax.set_xlim(-width,len(ind)+width)

        self.cross_section(x, x_ymid_mean, x_ymid_real, x_ymid_5th, x_ymid_95th, 'x_ymid_cross')
        self.cross_section(x_, y_xmid_mean, y_xmid_real, y_xmid_5th, y_xmid_95th, 'y_xmid_cross')


       

    def cross_section(self, x, pred, real, lower, higher, fname):

        size = 15

        plt.tick_params(labelsize=size)
        params = {'legend.fontsize': size, 'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.plot(x,  real, label='Ground Truth') 
        plt.plot(x, pred, label='Badlands Pred.') 
        plt.grid(alpha=0.75)

        rmse_init = np.sqrt(np.sum(np.square(pred  -  real))  / real.size)   

        plt.fill_between(x, lower , higher, facecolor='g', alpha=0.2, label = 'Uncertainty')
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.legend(loc='best') 
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)

        plt.title("Topography  cross section   ", fontsize = size)
        plt.xlabel(' Distance (km)  ', fontsize = size)
        plt.ylabel(' Height (m)', fontsize = size)
        plt.tight_layout()
          
        plt.savefig(self.folder+'/'+fname+'.pdf')
        plt.clf()

        return rmse_init


    # Merge different MCMC chains y stacking them on top of each other
    def show_results(self, filename):

        burnin = int(self.NumSamples * self.burn_in)
        pos_param = np.zeros((self.num_chains, self.NumSamples - burnin, self.num_param))
        list_xslice = np.zeros((self.num_chains, self.NumSamples - burnin, self.real_elev.shape[1]))
        list_yslice = np.zeros((self.num_chains, self.NumSamples - burnin, self.real_elev.shape[0]))
        likehood_rep = np.zeros((self.num_chains, self.NumSamples - burnin, 2 )) # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
        accept_percent = np.zeros((self.num_chains, 1))
        accept_list = np.zeros((self.num_chains, self.NumSamples )) 
        topo  = self.real_elev
        replica_topo = np.zeros((self.sim_interval.size, self.num_chains, topo.shape[0], topo.shape[1])) #3D
        combined_topo = np.zeros(( self.sim_interval.size, topo.shape[0], topo.shape[1]))
        replica_erodep_pts = np.zeros(( self.num_chains, self.real_erodep_pts.shape[1] )) 
        combined_erodep = np.zeros((self.sim_interval.size, self.num_chains, self.NumSamples - burnin, self.real_erodep_pts.shape[1] ))
        timespan_erodep = np.zeros((self.sim_interval.size,  (self.NumSamples - burnin) * self.num_chains, self.real_erodep_pts.shape[1] ))
        rmse_elev = np.zeros((self.num_chains, self.NumSamples-burnin))
        rmse_erodep = np.zeros((self.num_chains, self.NumSamples-burnin))


        for i in range(self.num_chains):
            file_name = self.folder + '/posterior/pos_parameters/'+filename + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name) 
            pos_param[i, :, :] = dat[burnin:,:]

            file_name = self.folder + '/posterior/predicted_topo/chain_xslice_'+  str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name) 
            list_xslice[i, :, :] = dat[burnin:,:] 

            file_name = self.folder + '/posterior/predicted_topo/chain_yslice_'+  str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name) 
            list_yslice[i, :, :] = dat[burnin:,:] 

            file_name = self.folder + '/posterior/pos_likelihood/'+filename + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name) 
            likehood_rep[i, :] = dat[burnin:]

            file_name = self.folder + '/posterior/accept_list/' + filename + str(self.temperatures[i]) + '.txt'
            dat = np.loadtxt(file_name) 
            accept_list[i, :] = dat 

            file_name = self.folder + '/posterior/accept_list/' + filename + str(self.temperatures[i]) + '_accept.txt'
            dat = np.loadtxt(file_name) 
            accept_percent[i, :] = dat

            file_name = self.folder+'/posterior/rmse_elev_chain_'+ str(self.temperatures[i])+ '.txt'
            dat = np.loadtxt(file_name)
            rmse_elev[i,:] = dat[burnin:]

            file_name = self.folder+'/posterior/rmse_erodep_chain_'+ str(self.temperatures[i])+ '.txt'
            dat = np.loadtxt(file_name)
            rmse_erodep[i,:] = dat[burnin:]


            for j in range(self.sim_interval.size):

                file_name = self.folder+'/posterior/predicted_topo/chain_'+str(self.sim_interval[j])+'_'+ str(self.temperatures[i])+ '.txt'
                dat_topo = np.loadtxt(file_name)
                replica_topo[j,i,:,:] = dat_topo

                file_name = self.folder+'/posterior/predicted_erodep/chain_'+str(self.sim_interval[j])+'_'+ str(self.temperatures[i])+ '.txt'
                dat_erodep = np.loadtxt(file_name)
                combined_erodep[j,i,:,:] = dat_erodep[burnin:,:]


        posterior = pos_param.transpose(2,0,1).reshape(self.num_param,-1)    
        xslice = list_xslice.transpose(2,0,1).reshape(self.real_elev.shape[1],-1) 
        yslice = list_yslice.transpose(2,0,1).reshape(self.real_elev.shape[0],-1) 

        rmse_elev = rmse_elev.reshape(self.num_chains*(self.NumSamples - burnin),1)
        rmse_erodep = rmse_erodep.reshape(self.num_chains*(self.NumSamples - burnin),1)

        likelihood_vec = likehood_rep.transpose(2,0,1).reshape(2,-1) 


        for j in range(self.sim_interval.size):
            for i in range(self.num_chains):
                combined_topo[j,:,:] += replica_topo[j,i,:,:]  
            combined_topo[j,:,:] = combined_topo[j,:,:]/self.num_chains

            dx = combined_erodep[j,:,:,:].transpose(2,0,1).reshape(self.real_erodep_pts.shape[1],-1)

            timespan_erodep[j,:,:] = dx.T


        accept = np.sum(accept_percent)/self.num_chains

        pred_topofinal = combined_topo[-1,:,:] # get the last mean pedicted topo to calculate mean squared error loss 

        np.savetxt(self.folder + '/pos_param.txt', posterior.T) 
        np.savetxt(self.folder + '/likelihood.txt', likelihood_vec.T, fmt='%1.5f')
        np.savetxt(self.folder + '/accept_list.txt', accept_list, fmt='%1.2f')
        np.savetxt(self.folder + '/acceptpercent.txt', [accept], fmt='%1.2f')

        return posterior, likelihood_vec.T, accept_list, combined_topo,   timespan_erodep, accept, pred_topofinal, xslice, yslice, rmse_elev, rmse_erodep


    def find_nearest(self, array,value): # just to find nearest value of a percentile (5th or 9th from pos likelihood)
        idx = (np.abs(array-value)).argmin()
        return array[idx], idx

    def get_uncertainity(self, likehood_rep, pos_param ): 

        likelihood_pos = likehood_rep[:,1]

        a = np.percentile(likelihood_pos, 5)   
        lhood_5thpercentile, index_5th = self.find_nearest(likelihood_pos,a)  
        b = np.percentile(likelihood_pos, 95) 
        lhood_95thpercentile, index_95th = self.find_nearest(likelihood_pos,b)  
        max_index = np.argmax(likelihood_pos) # find max of pos liklihood to get the max or optimal pos value  

        optimal_para = pos_param[:, max_index] 
        para_5thperc = pos_param[:, index_5th]
        para_95thperc = pos_param[:, index_95th] 

        return optimal_para, para_5thperc, para_95thperc


    # this is same method in Replica class - copied here to get error uncertainity in topo pred
    def run_badlands(self, input_vector):
        #Runs a badlands model with the specified inputs

        #Create a badlands model instance
        model = badlandsModel()

        # Load the XmL input file
        model.load_xml(str(self.run_nb), self.input, muted=True)

        # Adjust erodibility based on given parameter
        model.input.SPLero = input_vector[1] 
        model.flow.erodibility.fill(input_vector[1] )

        # Adjust precipitation values based on given parameter
        model.force.rainVal[:] = input_vector[0] 

        # Adjust m and n values
        model.input.SPLm = input_vector[2] 
        model.input.SPLn = input_vector[3] 

        #Check if it is the etopo extended problem
        if problem == 4 or problem == 3:  # will work for more parameters
            model.input.CDm = input_vector[4] # submarine diffusion
            model.input.CDa = input_vector[5] # aerial diffusion


        #Check if it is the mountain problem
        if problem==6:
            #Round the input vector 
            k=round(input_vector[4],1) #to closest 0.1

            #Load the current tectonic uplift parameters
            tectonicValues=pandas.read_csv(str(model.input.tectFile[0]),sep=r'\s+',header=None,dtype=np.float).values
        
            #Adjust the parameters by our value k, and save them out
            newFile = "Examples/mountain/tect/uplift"+str(self.temperature)+"_"+str(k)+".csv"
            newtect = pandas.DataFrame(tectonicValues*k)
            newtect.to_csv(newFile,index=False,header=False)
            
            #Update the model uplift tectonic values
            model.input.tectFile[0]=newFile

        elev_vec = collections.OrderedDict()
        erodep_vec = collections.OrderedDict()
        erodep_pts_vec = collections.OrderedDict()

        for x in range(len(self.sim_interval)):
            self.simtime = self.sim_interval[x]
            
            model.run_to_time(self.simtime, muted=True)

            elev, erodep = self.interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)

            erodep_pts = np.zeros((self.erodep_coords.shape[0]))

            for count, val in enumerate(self.erodep_coords):
                erodep_pts[count] = erodep[val[0], val[1]]

            elev_vec[self.simtime] = elev
            erodep_vec[self.simtime] = erodep
            erodep_pts_vec[self.simtime] = erodep_pts

        return elev_vec, erodep_vec, erodep_pts_vec


    def interpolateArray(self, coords=None, z=None, dz=None):
        """
        Interpolate the irregular spaced dataset from badlands on a regular grid.
        """
        x, y = np.hsplit(coords, 2)
        dx = (x[1]-x[0])[0]

        nx = int((x.max() - x.min())/dx+1)
        ny = int((y.max() - y.min())/dx+1)
        xi = np.linspace(x.min(), x.max(), nx)
        yi = np.linspace(y.min(), y.max(), ny)

        xi, yi = np.meshgrid(xi, yi)
        xyi = np.dstack([xi.flatten(), yi.flatten()])[0]
        XY = np.column_stack((x,y))

        tree = cKDTree(XY)
        distances, indices = tree.query(xyi, k=3)
        if len(z[indices].shape) == 3:
            z_vals = z[indices][:,:,0]
            dz_vals = dz[indices][:,:,0]
        else:
            z_vals = z[indices]
            dz_vals = dz[indices]

        zi = np.average(z_vals,weights=(1./distances), axis=1)
        dzi = np.average(dz_vals,weights=(1./distances), axis=1)
        onIDs = np.where(distances[:,0] == 0)[0]
        if len(onIDs) > 0:
            zi[onIDs] = z[indices[onIDs,0]]
            dzi[onIDs] = dz[indices[onIDs,0]]
        zreg = np.reshape(zi,(ny,nx))
        dzreg = np.reshape(dzi,(ny,nx))
        return zreg,dzreg



    def plot_figure(self, list, title, real_value ): 

        list_points =  list
        fname = self.folder
         


        size = 15


        plt.tick_params(labelsize=size)
        params = {'legend.fontsize': size, 'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.grid(alpha=0.75)

        plt.hist(list_points,  bins = 20, color='#0504aa',
                            alpha=0.7)   

        plt.title("Posterior distribution ", fontsize = size)
        plt.xlabel(' Parameter value  ', fontsize = size)
        plt.ylabel(' Frequency ', fontsize = size)
        plt.axvline(x=real_value, linewidth=2, color='r')
        plt.tight_layout()  
        plt.savefig(fname + '/pos_plots/' + title  + '_posterior.pdf')
        plt.clf()


        plt.tick_params(labelsize=size)
        params = {'legend.fontsize': size, 'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.grid(alpha=0.75)

        listx = np.asarray(np.split(list_points,  self.num_chains ))
        plt.plot(listx.T)   

        plt.title("Parameter trace plot", fontsize = size)
        plt.xlabel(' Number of Samples  ', fontsize = size)
        plt.ylabel(' Parameter value ', fontsize = size)
        plt.tight_layout()  
        plt.savefig(fname + '/pos_plots/' + title  + '_trace.pdf')
        plt.clf()


        #---------------------------------------
        



    def viewGrid(self, width=1000, height=1000, zmin=None, zmax=None, zData=None, title='Predicted Topography', time_frame=None, filename=None):

        if zmin == None:
            zmin =  zData.min()

        if zmax == None:
            zmax =  zData.max()

        tickvals= [0,50,75,-50]

        xx = (np.linspace(0, zData.shape[0]* self.resolu_factor, num=zData.shape[0]/10 )) 
        yy = (np.linspace(0, zData.shape[1] * self.resolu_factor, num=zData.shape[1]/10 )) 

        xx = np.around(xx, decimals=0)
        yy = np.around(yy, decimals=0)
        print (xx,' xx')
        print (yy,' yy')

        # range = [0,zData.shape[0]* self.resolu_factor]
        #range = [0,zData.shape[1]* self.resolu_factor],


        #https://plot.ly/r/reference/#scatter3d 

        #https://plot.ly/python/reference/#layout-yaxis-title-font-size
        #https://plot.ly/r/reference/#heatmap-showscale



        axislabelsize = 20

        data = Data([Surface(x= zData.shape[0] , y= zData.shape[1] , z=zData, colorscale='YlGnBu')])

        layout = Layout(  autosize=True, width=width, height=height,scene=Scene(
                    zaxis=ZAxis(title = 'Elev.   ', range=[zmin,zmax], autorange=False, nticks=5, gridcolor='rgb(255, 255, 255)',
                                gridwidth=2, zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, showticklabels = True,  titlefont=dict(size=axislabelsize),  tickfont=dict(size=14 ),),
                    xaxis=XAxis(title = 'x-axis  ',  tickvals= xx,      gridcolor='rgb(255, 255, 255)', gridwidth=2,
                                zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, showticklabels = True,  titlefont=dict(size=axislabelsize),  tickfont=dict(size=14 ),),
                    yaxis=YAxis(title = 'y-axis  ', tickvals= yy,    gridcolor='rgb(255, 255, 255)', gridwidth=2,
                                zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2, showticklabels = True,  titlefont=dict(size=axislabelsize),  tickfont=dict(size=14 ),),
                    bgcolor="rgb(244, 244, 248)"
                )
            )

        fig = Figure(data=data, layout=layout) 
        graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename= self.folder +  '/pred_plots'+ '/pred_'+filename+'_'+str(time_frame)+ '_.html', validate=False)

        fname = self.folder + '/pred_plots'+'/pred_'+filename+'_'+str(time_frame)+ '_.pdf' 
        elev_data = np.reshape(zData, zData.shape[0] * zData.shape[1] )   
        hist, bin_edges = np.histogram(elev_data, density=True)

        size = 15 
        plt.tick_params(labelsize=size)
        params = {'legend.fontsize': size, 'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.hist(elev_data, bins='auto')  

        #plt.title("Topography")  
        plt.xlabel('Elevation (m)', fontsize = size)
        plt.ylabel('Frequency', fontsize = size)
        plt.grid(alpha=0.75)


        plt.tight_layout()  
        plt.savefig(fname )
        plt.clf()


# class  above this line -------------------------------------------------------------------------------------------------------


def mean_sqerror(  pred_erodep, pred_elev,  real_elev,  real_erodep_pts):
        
        elev = np.sqrt(np.sum(np.square(pred_elev -  real_elev))  / real_elev.size)  
        sed =  np.sqrt(  np.sum(np.square(pred_erodep -  real_erodep_pts)) / real_erodep_pts.size  ) 

        return elev + sed, sed


def make_directory (directory): 
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_erodeposition(erodep_mean, erodep_std, groundtruth_erodep_pts, sim_interval, fname):

    ticksize = 15

    fig = plt.figure()
    ax = fig.add_subplot(111)
    index = np.arange(groundtruth_erodep_pts.size) 
    ground_erodepstd = np.zeros(groundtruth_erodep_pts.size) 
    opacity = 0.8
    width = 0.35       # the width of the bars

    rects1 = ax.bar(index, erodep_mean, width,
                color='blue',
                yerr=erodep_std,
                error_kw=dict(elinewidth=2,ecolor='red'))

    rects2 = ax.bar(index+width, groundtruth_erodep_pts, width, color='green', 
                yerr=ground_erodepstd,
                error_kw=dict(elinewidth=2,ecolor='red') )
 

    ax.tick_params(labelsize=ticksize)
    ax.grid(alpha=0.75)
    ax.set_ylabel('Height in meters', fontsize=ticksize)
    ax.set_xlabel('Location ID ', fontsize=ticksize)
    ax.set_title('Erosion/Deposition', fontsize=ticksize)
 
    plotlegend = ax.legend( (rects1[0], rects2[0]), ('Predicted  ', ' Ground-truth ') )
    
    plt.savefig(fname +'/pos_erodep_'+str( sim_interval) +'_.pdf')
    plt.clf()    
  



def main():

    random.seed(time.time()) 

    if problem == 1:
        problemfolder = 'Examples/crater/'
        xmlinput = problemfolder + 'crater.xml'
        simtime = 50000
        resolu_factor =  0.002

        true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt') 

        print(true_parameter_vec, ' true_parameter_vec')

        m = 0.5 # used to be constants  
        n = 1

        real_rain = 1.5
        real_erod = 5.e-5 

        likelihood_sediment = True

        #Rainfall, erodibility, m, n
        #maxlimits_vec = [3.0,7.e-5, 2, 2] 
        #minlimits_vec = [0.0 ,3.e-5, 0, 0]  

        maxlimits_vec = [3.0,7.e-5, m, n] # setting to real values means its now fixed, not free parameter
        minlimits_vec = [0.0 ,3.e-5, m, n]  
        vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) 
        
        stepsize_ratio  = 0.02
        
        stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
        num_param = vec_parameters.size 

        erodep_coords =  np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]]) # need to hand pick given your problem

        if (true_parameter_vec.shape[0] != vec_parameters.size ) :
            print( 'vec_params != true_values.txt ',true_parameter_vec.shape,vec_parameters.size)
            print( 'make sure that this is updated in case when you intro more parameters. should have as many rows as parameters ') 
            
            return

    elif problem == 2:
        problemfolder = 'Examples/etopo/'
        xmlinput = problemfolder + 'etopo.xml'
        simtime = 1000000
        resolu_factor = 1

        true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
        likelihood_sediment = True


        real_rain = 1.5 #m/a
        real_erod = 5.e-6 
        m = 0.5  #Stream flow parameters
        n = 1 #
        real_cmarine = 5.e-1 # Marine diffusion coefficient [m2/a] -->
        real_caerial = 8.e-1 #aerial diffusion

        #Rainfall, erodibility, m, n, marine, aerial
        #minlimits_vec = [0.0, 3.e-6, 0, 0, 0.6, 0.3]
        #maxlimits_vec = [3.0, 7.e-6, 2, 2, 1.0, 0.7]


        minlimits_vec = [0.0, 3.e-6, m, n, real_cmarine, real_caerial]  # setting to real values means its now fixed, not free parameter
        maxlimits_vec = [3.0, 7.e-6, m, n,real_cmarine, real_caerial]
   
        vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
    
        stepsize_ratio  = 0.1 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

        stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
        num_param = vec_parameters.size

        print(vec_parameters) 

        erodep_coords = np.array([[42,10],[39,8],[75,51],[59,13],[40,5],[6,20],[14,66],[4,40],[72,73],[46,64]])  # need to hand pick given your problem

        if (true_parameter_vec.shape[0] != vec_parameters.size ) :
            print( 'vec_params != true_values.txt ',true_parameter_vec.shape,vec_parameters.size)
            print( 'make sure that this is updated in case when you intro more parameters. should have as many rows as parameters ') 
            
            return

    elif problem == 3:
        problemfolder = 'Examples/mountain/'
        xmlinput = problemfolder + 'mountain.xml'
        simtime = 1000000
        resolu_factor = 1
        true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
        likelihood_sediment = True

        #Set variables
        m = 0.5
        m_min = 0.
        m_max = 2
        
        n = 1.
        n_min = 0.
        n_max = 2.

        rain_real = 1.5
        rain_min = 0.
        rain_max = 3.

        erod_real = 5.e-6
        erod_min = 3.e-6
        erod_max = 7.e-6
                
        #uplift_real = 50000
        uplift_min = 0.1 # X uplift_real
        uplift_max = 5.0 # X uplift_real
                
        #Rainfall, erodibility, m, n, uplift
        minlimits_vec=[rain_min,erod_min,m_min,n_min,uplift_min]
        maxlimits_vec=[rain_max,erod_max,m_max,n_max,uplift_max]
                
        vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters

        stepsize_ratio  = 0.02 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

        stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
        stepratio_vec = [stepsize_ratio, stepsize_ratio, stepsize_ratio, stepsize_ratio, 0.02] 
        stepratio_vec = [0.1, 0.1, 0.1, 0.1, 0.1]
        print("steps: ", stepratio_vec)
        num_param = vec_parameters.size
        erodep_coords=np.array([[5,5],[10,10],[20,20],[30,30],[40,40],[50,50],[25,25],[37,30],[44,27],[46,10]])

        if (true_parameter_vec.shape[0] != vec_parameters.size ) :
            print( 'vec_params != true_values.txt ',true_parameter_vec.shape,vec_parameters.size)
            print( 'make sure that this is updated in case when you intro more parameters. should have as many rows as parameters ') 
            
            return

    else:
        print('choose some problem  ')

    datapath = problemfolder + 'data/final_elev.txt'
    groundtruth_elev = np.loadtxt(datapath)
    groundtruth_erodep = np.loadtxt(problemfolder + 'data/final_erdp.txt')
    groundtruth_erodep_pts = np.loadtxt(problemfolder + 'data/final_erdp_pts.txt')

    fname = ""
    run_nb = 0
    while os.path.exists(problemfolder +'results_%s' % (run_nb)):
        run_nb += 1
    if not os.path.exists(problemfolder +'results_%s' % (run_nb)):
        os.makedirs(problemfolder +'results_%s' % (run_nb))
        fname = (problemfolder +'results_%s' % (run_nb))

    #fname = ('sampleresults')

    make_directory((fname + '/posterior/pos_parameters')) 
    make_directory((fname + '/pos_plots')) 
    make_directory((fname + '/posterior/predicted_topo'))
    make_directory((fname + '/posterior/pos_likelihood'))
    make_directory((fname + '/posterior/accept_list'))
    make_directory((fname + '/posterior/predicted_erodep'))
    make_directory((fname + '/pred_plots'))

    run_nb_str = 'results_' + str(run_nb)

    #-------------------------------------------------------------------------------------
    # Number of chains of MCMC required to be run
    # PT is a multicore implementation must num_chains >= 2
    # Choose a value less than the numbe of core available (avoid context swtiching)
    #-------------------------------------------------------------------------------------
    #num_chains = int(sys.argv[3]) #8  
    #swap_ratio = 0.05   #adapt these 
    #burn_in =0.3 
    #num_successive_topo = 4
    #swap_interval =   int(swap_ratio * (samples/num_chains)) #how ofen you swap neighbours
    #print(swap_interval, ' swap')

    timer_start = time.time()

    sim_interval = np.arange(0,  simtime+1, simtime/num_successive_topo) # for generating successive topography
    print("Simulation time interval", sim_interval)


    #-------------------------------------------------------------------------------------
    #Create A a Patratellel Tempring object instance 
    #-------------------------------------------------------------------------------------
    pt = ParallelTempering(  vec_parameters, num_chains, maxtemp, samples,swap_interval,fname, true_parameter_vec, num_param  ,  groundtruth_elev,  groundtruth_erodep_pts , erodep_coords, simtime, sim_interval, resolu_factor, run_nb_str, xmlinput)
    
    #-------------------------------------------------------------------------------------
    # intialize the MCMC chains
    #-------------------------------------------------------------------------------------
    pt.initialize_chains(    minlimits_vec, maxlimits_vec, stepratio_vec, likelihood_sediment,   burn_in)

    #-------------------------------------------------------------------------------------
    #run the chains in a sequence in ascending order
    #-------------------------------------------------------------------------------------
    pos_param,likehood_rep, accept_list,   combined_erodep, pred_elev,  swap_perc, accept_per,  rmse_elev, rmse_erodep  = pt.run_chains()


    print('sucessfully sampled') 
    timer_end = time.time() 
    likelihood = likehood_rep[:,0] # just plot proposed likelihood  
    likelihood = np.asarray(np.split(likelihood,  num_chains ))

    plt.plot(likelihood.T)
    plt.savefig( fname+'/likelihood.pdf')
    plt.clf()
    size = 15 

    plt.tick_params(labelsize=size)
    params = {'legend.fontsize': size, 'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.plot(accept_list.T)
    plt.title("Replica Acceptance ", fontsize = size)
    plt.xlabel(' Number of Samples  ', fontsize = size)
    plt.ylabel(' Number Accepted ', fontsize = size)
    plt.grid(alpha=0.75)
    plt.tight_layout()
    plt.savefig( fname+'/accept_list.pdf' )
    plt.clf()


    for i in range(sim_interval.size):
        pos_ed  = combined_erodep[i, :, :] 
        erodep_mean = pos_ed.mean(axis=0)  
        erodep_std = pos_ed.std(axis=0)  
        plot_erodeposition(erodep_mean, erodep_std, groundtruth_erodep_pts[i,:], sim_interval[i], fname) 

    pred_erodep = np.zeros(( groundtruth_erodep_pts.shape[0], groundtruth_erodep_pts.shape[1] )) # just to get the right size

    for i in range(sim_interval.size): 
        pos_ed  = combined_erodep[i, :, :] # get final one for comparision
        pred_erodep[i,:] = pos_ed.mean(axis=0)   

    rmse, rmse_sed= mean_sqerror(  pred_erodep, pred_elev,  groundtruth_elev,  groundtruth_erodep_pts)

    
    print ('time taken  in minutes = ', (timer_end-timer_start)/60)
    print ('Folder: ', run_nb_str)
    np.savetxt(fname+'/time_sqerror.txt',[ (timer_end-timer_start)/60,  rmse_sed, rmse], fmt='%1.2f'  )

    print(pos_param)

    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)

    size = 15

    ax.tick_params(labelsize=size)

    plt.legend(loc='upper right') 

    ax.boxplot(pos_param.T) 
    ax.set_xlabel('Parameter ID', fontsize=size)
    ax.set_ylabel('Posterior', fontsize=size) 
    plt.title("Boxplot of Posterior", fontsize=size) 
    plt.savefig(fname+'/badlands_pos.pdf')
    
    print (num_chains, problemfolder, run_nb_str, (timer_end-timer_start)/60, rmse_sed, rmse)


    timer_end = time.time() 
    likelihood = likehood_rep[:,0] # just plot proposed likelihood  
    likelihood = np.asarray(np.split(likelihood,  num_chains ))

    rmse_el = np.mean(rmse_elev[:])
    rmse_el_std = np.std(rmse_elev[:])
    rmse_el_min = np.amin(rmse_elev[:])
    rmse_er = np.mean(rmse_erodep[:])
    rmse_er_std = np.std(rmse_erodep[:])
    rmse_er_min = np.amin(rmse_erodep[:])

    time_total = (timer_end-timer_start)/60

    resultingfile_db = open(problemfolder+'/master_result_file.txt','a+')  
    allres =  np.asarray([ problem, num_chains, maxtemp, samples,swap_interval,  rmse_el, 
                        rmse_er, rmse_el_std, rmse_er_std, rmse_el_min, 
                        rmse_er_min,  swap_perc, accept_per, time_total]) 
    print(allres, '  result')
        
    np.savetxt(resultingfile_db,   allres   , fmt='%1.4f',  newline=' ' )  
    xv=problemfolder+'_'+str(run_nb)
    np.savetxt(resultingfile_db, [xv]   ,  fmt="%s", newline=' \n' ) 

    print("NumChains, problem, folder, time, RMSE_sed, RMSE,samples,swap,maxtemp,burn")
    print (num_chains, problemfolder, run_nb_str, (timer_end-timer_start)/60, rmse_sed, rmse,samples, swap_ratio,maxtemp,burn_in)

    dir_name = fname + '/posterior'
    print(dir_name)
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)

    fname_remove = fname +'/pos_param.txt'

    if os.path.exists(fname_remove):  # comment if you wish to keep pos file
        os.remove(fname_remove)

    #stop()
if __name__ == "__main__": main()
