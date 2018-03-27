from __future__ import print_function, division
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


import fnmatch
import shutil
from PIL import Image
from io import StringIO
from cycler import cycler
import os

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from scipy.spatial import cKDTree
from scipy import stats 
from pyBadlands.model import Model as badlandsModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import itertools

#nb = 0 


import sys


import plotly
import plotly.plotly as py
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()
from plotly.offline.offline import _plot_html


class ptReplica(multiprocessing.Process):
	def __init__(self, vec_parameters, minlimits_vec, maxlimits_vec, stepratio_vec, realvalues_vec, check_likelihood_sed ,  swap_interval, sim_interval, muted, simtime, samples, real_elev,real_erodep, real_erodep_pts, erodep_coords, filename, xmlinput,  run_nb, tempr, parameter_queue,event , main_proc,   burn_in):
		 		#	vec_parameters, minlimits_vec, maxlimits_vec, stepratio_vec, self.realvalues_vec, check_likelihood_sed , sim_interval, muted, simtime, self.NumSamples, real_elev,  real_erodep, real_erodep_pts, erodep_coords, filename, xmlinput,  run_nb_str,self.tempratures[i], self.chain_parameters[i], self.event[i], self.wait_chain[i],self.swap_interval, burn_in))
		

		#--------------------------------------------------------
		multiprocessing.Process.__init__(self)
		self.processID = tempr      
		self.parameter_queue = parameter_queue
		self.event = event
		self.signal_main = main_proc
		self.temperature = tempr
		self.swap_interval = swap_interval
		#self.lhood = 0

		self.filename = filename
		self.input = xmlinput  


		self.simtime = simtime
		self.samples = samples
		self.run_nb = run_nb
		self.muted = muted 

		self.font = 9
		self.width = 1 

		self.vec_parameters = np.asarray(vec_parameters)
		self.minlimits_vec = np.asarray(minlimits_vec)
		self.maxlimits_vec = np.asarray(maxlimits_vec)
		self.stepratio_vec = np.asarray(stepratio_vec)

		self.realvalues_vec = np.asarray(realvalues_vec) # true values of free parameters for comparision. Note this will not be avialable in real world application		 

		self.check_likehood_sed =  check_likelihood_sed

		self.real_erodep_pts = real_erodep_pts
		self.real_erodep =  real_erodep  # this is 3D eroddep - this will barely be used in likelihood - so there is no need for it
		self.erodep_coords = erodep_coords
		self.real_elev = real_elev

		self.eta_stepratio =0


		self.burn_in = burn_in

		self.sim_interval = sim_interval

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

	def run_badlands(self, rain, erodibility, m, n):

		model = badlandsModel()

		# Load the XmL input file
		model.load_xml(str(self.run_nb), self.input, muted=self.muted)

		# Adjust erodibility based on given parameter
		model.input.SPLero = erodibility
		model.flow.erodibility.fill(erodibility)

		# Adjust precipitation values based on given parameter
		model.force.rainVal[:] = rain

		# Adjust m and n values
		model.input.SPLm = m
		model.input.SPLn = n

		elev_vec = collections.OrderedDict()
		erodep_vec = collections.OrderedDict()
		erodep_pts_vec = collections.OrderedDict()

		for x in range(len(self.sim_interval)):
			self.simtime = self.sim_interval[x]

			model.run_to_time(self.simtime, muted=self.muted)

			#elev, erodep = self.interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)
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


	def likelihood_func(self,input_vector,  tausq, tau_erodep , tau_erodep_pts):
		pred_elev_vec, pred_erodep_vec, pred_erodep_pts_vec = self.run_badlands(input_vector[0], input_vector[1], input_vector[2],input_vector[3])

		likelihood_elev = - 0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(pred_elev_vec[self.simtime] - self.real_elev) / tausq

		if self.check_likehood_sed  == True:
			#likelihood_erodep  = -0.5 * np.log(2* math.pi * tausq_erodep) - 0.5 * np.square(pred_erodep_vec[self.simtime] - real_erodep) / tausq_erodep		# In case erosion dep is full grid
			likelihood_erodep_pts = -0.5 * np.log(2 * math.pi * tau_erodep_pts) - 0.5 * np.square(pred_erodep_pts_vec[self.simtime] - self.real_erodep_pts) / tau_erodep_pts # only considers point or core of erodep
			likelihood = np.sum(likelihood_elev) + np.sum(likelihood_erodep_pts)

		else:
			likelihood = np.sum(likelihood_elev)


		return [likelihood *(1.0/self.temperature), pred_elev_vec, pred_erodep_vec, pred_erodep_pts_vec]

	def run(self):

		#note this is a chain that is distributed to many cores. The chain is also known as Replica in Parallel Tempering




		samples = self.samples

		count_list = [] 
 

		stepsize_vec = np.zeros(self.maxlimits_vec.size)
		span = (self.maxlimits_vec-self.minlimits_vec) 


		for i in range(stepsize_vec.size): # calculate the step size of each of the parameters
			stepsize_vec[i] = self.stepratio_vec[i] * span[i]

		print(stepsize_vec, 'stesize_vec')




		v_proposal = self.vec_parameters # initial values of the parameters to be passed to Blackbox model 
		v_current = v_proposal # to give initial value of the chain
 

		#  initial predictions from Blackbox model
		initial_predicted_elev, initial_predicted_erodep, init_pred_erodep_pts_vec = self.run_badlands(v_current[0], v_current[1], v_current[2], v_current[3])
	 	
	 	#------------------------------------------
	 	#calculate eta - tau that are parameters that add noise to the predictions. Note each element of prediction (elev, seddep) has a separate eta- tau. 
	 	#reason eta is used is to resolve numberical instablity by using the log scale. so actually both eta and tau represent one parameter that is pertubed by random-walk
	 	
	 	# we first get the estimate for initual values of eta and then calcualte the tau by exp 

	 	eta_elev = np.log(np.var(initial_predicted_elev[self.simtime] - self.real_elev))
		eta_erodep = np.log(np.var(initial_predicted_erodep[self.simtime] - self.real_erodep))
		eta_erodep_pts = np.log(np.var(init_pred_erodep_pts_vec[self.simtime] - self.real_erodep_pts))

		tau_elev = np.exp(eta_elev)
		tau_erodep = np.exp(eta_erodep)
		tau_erodep_pts = np.exp(eta_erodep_pts)
		#-----------------------------------------------------------------

		#based on the initial values of eta and tau, estimate the step size of eta for each element (seddep, elev)



		print(eta_elev, 'eta', tau_elev , ' tau')

		step_eta_elev = np.abs(eta_elev * self.eta_stepratio)
		step_eta_erodep = np.abs(eta_erodep * self.eta_stepratio)
		step_eta_erodep_pts = np.abs(eta_erodep_pts * self.eta_stepratio)

		print('eta_erodep_pts = ', eta_erodep_pts, 'step_eta_erodep_pts', step_eta_erodep_pts)

		#----------------------------------------------------------------------------

 
		#calc initial likelihood with initial parameters
 
		[likelihood, predicted_elev, pred_erodep, pred_erodep_pts] = self.likelihood_func(v_current,  tau_elev, tau_erodep, tau_erodep_pts)

		print('\tinitial likelihood:', likelihood)

		#---------------------------------------
		#now, create memory to save all the accepted tau proposals

		pos_tau = np.zeros(samples)
		pos_tau_erodep = np.zeros(samples)
		pos_tau_erodep_pts = np.zeros(samples ) 

		likeh_list = np.zeros((samples,2)) # one for posterior of likelihood and the other for all proposed likelihood

		count_list.append(0) # just to count number of accepted for each chain (replica)
		accept_list = np.zeros(samples)
		

		#---------------------------------------
		#now, create memory to save all the accepted tau proposals


		prev_accepted_elev = deepcopy(predicted_elev)
		prev_acpt_erodep = deepcopy(pred_erodep)
		prev_acpt_erodep_pts = deepcopy(pred_erodep_pts) 


		sum_elev = deepcopy(predicted_elev)
		sum_erodep = deepcopy(pred_erodep)  # Creating storage for data
		sum_erodep_pts = deepcopy(pred_erodep_pts)

 

		print('time to change')

		#---------------------------------------
		#now, create memory to save all the accepted   proposals of rain, erod, etc etc, plus likelihood


		pos_param = np.zeros((samples,v_current.size)) 

		start = time.time() 

		num_accepted = 0

		num_div = 0 

		burnsamples = int(samples*self.burn_in)
		#save

		with file(('%s/description.txt' % (self.filename)),'a') as outfile:
			outfile.write('\n\samples: {0}'.format(self.samples))
			outfile.write('\n\tstep_vec: {0}'.format(stepsize_vec))  
			outfile.write('\n\tstep_eta: {0}'.format(step_eta_elev))
			outfile.write('\n\tInitial_proposed_vec: {0}'.format(v_proposal))  


		
		for i in range(samples-1):

			print (self.temperature, ' - Sample : ', i)

			# Update by perturbing all the  parameters via "random-walk" sampler and check limits
			v_proposal =  np.random.normal(v_current,stepsize_vec)

			for j in range(v_current.size):
				if v_proposal[j] > self.maxlimits_vec[j]:
					v_proposal[j] = v_current[j]
				elif v_proposal[j] < self.minlimits_vec[j]:
					v_proposal[j] = v_current[j]

			print(v_proposal) 


			eta_elev_pro = np.random.normal(eta_elev,  step_eta_elev ) 

			eta_erodep_pro =  np.random.normal(eta_erodep,  step_eta_erodep ) 

			eta_erodep_pts_pro =  np.random.normal(eta_erodep_pts, step_eta_erodep_pts ) 


			# Passing paramters to calculate likelihood and rmse with new tau
			[likelihood_proposal, predicted_elev, predicted_erodep,pred_erodep_pts] = self.likelihood_func(v_proposal, math.exp(eta_elev_pro), math.exp(eta_erodep_pro)
,  math.exp(eta_erodep_pts_pro))
 
 
			# Difference in likelihood from previous accepted proposal
			diff_likelihood = likelihood_proposal - likelihood

			try:
				mh_prob = min(1, math.exp(diff_likelihood))
			except OverflowError as e:
				mh_prob = 1

			u = random.uniform(0,1)

			



			accept_list[i+1] = num_accepted

			likeh_list[i+1,0] = likelihood_proposal


			print((i % self.swap_interval), i,  self.swap_interval, ' mod swap')



			if u < mh_prob: # Accept sample
				print (v_proposal,   i,likelihood_proposal, self.temperature, num_accepted, ' is accepted - rain, erod, step rain, step erod, likh')
				count_list.append(i)            # Append sample number to accepted list
				likelihood = likelihood_proposal
				eta_elev = eta_elev_pro
				eta_erodep = eta_erodep_pro
				eta_erodep_pts = eta_erodep_pts_pro

				v_current = v_proposal
  
				pos_param[i+1,:] = v_proposal

				likeh_list[i + 1,1]=likelihood 

				pos_tau[i + 1] = math.exp(eta_elev_pro) # not so important to output and visualize
				pos_tau_erodep[i + 1] = math.exp(eta_erodep_pro)  # not so important to output and visualize
				pos_tau_erodep_pts[i + 1] = math.exp(eta_erodep_pts_pro)  # not so important to output and visualize


				num_accepted = num_accepted + 1


				prev_accepted_elev.update(predicted_elev)

				if i>burnsamples:
					for k, v in prev_accepted_elev.items():
						sum_elev[k] += v

					for k, v in pred_erodep.items():
						sum_erodep[k] += v

					for k, v in pred_erodep_pts.items():
						sum_erodep_pts[k] += v

					num_div += 1

			else: # Reject sample
				pos_tau[i + 1] = pos_tau[i,]  # not so important to output and visualize 
				pos_tau_erodep[i + 1] = pos_tau_erodep[i,]  # not so important to output and visualize
				pos_tau_erodep_pts[i + 1] = pos_tau_erodep_pts[i,]  # not so important to output and visualize

				likeh_list[i + 1, 1]=likeh_list[i,1] 

				pos_param[i+1,:] = pos_param[i,:]
 

				#print(v_proposal, i,likelihood_proposal, self.temperature, num_accepted,' rejected - rain, erod')

				if i>burnsamples:

					print('saving sum ele')

					for k, v in prev_accepted_elev.items():
						sum_elev[k] += v

					for k, v in prev_acpt_erodep.items():
						sum_erodep[k] += v

					for k, v in prev_acpt_erodep_pts.items():
						sum_erodep_pts[k] += v

					num_div += 1
 

			if ( i % self.swap_interval == 0 ):


				print(' CHECK if can SWAP -----------------------------------------------------------------------------------------------------> ')

				if i> burnsamples:
					hist, bin_edges = np.histogram(pos_param[burnsamples:i,0], density=True)
					plt.hist(pos_param[burnsamples:i,0], bins='auto')  # arguments are passed to np.histogram
					plt.title("Parameter 1 Histogram")

					file_name = self.filename + '/posterior/pos_parameters/hist_current' + str(self.temperature)
					plt.savefig(file_name+'_0.png')
					plt.clf()

					np.savetxt(file_name+'.txt',  pos_param[ :i,:] ,  fmt='%1.9f')

					hist, bin_edges = np.histogram(pos_param[burnsamples:i,1], density=True)
					plt.hist(pos_param[burnsamples:i,1], bins='auto')  # arguments are passed to np.histogram
					plt.title("Parameter 2 Histogram")
 
					plt.savefig(file_name + '_1.png')
					plt.clf()

 
				others = np.asarray([eta_elev, eta_erodep_pts, likelihood])
				param = np.concatenate([v_current,others])     
			 
				# paramater placed in queue for swapping between chains
				self.parameter_queue.put(param)
				
				
				#signal main process to start and start waiting for signal for main
				self.signal_main.set()              
				self.event.wait()
				

				# retrieve parametsrs fom ques if it has been swapped
				if not self.parameter_queue.empty() : 
					try:
						result =  self.parameter_queue.get()

						print(result, ' res back')
						
						v_current= result[0:v_current.size]   
						eta_elev = result[v_current.size]                             # %%%%%%%%%%%%%%%%%%%   UPDATE when Region Rainfall with erod vec changes
						eta_erodep_pts = result[v_current.size+1] 
						likelihood = result[v_current.size+2]

					except:
						print ('error')

		
		end = time.time()
		total_time = end - start 


		print ('Time elapsed:', total_time)
		accepted_count =  len(count_list)
		print (accepted_count, ' number accepted')
		print (len(count_list) / (samples * 0.01), '% was accepted')
		accept_ratio = accepted_count / (samples * 1.0) * 100



		#--------------------------------------------------------------- 

		others = np.asarray([eta_elev, eta_erodep_pts, likelihood])
		param = np.concatenate([v_current,others])   

		self.parameter_queue.put(param)

 
		file_name = self.filename+'/posterior/pos_tau/chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name,pos_tau, fmt='%1.2f')
 
		file_name = self.filename+'/posterior/pos_parameters/chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name,pos_param )

		file_name = self.filename+'/posterior/pos_likelihood/chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name,likeh_list, fmt='%1.2f')
 
		file_name = self.filename + '/posterior/accept_list/chain_' + str(self.temperature) + '_accept.txt'
		np.savetxt(file_name, [accept_ratio], fmt='%1.2f')

		file_name = self.filename + '/posterior/accept_list/chain_' + str(self.temperature) + '.txt'
		np.savetxt(file_name, accept_list, fmt='%1.2f')


		for k, v in sum_elev.items():
			sum_elev[k] = np.divide(sum_elev[k], num_div)
			mean_pred_elevation = sum_elev[k]

			sum_erodep_pts[k] = np.divide(sum_erodep_pts[k], num_div)
			mean_pred_erodep_pnts = sum_erodep_pts[k]

			file_name = self.filename + '/posterior/predicted_topo/chain_' + str(k) + '_' + str(self.temperature) + '.txt'
			np.savetxt(file_name, mean_pred_elevation, fmt='%.2f')

			file_name = self.filename + '/posterior/predicted_erodep/chain_' + str(k) + '_' + str(self.temperature) + '.txt'
			np.savetxt(file_name, mean_pred_erodep_pnts, fmt='%.2f')

		# signal main process to resume
		self.signal_main.set()


class ParallelTempering:

	def __init__(self, num_chains, maxtemp,NumSample,swap_interval, fname, realvalues_vec, num_param ):


		self.swap_interval = swap_interval
		self.folder = fname
		self.maxtemp = maxtemp
		self.num_chains = num_chains
		self.chains = []
		self.tempratures = []
		self.NumSamples = int(NumSample/self.num_chains)
		self.sub_sample_size = max(1, int( 0.05* self.NumSamples))


		self.erodep_pts  = []
		self.real_elev = [] 

		self.num_param = num_param


		self.realvalues_vec = np.asarray(realvalues_vec)
		
		# create queues for transfer of parameters between process chain
		self.chain_parameters = [multiprocessing.Queue() for i in range(0, self.num_chains) ]

		# two ways events are used to synchronize chains
		self.event = [multiprocessing.Event() for i in range (self.num_chains)]
		self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]

	
	# assigin tempratures dynamically   
	def assign_temptarures(self):
		tmpr_rate = (self.maxtemp /self.num_chains)
		temp = 1
		for i in xrange(0, self.num_chains):            
			self.tempratures.append(temp)
			temp += tmpr_rate
			print(self.tempratures[i])
			
	
	# Create the chains.. Each chain gets its own temprature
	def initialize_chains (self, vec_parameters, minlimits_vec, maxlimits_vec, stepratio_vec,   check_likelihood_sed, sim_interval, muted, simtime,   real_elev,   real_erodep, real_erodep_pts, erodep_coords, filename, xmlinput,   run_nb_str, burn_in):
 		self.burn_in = burn_in

		self.sim_interval = sim_interval

		self.erodep_pts  = real_erodep_pts
		self.real_elev = real_elev
		self.vec_parameters = np.asarray(vec_parameters)


		self.assign_temptarures()
		for i in xrange(0, self.num_chains):
			self.chains.append(ptReplica( vec_parameters, minlimits_vec, maxlimits_vec, stepratio_vec, self.realvalues_vec, check_likelihood_sed ,self.swap_interval, sim_interval, muted, simtime, self.NumSamples, real_elev,  real_erodep, real_erodep_pts, erodep_coords, filename, xmlinput,  run_nb_str,self.tempratures[i], self.chain_parameters[i], self.event[i], self.wait_chain[i],burn_in))
		 	
			
	

	def run_chains (self ):
		
		# only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
		swap_proposal = np.ones(self.num_chains-1) 
		
		# create parameter holders for paramaters that will be swapped
		replica_param = np.zeros((self.num_chains, self.num_param)) 
		replica_eta_elev = np.zeros(self.num_chains)
		replica_eta_erodep = np.zeros(self.num_chains)
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



		flag_running = True 

		
		while flag_running:          

			#-------------------------------------------------------------------------------------
			# wait for chains to complete one pass through the samples
			#-------------------------------------------------------------------------------------

			for j in range(0,self.num_chains): 
				#print (j, ' - waiting')
				self.wait_chain[j].wait()
			

			
			#-------------------------------------------------------------------------------------
			#get info from chains
			#-------------------------------------------------------------------------------------
			
			for j in range(0,self.num_chains): 
				if self.chain_parameters[j].empty() is False :
					result =  self.chain_parameters[j].get()
					replica_param[j,:] = result[0:self.num_param]  
					replica_eta_elev[j] = result[self.num_param] 
					replica_eta_erodep[j] = result[self.num_param+1]
					lhood[j] = result[self.num_param+2]
 
 

			# create swapping proposals between adjacent chains
			for k in range(0, self.num_chains-1): 
				swap_proposal[k]=  (lhood[k]/[1 if lhood[k+1] == 0 else lhood[k+1]])*(1/self.tempratures[k] * 1/self.tempratures[k+1])

			#print(' before  swap_proposal  --------------------------------------+++++++++++++++++++++++=-')

			for l in range( self.num_chains-1, 0, -1):
				#u = 1
				u = random.uniform(0, 1)
				swap_prob = swap_proposal[l-1]



				if u < swap_prob : 

					number_exchange[l] = number_exchange[l] +1  

					others = np.asarray([replica_eta_elev[l-1], replica_eta_erodep[l-1], lhood[l-1] ]  ) 
					para = np.concatenate([replica_param[l-1,:],others])   
 
				   
					self.chain_parameters[l].put(para) 

					others = np.asarray([replica_eta_elev[l], replica_eta_erodep[l], lhood[l] ] )
					param = np.concatenate([replica_param[l,:],others])
 
					self.chain_parameters[l-1].put(param)
					
				else:


					others = np.asarray([replica_eta_elev[l-1], replica_eta_erodep[l-1], lhood[l-1] ])
					para = np.concatenate([replica_param[l-1,:],others]) 
 
				   
					self.chain_parameters[l-1].put(para) 

					others = np.asarray([replica_eta_elev[l], replica_eta_erodep[l], lhood[l]  ])
					param = np.concatenate([replica_param[l,:],others])
 
					self.chain_parameters[l].put(param)


			#-------------------------------------------------------------------------------------
			# resume suspended process
			#-------------------------------------------------------------------------------------
			for k in range (self.num_chains):
					self.event[k].set()
								

			#-------------------------------------------------------------------------------------
			#check if all chains have completed runing
			#-------------------------------------------------------------------------------------
			count = 0
			for i in range(self.num_chains):
				if self.chains[i].is_alive() is False:
					count+=1
					while self.chain_parameters[i].empty() is False:
						dummy = self.chain_parameters[i].get()

			if count == self.num_chains :
				flag_running = False
			

		#-------------------------------------------------------------------------------------
		#wait for all processes to jin the main process
		#-------------------------------------------------------------------------------------     
		for j in range(0,self.num_chains): 
			self.chains[j].join()

		print(number_exchange, 'num_exchange, process ended')


		pos_param, likelihood_rep, accept_list, pred_topo, combined_erodep, accept = self.show_results('chain_')

		print(pos_param, 'pos +++')

		for s in range(self.num_param): 
			self.plot_figure(pos_param[s,:], 'pos_distri_'+str(s),   self.realvalues_vec[s])
		 

 

 
		for i in range(self.sim_interval.size):

			self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=pred_topo[i,:,:], title='Predicted Topography', time_frame=self.sim_interval[i])

		

		#-------------------------------------------------------------------------------------
		# recover posterior chains and merge into a single posterior chain
		#-------------------------------------------------------------------------------------


		
		return (pos_param,likelihood_rep, accept_list,combined_erodep)


	# Merge different MCMC chains y stacking them on top of each other
	def show_results(self, filename):

		burnin = int(self.NumSamples * self.burn_in)
		pos_param = np.zeros((self.num_chains, self.NumSamples - burnin, self.num_param))
		#pos_erod = np.zeros((self.num_chains, self.NumSamples - burnin))
		likehood_rep = np.zeros((self.num_chains, self.NumSamples - burnin, 2 )) # index 1 for likelihood posterior and index 0 for Likelihood proposals. Note all likilihood proposals plotted only
		accept_percent = np.zeros((self.num_chains, 1))

		accept_list = np.zeros((self.num_chains, self.NumSamples )) 

		topo  = self.real_elev


		replica_topo = np.zeros((self.sim_interval.size, self.num_chains, topo.shape[0], topo.shape[1])) #3D
		combined_topo = np.zeros(( self.sim_interval.size, topo.shape[0], topo.shape[1]))

		replica_erodep_pts = np.zeros(( self.num_chains, self.erodep_pts.shape[0] ))
		combined_erodep = np.zeros((self.erodep_pts.shape[0] ))
 

		for i in range(self.num_chains):
			file_name = self.folder + '/posterior/pos_parameters/'+filename + str(self.tempratures[i]) + '.txt'
			dat = np.loadtxt(file_name)

			pos_param[i, :, :] = dat[burnin:,:]


			file_name = self.folder + '/posterior/pos_likelihood/'+filename + str(self.tempratures[i]) + '.txt'
			dat = np.loadtxt(file_name)

			likehood_rep[i, :] = dat[burnin:]

			file_name = self.folder + '/posterior/accept_list/' + filename + str(self.tempratures[i]) + '.txt'
			dat = np.loadtxt(file_name)

			accept_list[i, :] = dat


			file_name = self.folder + '/posterior/accept_list/' + filename + str(self.tempratures[i]) + '_accept.txt'
			dat = np.loadtxt(file_name)

			accept_percent[i, :] = dat

			#print('printed pos for chain', i)

			for j in range(self.sim_interval.size):

				file_name = self.folder+'/posterior/predicted_topo/chain_'+str(self.sim_interval[j])+'_'+ str(self.tempratures[i])+ '.txt'
				dat_topo = np.loadtxt(file_name)
				replica_topo[j,i,:,:] = dat_topo



			file_name = self.folder + '/posterior/predicted_erodep/chain_' + str(self.sim_interval[-1]) + '_' + str(self.tempratures[i]) + '.txt' # access last sed
			data_erodep = np.loadtxt(file_name)
			print(data_erodep)
				 
			replica_erodep_pts[i, :] = data_erodep

 

		posterior = pos_param.transpose(2,0,1).reshape(self.num_param,-1)

		likelihood_vec = likehood_rep.transpose(2,0,1).reshape(2,-1)


		#print(pos_param)

		#print(x)
 
 
			 

 

		for j in range(self.sim_interval.size):
			for i in range(self.num_chains):
				combined_topo[j,:,:] += replica_topo[j,i,:,:]  
			combined_topo[j,:,:] = combined_topo[j,:,:]/self.num_chains

			#combined_erodep += replica_erodep_pts[j,:]

		combined_erodep = np.mean(replica_erodep_pts, axis = 0)

		print(combined_erodep, 'combined_erodepm') 

		#likehood_rep = np.reshape(likehood_rep, (self.num_chains * (self.NumSamples - burnin)))

		accept = np.sum(accept_percent)/self.num_chains




		np.savetxt(self.folder + '/pos_param.txt', posterior.T)
 

		np.savetxt(self.folder + '/likelihood.txt', likelihood_vec.T, fmt='%1.5f')

		np.savetxt(self.folder + '/accept_list.txt', accept_list, fmt='%1.2f')


		np.savetxt(self.folder + '/acceptpercent.txt', [accept], fmt='%1.2f')

		return posterior, likelihood_vec.T, accept_list, combined_topo, combined_erodep, accept

	def plot_figure(self, list, title, real_value): 

		list_points =  list

		fname = self.folder
		width = 9 

		font = 9

		fig = plt.figure(figsize=(10, 12))
		ax = fig.add_subplot(111)
 

		slen = np.arange(0,len(list),1) 
		 
		fig = plt.figure(figsize=(10,12))
		ax = fig.add_subplot(111)
		ax.spines['top'].set_color('none')
		ax.spines['bottom'].set_color('none')
		ax.spines['left'].set_color('none')
		ax.spines['right'].set_color('none')
		ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
		ax.set_title(' Posterior distribution', fontsize=  font+2)#, y=1.02)
	
		ax1 = fig.add_subplot(211)
		#ax1.set_facecolor('#f2f2f3')
		# ax1.set_axis_bgcolor("white")
		n, rainbins, patches = ax1.hist(list_points,   alpha=0.5, facecolor='sandybrown', normed=False)	
		ax1.axvline(real_value)
		ax1.grid(True)
		ax1.set_ylabel('Frequency',size= font+1)
		ax1.set_xlabel('Parameter values', size= font+1)
	
		ax2 = fig.add_subplot(212)

		list_points = np.asarray(np.split(list_points,  self.num_chains ))
 


		#slen = np.arange(0,list_points.shape[0],1) 

		ax2.set_facecolor('#f2f2f3') 
		ax2.plot( list_points.T , label=None)
		ax2.set_title(r'Trace plot',size= font+2)
		ax2.set_xlabel('Samples',size= font+1)
		ax2.set_ylabel('Parameter values', size= font+1) 

		fig.tight_layout()
		fig.subplots_adjust(top=0.88)
		 
 
		plt.savefig(fname + '/' + title  + '_pos_.png', bbox_inches='tight', dpi=300, transparent=False)
		plt.clf()

 


		'''list_points = np.asarray(np.split(list_points,  self.num_chains ))

		plt.plot(list_points.T)  
		plt.xlabel('Samples')
		plt.ylabel(' Parameter values')
		plt.title("Parameter trace-plot")

		plt.savefig(fname + '/' + title  + '_trace_.png', bbox_inches='tight', dpi=300, transparent=False)
		plt.clf()'''


	def viewGrid(self, width=1000, height=1000, zmin=None, zmax=None, zData=None, title='Predicted Topography', time_frame=None):

		if zmin == None:
			zmin =  zData.min()

		if zmax == None:
			zmax =  zData.max()

		data = Data([Surface(x=zData.shape[0], y=zData.shape[1], z=zData, colorscale='YIGnBu')])

		layout = Layout(title='Predicted Topography ' , autosize=True, width=width, height=height,scene=Scene(
					zaxis=ZAxis(range=[zmin, zmax], autorange=False, nticks=10, gridcolor='rgb(255, 255, 255)',
								gridwidth=2, zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2),
					xaxis=XAxis(nticks=10, gridcolor='rgb(255, 255, 255)', gridwidth=2,
								zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2),
					yaxis=YAxis(nticks=10, gridcolor='rgb(255, 255, 255)', gridwidth=2,
								zerolinecolor='rgb(255, 255, 255)', zerolinewidth=2),
					bgcolor="rgb(244, 244, 248)"
				)
			)

		fig = Figure(data=data, layout=layout)

		graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename= self.folder +'/pred_timeframe_'+str(time_frame)+ '_.html', validate=False)


		return

def make_directory (directory): 
	if not os.path.exists(directory):
		os.makedirs(directory)

def main():

	random.seed(time.time())
	muted = True

	samples = 20000     # total number of samples by all the chains (replicas) in parallel tempering

	run_nb = 0

	#problem = input("Which problem do you want to choose 1. crater-fast, 2. crater  3. etopo-fast 4. etopo 5. island ")
	problem = 1
  
	if problem == 1:
		problemfolder = 'Examples/crater_fast/'
		xmlinput = problemfolder + 'crater.xml'
		print('xmlinput', xmlinput)
		simtime = 15000 

		m = 0.5 # used to be constants  
		n = 1

		real_rain = 1.5
		real_erod = 5.e-5


		likelihood_sediment = True

		maxlimits_vec = [3.0,7.e-5, m, n]  # [rain, erod] this can be made into larger vector, with region based rainfall, or addition of other parameters
		minlimits_vec = [0.0 ,3.e-5, m, n]   # hence, for 4 regions of rain and erod[rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod_reg1, erod_reg2, erod_reg3, erod_reg4 ]
									## hence, for 4 regions of rain and 1 erod, plus other free parameters (p1, p2) [rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod, p1, p2 ]

									#if you want to freeze a parameter, keep max and min limits the same
		vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
		realvalues_vec = [real_rain,real_erod, m, n]
		
		stepsize_ratio  = 0.04 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

		stepratio_vec = [stepsize_ratio, stepsize_ratio, stepsize_ratio, stepsize_ratio]

		num_param = vec_parameters.size

		print(vec_parameters) 

 

 
		erodep_coords = np.array([[60, 60], [72, 66], [85, 73], [90, 75]])  # need to hand pick given your problem

	elif problem == 2:
		problemfolder = 'Examples/crater/'
		xmlinput = problemfolder + 'crater.xml'
		simtime = 50000


		m = 0.5 # used to be  constants - we are now making them free parameters - to test here
		n = 1
		
		likelihood_sediment = True

		maxlimits_vec = [3.0,7.e-5, 2, 2]  # [rain, erod] this can be made into larger vector, with region based rainfall, or addition of other parameters
		minlimits_vec = [0.0,3.e-5, 0, 0]   # hence, for 4 regions of rain and erod[rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod_reg1, erod_reg2, erod_reg3, erod_reg4 ]
									## hence, for 4 regions of rain and 1 erod, plus other free parameters (p1, p2) [rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod, p1, p2 ]
		vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
		realvalues_vec = [1.5,5.e-5, m, n]
		
		stepsize_ratio  = 0.025 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

		stepratio_vec = [stepsize_ratio, stepsize_ratio, stepsize_ratio, stepsize_ratio]

		num_param = vec_parameters.size

		print(vec_parameters) 

 


		erodep_coords = np.array([[60, 60], [72, 66], [85, 73], [90, 75]])  # need to hand pick given your problem

	elif problem == 3:
		problemfolder = 'Examples/etopo_fast/'
		xmlinput = problemfolder + 'etopo.xml'
		simtime = 500000


		m = 0.5 # used to be  constants - we are now making them free parameters - to test here
		n = 1
		real_rain = 1.5
		real_erod = 5.e-6


		likelihood_sediment = True 

		maxlimits_vec = [3,7.e-6, m, n]  # [rain, erod] this can be made into larger vector, with region based rainfall, or addition of other parameters
		minlimits_vec = [0 ,3.e-6, m, n]   # hence, for 4 regions of rain and erod[rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod_reg1, erod_reg2, erod_reg3, erod_reg4 ]
									## hence, for 4 regions of rain and 1 erod, plus other free parameters (p1, p2) [rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod, p1, p2 ]

									#if you want to freeze a parameter, keep max and min limits the same
		vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
		realvalues_vec = [real_rain ,real_erod, m, n]
		
		stepsize_ratio  = 0.05 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

		stepratio_vec = [stepsize_ratio, stepsize_ratio, stepsize_ratio, stepsize_ratio]

		num_param = vec_parameters.size
		print(vec_parameters) 

		erodep_coords = np.array([[42, 10], [39, 8], [75, 51], [59, 13], [40,5], [6,20], [14,66], [4,40],[72,73],[46,64]])  # need to hand pick given your problem



	elif problem == 4:
		problemfolder = 'Examples/etopo/'
		xmlinput = problemfolder + 'etopo.xml'
		simtime = 500000
		rainlimits = [0.0, 3.0]
		erodlimts = [3.e-6, 7.e-6]
		mlimit = [0.4, 0.6]
		nlimit = [0.9, 1.1]
		true_rain = 1.5
		true_erod = 5.e-6

		stepsize_ratio_rain = 0.025
		stepsize_ratio_erod = 0.025
		likelihood_sediment = True

		erodep_coords = np.array([[60, 60], [72, 66], [85, 73], [90, 75]])  # need to hand pick given your problem

	elif problem == 5:
		problemfolder = 'Examples/delta/'
		xmlinput = problemfolder + 'delta.xml'
		simtime = 500000

		stepsize_ratio_rain = 0.025
		stepsize_ratio_erod = 0.025
		likelihood_sediment = True

		erodep_coords = np.array([[60, 60], [72, 66], [85, 73], [90, 75]])  # need to hand pick given your problem
	else:
		print('choose some problem  ')

	datapath = problemfolder + 'data/final_elev.txt'
	final_elev = np.loadtxt(datapath)
	final_erodep = np.loadtxt(problemfolder + 'data/final_erdp.txt')

	final_erodep_pts = np.loadtxt(problemfolder + 'data/final_erdp_pts.txt')

	fname = ""
	run_nb = 0
	while os.path.exists(problemfolder +'results_%s' % (run_nb)):
		run_nb += 1
	if not os.path.exists(problemfolder +'results_%s' % (run_nb)):
		os.makedirs(problemfolder +'results_%s' % (run_nb))
		fname = (problemfolder +'results_%s' % (run_nb))

	#filename = ('trials_vec')
 
	make_directory((fname + '/posterior/pos_tau'))
	make_directory((fname + '/posterior/pos_parameters')) 
	make_directory((fname + '/posterior/predicted_topo'))
	make_directory((fname + '/posterior/pos_likelihood'))
	make_directory((fname + '/posterior/accept_list'))
	make_directory((fname + '/posterior/predicted_erodep'))

	run_nb_str = 'results_' + str(run_nb)

	#-------------------------------------------------------------------------------------
	# Number of chains of MCMC required to be run
	# PT is a multicore implementation must num_chains >= 2
	# Choose a value less than the numbe of core available (avoid context swtiching)
	#-------------------------------------------------------------------------------------
	num_chains = 10


 

	#parameters for Parallel Tempering
	maxtemp = int(num_chains * 5)/2
	swap_interval =   int(0.2 * (samples/num_chains)) #how ofen you swap neighbours
	print(swap_interval, ' swap')

	timer_start = time.time()

	burn_in =0.1

	num_successive_topo = 4

	sim_interval = np.arange(0,  simtime+1, simtime/num_successive_topo) # for generating successive topography
	print(sim_interval)

	#-------------------------------------------------------------------------------------
	#Create A a Patratellel Tempring object instance 
	#-------------------------------------------------------------------------------------
	pt = ParallelTempering(num_chains, maxtemp, samples,swap_interval,fname, realvalues_vec, num_param)
	#-------------------------------------------------------------------------------------
	# intialize the MCMC chains
	#-------------------------------------------------------------------------------------
	pt.initialize_chains( vec_parameters, minlimits_vec, maxlimits_vec, stepratio_vec, likelihood_sediment, sim_interval,  muted, simtime, final_elev, final_erodep, final_erodep_pts, erodep_coords, fname, xmlinput,   run_nb_str, burn_in)
	 


	#-------------------------------------------------------------------------------------
	#run the chains in a sequence in ascending order
	#-------------------------------------------------------------------------------------
	pos_param,likehood_rep, accept_list, combined_erodep = pt.run_chains()
	print('sucessfully sampled')
	#print(pos_rain)
	#print(pos_erouud)

	timer_end = time.time()

	likelihood = likehood_rep[:,0] # just plot proposed likelihood 
 
	likelihood = np.asarray(np.split(likelihood,  num_chains ))
 
 

	plt.plot(likelihood.T)
	plt.savefig( fname+'/likelihood.png')
	plt.clf()
	plt.plot(accept_list.T)
	plt.savefig( fname+'/accept_list.png')
	plt.clf()

	print(combined_erodep) 

	fig, ax = plt.subplots()
	index = np.arange(final_erodep_pts.size)
	bar_width = 0.35
	opacity = 0.8
 
	rects1 = plt.bar(index, final_erodep_pts, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Real')
 
	rects2 = plt.bar(index + bar_width, combined_erodep, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Predicted with uncertainity')
 
	plt.xlabel('Selected Coordinates')
	plt.ylabel('Height in meters')
	plt.title('Erosion Deposition')
	#plt.xticks(index + bar_width, ('(x1,y1)', '(x2,y2)', '(x3,y3)', '(x4,y4)'))
	plt.legend() 
	plt.tight_layout() 
	plt.savefig(fname + '/pos_erodep_pts.png')
	plt.clf()

	print ('time taken  in minutes = ', (timer_end-timer_start)/60)
	np.savetxt(fname+'/time.txt',[ (timer_end-timer_start)/60], fmt='%1.1f'  )
	#stop()
if __name__ == "__main__": main()