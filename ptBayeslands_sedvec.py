

#Main Contributers:   Rohitash Chandra and Ratneel Deo  Email: c.rohitash@gmail.com, deo.ratneel@gmail.com

# Bayeslands II: Parallel tempering for multi-core systems - Badlands


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


from scipy import special


import fnmatch
import shutil
from PIL import Image
from io import StringIO
from cycler import cycler
import os


import matplotlib as mpl
mpl.use('Agg')

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
	def __init__(self, vec_parameters, minlimits_vec, maxlimits_vec, stepratio_vec,   check_likelihood_sed ,  swap_interval, sim_interval, simtime, samples, real_elev,  real_erodep_pts, erodep_coords, filename, xmlinput,  run_nb, tempr, parameter_queue,event , main_proc,   burn_in):
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

		self.font = 9
		self.width = 1 

		self.vec_parameters = np.asarray(vec_parameters)
		self.minlimits_vec = np.asarray(minlimits_vec)
		self.maxlimits_vec = np.asarray(maxlimits_vec)
		self.stepratio_vec = np.asarray(stepratio_vec)

		#self.realvalues_vec = np.asarray(realvalues_vec) # true values of free parameters for comparision. Note this will not be avialable in real world application		 

		self.check_likehood_sed =  check_likelihood_sed

		self.real_erodep_pts = real_erodep_pts
		#self.real_erodep =  real_erodep  # this is 3D eroddep - this will barely be used in likelihood - so there is no need for it
		self.erodep_coords = erodep_coords
		self.real_elev = real_elev
 

		self.runninghisto = True  # if you want to have histograms of the chains during runtime in pos_variables folder NB: this has issues in Artimis


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

	def run_badlands(self, input_vector):

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

		pred_elev_vec, pred_erodep_vec, pred_erodep_pts_vec = self.run_badlands(input_vector )

		tausq = np.sum(np.square(pred_elev_vec[self.simtime] - self.real_elev))/self.real_elev.size 

		tau_erodep_pts =  (np.sum(np.square(pred_erodep_pts_vec[self.simtime] - self.real_erodep_pts)))/self.real_erodep_pts.size
 	
		likelihood_elev = - 0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(pred_elev_vec[self.simtime] - self.real_elev) / tausq 

		
		if self.check_likehood_sed  == True:
			likelihood_erodep_pts = -0.5 * np.log(2 * math.pi * tau_erodep_pts) - 0.5 * np.square(pred_erodep_pts_vec[self.simtime] - self.real_erodep_pts) / tau_erodep_pts # only considers point or core of erodep
			likelihood = np.sum(likelihood_elev) + np.sum(likelihood_erodep_pts)

		else:
			likelihood = np.sum(likelihood_elev)

		return [likelihood *(1.0/self.temperature), pred_elev_vec,   pred_erodep_pts_vec]

 

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
		initial_predicted_elev, initial_predicted_erodep, init_pred_erodep_pts_vec = self.run_badlands(v_current)
		 

		#----------------------------------------------------------------------------

 
		#calc initial likelihood with initial parameters
 
		[likelihood, predicted_elev,  pred_erodep_pts] = self.likelihood_func(v_current )

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
		  # Creating storage for data
		sum_erodep_pts = deepcopy(pred_erodep_pts)



 

		print('time to change')


		burnsamples = int(samples*self.burn_in)

		#---------------------------------------
		#now, create memory to save all the accepted   proposals of rain, erod, etc etc, plus likelihood


		pos_param = np.zeros((samples,v_current.size)) 


		list_yslicepred = np.zeros((samples,self.real_elev.shape[0]))  # slice taken at mid of topography along y axis  
		list_xslicepred = np.zeros((samples,self.real_elev.shape[1])) # slice taken at mid of topography along x axis  
		ymid = int(self.real_elev.shape[1]/2 ) #   cut the slice in the middle 
		xmid = int(self.real_elev.shape[0]/2)

		list_erodep  = np.zeros((samples,pred_erodep_pts[self.simtime].size))

		list_erodep_time  = np.zeros((samples , self.sim_interval.size , pred_erodep_pts[self.simtime].size))



		start = time.time() 

		num_accepted = 0

		num_div = 0 

		#save

		with file(('%s/description.txt' % (self.filename)),'a') as outfile:
			outfile.write('\n\samples: {0}'.format(self.samples))
			outfile.write('\n\tstepsize_vec: {0}'.format(stepsize_vec))  
			outfile.write('\n\tstep_ratio_vec: {0}'.format(self.stepratio_vec)) 
			outfile.write('\n\tswap interval: {0}'.format(self.swap_interval))   
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
			# Passing paramters to calculate likelihood and rmse with new tau
			[likelihood_proposal, predicted_elev,  pred_erodep_pts] = self.likelihood_func(v_proposal)

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


			print((i % self.swap_interval), i,  self.swap_interval, ' mod swap')



			if u < mh_prob: # Accept sample
				print (v_proposal,   i,likelihood_proposal, self.temperature, num_accepted, ' is accepted - rain, erod, step rain, step erod, likh')
				count_list.append(i)            # Append sample number to accepted list
				
				likelihood = likelihood_proposal
				 
				v_current = v_proposal
  
				pos_param[i+1,:] = v_current # features rain, erodibility and others  (random walks is only done for this vector)

				likeh_list[i + 1,1]=likelihood  # contains  all proposal liklihood (accepted and rejected ones)

				list_yslicepred[i+1,:] =  final_predtopo[:, ymid] # slice taken at mid of topography along y axis  
				list_xslicepred[i+1,:]=   final_predtopo[xmid, :]  # slice taken at mid of topography along x axis 

				list_erodep[i+1,:] = pred_erodep

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

			
 
 

				if i>burnsamples:

					print('saving sum ele')



					for k, v in prev_accepted_elev.items():
						sum_elev[k] += v
 

					for k, v in prev_acpt_erodep_pts.items():
						sum_erodep_pts[k] += v

					num_div += 1
 

			if ( i % self.swap_interval == 0 ):


				print(' CHECK if can SWAP -----------------------------------------------------------------------------------------------------> ')

				if i> burnsamples and self.runninghisto == True:
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

 
				others = np.asarray([likelihood])
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
						likelihood = result[v_current.size]

					except:
						print ('error')

		 
		accepted_count =  len(count_list) 
		accept_ratio = accepted_count / (samples * 1.0) * 100



		#--------------------------------------------------------------- 

		others = np.asarray([ likelihood])
		param = np.concatenate([v_current,others])   

		self.parameter_queue.put(param)

  
 
		file_name = self.filename+'/posterior/pos_parameters/chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name,pos_param )


		#file_name = self.filename+'/posterior/predicted_erodep/chain_erodep_'+ str(self.temperature)+ '.txt'
		#np.savetxt(file_name, list_erodep )

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

			#file_name = self.filename + '/posterior/predicted_erodep/chain_' + str(k) + '_' + str(self.temperature) + '.txt'
			#np.savetxt(file_name, mean_pred_erodep_pnts, fmt='%.2f')
 
		self.signal_main.set()


class ParallelTempering:

	def __init__(self, vec_parameters, num_chains, maxtemp,NumSample,swap_interval, fname, realvalues_vec, num_param,  real_elev, erodep_pts, erodep_coords, simtime, siminterval, resolu_factor, run_nb, inputxml):


		self.swap_interval = swap_interval
		self.folder = fname
		self.maxtemp = maxtemp
		self.num_chains = num_chains
		self.chains = []
		self.tempratures = []
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
			
	 
	def initialize_chains (self,     minlimits_vec, maxlimits_vec, stepratio_vec,  check_likelihood_sed,   burn_in):
		self.burn_in = burn_in
 
		self.vec_parameters =   np.random.uniform(minlimits_vec, maxlimits_vec) # will begin from diff position in each replica (comment if not needed)


		self.assign_temptarures()
		for i in xrange(0, self.num_chains):
			self.chains.append(ptReplica( self.vec_parameters, minlimits_vec, maxlimits_vec, stepratio_vec,  check_likelihood_sed ,self.swap_interval, self.sim_interval,   self.simtime, self.NumSamples, self.real_elev,   self.real_erodep_pts, self.erodep_coords, self.folder, self.xmlinput,  self.run_nb,self.tempratures[i], self.chain_parameters[i], self.event[i], self.wait_chain[i],burn_in))
			
			
	

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
					lhood[j] = result[self.num_param]
 
 

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

					others = np.asarray([  lhood[l-1] ]  ) 
					para = np.concatenate([replica_param[l-1,:],others])   
 
				   
					self.chain_parameters[l].put(para) 

					others = np.asarray([ lhood[l] ] )
					param = np.concatenate([replica_param[l,:],others])
 
					self.chain_parameters[l-1].put(param)
					
				else:


					others = np.asarray([  lhood[l-1] ])
					para = np.concatenate([replica_param[l-1,:],others]) 
 
				   
					self.chain_parameters[l-1].put(para) 

					others = np.asarray([  lhood[l]  ])
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


		pos_param, likelihood_rep, accept_list, pred_topo,  combined_erodep, accept, pred_topofinal, list_xslice, list_yslice = self.show_results('chain_')
 

 

		self.view_crosssection_uncertainity(list_xslice, list_yslice)

		optimal_para, para_5thperc, para_95thperc = self.get_uncertainity(likelihood_rep, pos_param)
		np.savetxt(self.folder+'/optimal_percentile_para.txt', [optimal_para, para_5thperc, para_95thperc] )
 
		
		



		for s in range(self.num_param):  
			self.plot_figure(pos_param[s,:], 'pos_distri_'+str(s), self.realvalues[s,:]  ) 

 
		for i in range(self.sim_interval.size):

			self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=pred_topo[i,:,:], title='Predicted Topography ', time_frame=self.sim_interval[i],  filename= 'mean')
 
		if self.show_fulluncertainity == True: # this to be used when you need output of the topo predictions - 5th and 95th percentiles

 
			pred_elev5th, pred_eroddep5th, pred_erd_pts5th = self.run_badlands(np.asarray(para_5thperc)) 
		
			self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=pred_elev5th[self.simtime], title='Pred. Topo. - 5th Percentile', time_frame= self.simtime, filename= '5th')
 
			pred_elev95th, pred_eroddep95th, pred_erd_pts95th = self.run_badlands(para_95thperc)
		
			self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=pred_elev95th[self.simtime], title='Pred. Topo. - 95th Percentile', time_frame= self.simtime, filename = '95th')

			pred_elevoptimal, pred_eroddepoptimal, pred_erd_optimal = self.run_badlands(optimal_para)
		
			self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=pred_elevoptimal[self.simtime], title='Pred. Topo. - Optimal', time_frame= self.simtime, filename = 'optimal')

			self.viewGrid(width=1000, height=1000, zmin=None, zmax=None, zData=  self.real_elev , title='Ground truth Topography', time_frame= self.simtime, filename = 'ground_truth')

			
			

		return (pos_param,likelihood_rep, accept_list,   combined_erodep,  pred_topofinal)

	def view_crosssection_uncertainity(self,  list_xslice, list_yslice):

		ymid = int(self.real_elev.shape[1]/2 ) #   cut the slice in the middle 
		xmid = int(self.real_elev.shape[0]/2)

		print(self.real_elev)

		print(self.real_elev.shape, ' shape')

		x_ymid_real = self.real_elev[xmid, :] 

		print( x_ymid_real.shape , ' x_ymid_real shape')
		y_xmid_real = self.real_elev[:, ymid ] 

		 
		x_ymid_mean = list_xslice.mean(axis=1)


		print( x_ymid_mean.shape , ' x_ymid_mean shape')


		x_ymid_5th = np.percentile(list_xslice, 5, axis=1)
		x_ymid_95th= np.percentile(list_xslice, 95, axis=1)
  
		y_xmid_mean = list_yslice.mean(axis=1)
		y_xmid_5th = np.percentile(list_yslice, 5, axis=1)
		y_xmid_95th= np.percentile(list_yslice, 95, axis=1)
 

		x = np.linspace(0, x_ymid_mean.size / self.resolu_factor, num=x_ymid_mean.size) 
		x_ = np.linspace(0, y_xmid_mean.size / self.resolu_factor, num=y_xmid_mean.size)

	  



		plt.plot(x, x_ymid_real, label='ground truth') 
		plt.plot(x, x_ymid_mean, label='pred. (mean)')
		plt.plot(x, x_ymid_5th, label='pred.(5th percen.)')
		plt.plot(x, x_ymid_95th, label='pred.(95th percen.)')
		plt.fill_between(x, x_ymid_5th , x_ymid_95th, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right')
		#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)


		plt.title("Uncertainty in topography prediction (cross section)  ")
		plt.xlabel(' Distance in kilometers  ')
		plt.ylabel(' Height in meters')
		
		plt.savefig(self.folder+'/x_ymid_opt.png') 
		plt.clf()


		plt.plot(x_, y_xmid_real, label='ground truth') 
		plt.plot(x_, y_xmid_mean, label='pred. (mean)') 
		plt.plot(x_, y_xmid_5th, label='pred.(5th percen.)')
		plt.plot(x_, y_xmid_95th, label='pred.(95th percen.)')
		plt.xlabel(' Distance in kilometers ')
		plt.ylabel(' Height in meters')
		
		plt.fill_between(x_, y_xmid_5th , y_xmid_95th, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right')

		plt.title("Uncertainty in topography prediction  (cross section)  ")
		plt.savefig(self.folder+'/y_xmid_opt.png') 
		plt.clf()

		 



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

		replica_erodep_pts = np.zeros(( self.num_chains, self.real_erodep_pts.shape[0] ))


		#list_erodep = np.zeros(( self.num_chains,  self.NumSamples - burnin, self.real_erodep_pts.shape[0] )) # this will become 4D vec when you will consider time variant erodep


		combined_erodep = np.zeros((self.sim_interval.size, self.num_chains, self.NumSamples - burnin, self.real_erodep_pts.shape[0] ))

		timespan_erodep = np.zeros((self.sim_interval.size,  (self.NumSamples - burnin) * self.num_chains, self.real_erodep_pts.shape[0] ))



		#file_name = self.filename+'/posterior/predicted_topo/chain_erodep_'+ str(self.temperature)+ '.txt'
		#np.savetxt(file_name, list_erodep )

 

 


		for i in range(self.num_chains):
			file_name = self.folder + '/posterior/pos_parameters/'+filename + str(self.tempratures[i]) + '.txt'
			dat = np.loadtxt(file_name) 
			pos_param[i, :, :] = dat[burnin:,:]


			#file_name = self.folder + '/posterior/predicted_erodep/chain_erodep_'+  str(self.tempratures[i]) + '.txt'
			#dat = np.loadtxt(file_name) 
			#list_erodep[i, :, :] = dat[burnin:,:]

			file_name = self.folder + '/posterior/predicted_topo/chain_xslice_'+  str(self.tempratures[i]) + '.txt'
			dat = np.loadtxt(file_name) 
			list_xslice[i, :, :] = dat[burnin:,:]


			file_name = self.folder + '/posterior/predicted_topo/chain_yslice_'+  str(self.tempratures[i]) + '.txt'
			dat = np.loadtxt(file_name) 
			list_yslice[i, :, :] = dat[burnin:,:] 

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

				file_name = self.folder+'/posterior/predicted_erodep/chain_'+str(self.sim_interval[j])+'_'+ str(self.tempratures[i])+ '.txt'
				dat_erodep = np.loadtxt(file_name)
				combined_erodep[j,i,:,:] = dat_erodep[burnin:,:]
 
 
 

		posterior = pos_param.transpose(2,0,1).reshape(self.num_param,-1)    
		xslice = list_xslice.transpose(2,0,1).reshape(self.real_elev.shape[1],-1) 
		yslice = list_yslice.transpose(2,0,1).reshape(self.real_elev.shape[0],-1)
 

		likelihood_vec = likehood_rep.transpose(2,0,1).reshape(2,-1) 
 

		for j in range(self.sim_interval.size):
			for i in range(self.num_chains):
				combined_topo[j,:,:] += replica_topo[j,i,:,:]  
			combined_topo[j,:,:] = combined_topo[j,:,:]/self.num_chains

			dx = combined_erodep[j,:,:,:].transpose(2,0,1).reshape(self.real_erodep_pts.shape[0],-1)

			timespan_erodep[j,:,:] = dx.T

			 

 

		#combined_erodep = np.mean(replica_erodep_pts, axis = 0) 

		accept = np.sum(accept_percent)/self.num_chains


		pred_topofinal = combined_topo[-1,:,:] # get the last mean pedicted topo to calculate mean squared error loss
 

		np.savetxt(self.folder + '/pos_param.txt', posterior.T)

		
		np.savetxt(self.folder + '/likelihood.txt', likelihood_vec.T, fmt='%1.5f')

		np.savetxt(self.folder + '/accept_list.txt', accept_list, fmt='%1.2f')


		np.savetxt(self.folder + '/acceptpercent.txt', [accept], fmt='%1.2f')

		return posterior, likelihood_vec.T, accept_list, combined_topo,   timespan_erodep, accept, pred_topofinal, xslice, yslice


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


	def run_badlands(self, input_vector): # this is same method in Replica class - copied here to get error uncertainity in topo pred

		model = badlandsModel()

		# Load the XmL input file
		model.load_xml(str(self.run_nb), self.xmlinput, muted=True)

		print(input_vector, ' input badlands')

		# Adjust erodibility based on given parameter
		model.input.SPLero = input_vector[1] 
		model.flow.erodibility.fill(input_vector[1] )

		# Adjust precipitation values based on given parameter
		model.force.rainVal[:] = input_vector[0] 

		# Adjust m and n values
		model.input.SPLm = input_vector[2] 
		model.input.SPLn = input_vector[3] 

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

		n, rainbins, patches = ax1.hist(list_points,  bins = 20,  alpha=0.5, facecolor='sandybrown', normed=False)	
 
  
		color = ['blue','red', 'pink', 'green', 'purple', 'cyan', 'orange','olive', 'brown', 'black']
		for count, v in enumerate(real_value):
			ax1.axvline(x=v, color='%s' %(color[count]), linestyle='dashed', linewidth=1) # comment when go real value is 

		print(real_value)

		ax1.grid(True)
		ax1.set_ylabel('Frequency',size= font+1)
		ax1.set_xlabel('Parameter values', size= font+1)
	
		ax2 = fig.add_subplot(212)

		list_points = np.asarray(np.split(list_points,  self.num_chains ))
 

 

		ax2.set_facecolor('#f2f2f3') 
		ax2.plot( list_points.T , label=None)
		ax2.set_title(r'Trace plot',size= font+2)
		ax2.set_xlabel('Samples',size= font+1)
		ax2.set_ylabel('Parameter values', size= font+1) 

		fig.tight_layout()
		fig.subplots_adjust(top=0.88)
		 
 
		plt.savefig(fname + '/' + title  + '_pos_.png', bbox_inches='tight', dpi=300, transparent=False)
		plt.clf()
 


	def viewGrid(self, width=1000, height=1000, zmin=None, zmax=None, zData=None, title='Predicted Topography', time_frame=None, filename=None):

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
		graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename= self.folder +  '/pred_plots'+ '/pred_'+filename+'_'+str(time_frame)+ '_.html', validate=False)

		fname = self.folder + '/pred_plots'+'/pred_'+filename+'_'+str(time_frame)+ '_.png' 
		elev_data = np.reshape(zData, zData.shape[0] * zData.shape[1] )   
		hist, bin_edges = np.histogram(elev_data, density=True)
		plt.hist(elev_data, bins='auto')  
		plt.title("Predicted Topography Histogram")  
		plt.xlabel('Height in meters')
		plt.ylabel('Frequency')
		plt.savefig(fname )
		plt.clf()


def mean_sqerror(  pred_erodep, pred_elev,  real_elev,  real_erodep_pts):
		 
		elev = np.sqrt(np.sum(np.square(pred_elev -  real_elev))  / real_elev.size)
		sed =  np.sqrt(np.sum(np.square(pred_erodep -  real_erodep_pts))/ real_erodep_pts.size )

		return elev + sed, sed


def make_directory (directory): 
	if not os.path.exists(directory):
		os.makedirs(directory)


def plot_erodeposition(erodep_mean, erodep_std, groundtruth_erodep_pts, sim_interval, fname):

	fig = plt.figure()
	ax = fig.add_subplot(111)



	index = np.arange(groundtruth_erodep_pts.size) 


	ground_erodepstd = np.zeros(groundtruth_erodep_pts.size) 
	opacity = 0.8
 
	width = 0.35                      # the width of the bars
	
	## the bars
	rects1 = ax.bar(index, erodep_mean, width,
                color='blue',
                yerr=erodep_std,
                error_kw=dict(elinewidth=2,ecolor='red'))

	rects2 = ax.bar(index+width, groundtruth_erodep_pts, width, color='green', 
                yerr=ground_erodepstd,
                error_kw=dict(elinewidth=2,ecolor='red') )

	# axes and labels
	#ax.set_xlim(-width,len(ind)+width)
	#ax.set_ylim(0,0.2)
	ax.set_ylabel('Height in meters')
	ax.set_xlabel('Selected Coordinates')
	ax.set_title('Erosion Deposition')

	xTickMarks = [str(i) for i in range(1,21)]
	ax.set_xticks(index+width)
	xtickNames = ax.set_xticklabels(xTickMarks)
	plt.setp(xtickNames, rotation=0, fontsize=8)

	## add a legend
	plotlegend = ax.legend( (rects1[0], rects2[0]), ('Predicted  ', ' Ground-truth ') )
	 
	plt.savefig(fname +'/pos_erodep_'+str( sim_interval) +'_.png')
	plt.clf()    

	 




def main():

	random.seed(time.time()) 

	samples = 5000 # total number of samples by all the chains (replicas) in parallel tempering

	run_nb = 0

	#problem = input("Which problem do you want to choose 1. crater-fast, 2. crater  3. etopo-fast 4. etopo 5. island ")
	problem = 1

	if problem == 1:
		problemfolder = 'Examples/crater_fast/'
		xmlinput = problemfolder + 'crater.xml'
		print('xmlinput', xmlinput)
		simtime = 15000 

		resolu_factor =  200 # this helps visualize the surface distance in meters 

		true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
		 

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
		 
		
		stepsize_ratio  = 0.02 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

		stepratio_vec = [stepsize_ratio, stepsize_ratio, stepsize_ratio, stepsize_ratio]

		num_param = vec_parameters.size

		print(vec_parameters) 

 

 
		erodep_coords = np.array([[60,60],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69],[79,91],[96,77],[42,49]]) # need to hand pick given your problem

	elif problem == 2:
		problemfolder = 'Examples/crater/'
		xmlinput = problemfolder + 'crater.xml'
		simtime = 50000

		resolu_factor =  200


		true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
		 

		m = 0.5 # used to be constants  
		n = 1

		real_rain = 1.5
		real_erod = 5.e-5



		likelihood_sediment = True

		maxlimits_vec = [3.0,7.e-5, 2, 2]  # [rain, erod] this can be made into larger vector, with region based rainfall, or addition of other parameters
		minlimits_vec = [0.0 ,3.e-5, 0, 0]   # hence, for 4 regions of rain and erod[rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod_reg1, erod_reg2, erod_reg3, erod_reg4 ]
									## hence, for 4 regions of rain and 1 erod, plus other free parameters (p1, p2) [rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod, p1, p2 ]

									#if you want to freeze a parameter, keep max and min limits the same
		vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
		 
		
		stepsize_ratio  = 0.02 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now
		stepratio_vec = [stepsize_ratio, stepsize_ratio, stepsize_ratio, stepsize_ratio]

		num_param = vec_parameters.size

		print(vec_parameters) 


		erodep_coords =  np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]]) # need to hand pick given your problem

	elif problem == 3:
		problemfolder = 'Examples/etopo_fast/'
		xmlinput = problemfolder + 'etopo.xml'
		simtime = 500000
		resolu_factor = 1.5


		true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
		 

		m = 0.5 # used to be constants  
		n = 1

		real_rain = 1.5
		real_erod = 5.e-6


		likelihood_sediment = True

		maxlimits_vec = [3.0,7.e-6, m, n]  # [rain, erod] this can be made into larger vector, with region based rainfall, or addition of other parameters
		minlimits_vec = [0.0 ,3.e-6, m, n]   # hence, for 4 regions of rain and erod[rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod_reg1, erod_reg2, erod_reg3, erod_reg4 ]
									## hence, for 4 regions of rain and 1 erod, plus other free parameters (p1, p2) [rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod, p1, p2 ]

									#if you want to freeze a parameter, keep max and min limits the same
		vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
		 	
		stepsize_ratio  = 0.02 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

		stepratio_vec = [stepsize_ratio, stepsize_ratio, stepsize_ratio, stepsize_ratio]

		num_param = vec_parameters.size

		print(vec_parameters) 

		erodep_coords =  np.array([[42,10],[39,8],[75,51],[59,13],[40,5],[6,20],[14,66],[4,40],[72,73],[46,64]]) # need to hand pick given your problem



	elif problem == 4:
		problemfolder = 'Examples/etopo/'
		xmlinput = problemfolder + 'etopo.xml'
		simtime = 500000
		resolu_factor = 1.5

		true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
		 
		m = 0.5 # used to be constants  
		n = 1

		real_rain = 1.5
		real_erod = 5.e-5


		likelihood_sediment = True

		maxlimits_vec = [3.0,7.e-6, 2, 2]  # [rain, erod] this can be made into larger vector, with region based rainfall, or addition of other parameters
		minlimits_vec = [0.0 ,3.e-6, 0, 0]   # hence, for 4 regions of rain and erod[rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod_reg1, erod_reg2, erod_reg3, erod_reg4 ]
									## hence, for 4 regions of rain and 1 erod, plus other free parameters (p1, p2) [rain_reg1, rain_reg2, rain_reg3, rain_reg4, erod, p1, p2 ]

									#if you want to freeze a parameter, keep max and min limits the same
		vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
	 
 	
		stepsize_ratio  = 0.02 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

		stepratio_vec = [stepsize_ratio, stepsize_ratio, stepsize_ratio, stepsize_ratio]

		num_param = vec_parameters.size

		print(vec_parameters) 
		
		erodep_coords = np.array([[42,10],[39,8],[75,51],[59,13],[40,5],[6,20],[14,66],[4,40],[72,73],[46,64]])  # need to hand pick given your problem


	elif problem == 5:
		problemfolder = 'Examples/delta/'
		xmlinput = problemfolder + 'delta.xml'
		simtime = 500000
		resolu_factor = 1

		# to be done later

	 
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
	num_chains = 10
	swap_ratio = 0.1    #adapt these 
	burn_in =0.1 
	num_successive_topo = 4


 

	#parameters for Parallel Tempering
	maxtemp = int(num_chains * 5)/2
	
	swap_interval =   int(swap_ratio * (samples/num_chains)) #how ofen you swap neighbours
	print(swap_interval, ' swap')

	timer_start = time.time()


	sim_interval = np.arange(0,  simtime+1, simtime/num_successive_topo) # for generating successive topography
	print(sim_interval)

	#-------------------------------------------------------------------------------------
	#Create A a Patratellel Tempring object instance 
	#-------------------------------------------------------------------------------------
 

	pt = ParallelTempering(vec_parameters, num_chains, maxtemp, samples,swap_interval,fname, true_parameter_vec, num_param  ,  groundtruth_elev,  groundtruth_erodep_pts[-1,:], erodep_coords, simtime, sim_interval, resolu_factor, run_nb_str, xmlinput)
	#-------------------------------------------------------------------------------------
	# intialize the MCMC chains
	#-------------------------------------------------------------------------------------
	pt.initialize_chains(    minlimits_vec, maxlimits_vec, stepratio_vec, likelihood_sediment,   burn_in)
	 


	#-------------------------------------------------------------------------------------
	#run the chains in a sequence in ascending order
	#-------------------------------------------------------------------------------------
	pos_param,likehood_rep, accept_list,   combined_erodep, pred_elev  = pt.run_chains()
	print('sucessfully sampled') 

	timer_end = time.time() 
	likelihood = likehood_rep[:,0] # just plot proposed likelihood  
	likelihood = np.asarray(np.split(likelihood,  num_chains ))
 
 

	plt.plot(likelihood.T)
	plt.savefig( fname+'/likelihood.png')
	plt.clf()
	plt.plot(accept_list.T)
	plt.savefig( fname+'/accept_list.png')
	plt.clf()
 

	for i in range(sim_interval.size):
		pos_ed  = combined_erodep[i, :, :] 

		print(pos_ed) 
		erodep_mean = pos_ed.mean(axis=0)  
		erodep_std = pos_ed.std(axis=0) 
		print(erodep_std, ' std')   
		print(erodep_mean, '  mean')
		plot_erodeposition(erodep_mean, erodep_std, groundtruth_erodep_pts[i,:], sim_interval[i], fname) 
		#np.savetxt(fname + '/posterior/predicted_erodep/com_erodep_'+str(sim_interval[i]) +'_.txt', pos_ed)

	pos_ed  = combined_erodep[-1, :, :] # get final one for comparision
	erodep_mean = pos_ed.mean(axis=0)  

	rmse, rmse_sed= mean_sqerror(  erodep_mean, pred_elev,  groundtruth_elev,  groundtruth_erodep_pts[-1,:])
 

 



	
	print ('time taken  in minutes = ', (timer_end-timer_start)/60)
	np.savetxt(fname+'/time_sqerror.txt',[ (timer_end-timer_start)/60,  rmse_sed, rmse], fmt='%1.2f'  )


	 


	#stop()
if __name__ == "__main__": main()
