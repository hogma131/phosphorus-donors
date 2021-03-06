# Classes intended for simulating 31P donors in silicon.
# Initiated by Mark Hogg, 2018

import scipy.constants as const
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import qutip as qu
import numpy as np
import seaborn as sb


class SingleDonor():
	'''
	This is a class for a single 31P donor object.
	'''
	
	# Define some physical constants
	_mu_b = const.physical_constants['Bohr magneton'][0]
	_mu_n = const.physical_constants['nuclear magneton'][0]
	
	# Set up the basis I'm going to use. In this basis spin up is the state [1,0], spin down [0,1]
	# The first element is the electron spin, the second element the nuclear spin.
	upup = qu.tensor(qu.basis(2,0), qu.basis(2,0))
	updn = qu.tensor(qu.basis(2,0), qu.basis(2,1))
	dnup = qu.tensor(qu.basis(2,1), qu.basis(2,0))
	dndn = qu.tensor(qu.basis(2,1), qu.basis(2,1))
	
	# Defining electron and nuclear spin operators
	Sx_operator = qu.tensor(qu.sigmax(), qu.identity(2))
	Sy_operator = qu.tensor(qu.sigmay(), qu.identity(2))
	Sz_operator = qu.tensor(qu.sigmaz(), qu.identity(2))
	Ix_operator = qu.tensor(qu.identity(2), qu.sigmax())
	Iy_operator = qu.tensor(qu.identity(2), qu.sigmay())
	Iz_operator = qu.tensor(qu.identity(2), qu.sigmaz())

		
	def __init__(self, **kwargs):
		'''
		Initialise the single donor object. Electron and nuclear g factors can be specified, otherwise default values 
		are taken.
		Inputs:
			mag_field (optional) - static magnetic field in z direction. Default is 1 Tesla if not specified.
			electron_g_factor (optional) - electron g-factor, value is 2 if not specifiend
			nuclear_g_factor (optional) - nuclear g-factor
			hyperfine_coupling (optional) - hyperfine coupling constant between electron and nuclear spin. Default 
											value is bulk single donor
		'''
		self.b_z = kwargs.get('mag_field', 1.0)
		self._g_factor = kwargs.get('electron_g_factor', 2.0)
		self._ng_factor = kwargs.get('nuclear_g_factor', 2.2632)
		self.hyperfine = kwargs.get('hyperfine_coupling', 2*np.pi*114e6)
		
		self.electron_gyro = self._g_factor*self._mu_b/const.hbar
		self.nuclear_gyro = self._ng_factor*self._mu_n/const.hbar
		self.build_hamiltonian()
		
	def build_hamiltonian(self, **kwargs):
		'''
		Construct static single donor Hamiltonian for given physical parameters
		Inputs:
			mag_field (optional) - static bz field, defaults to object attribute if not specified. Note that this 
									function updates the object static magnetic field attribute if you do input one,
									so the object Hamiltonian changes.
		Todo:
			- Come up with a tidier way of managing static b_z attribute here.
		'''
		self.b_z = kwargs.get('mag_field', self.b_z)
		H_electron_zeeman = 0.5*self.electron_gyro*self.b_z*self.Sz_operator
		H_nuclear_zeeman = 0.5*self.nuclear_gyro*self.b_z*self.Iz_operator
		H_hyperfine = self.hyperfine*(qu.tensor(qu.sigmax(), qu.sigmax()) + qu.tensor(qu.sigmay(), qu.sigmay()) 
				+ qu.tensor(qu.sigmaz(), qu.sigmaz()))
		H = H_electron_zeeman - H_nuclear_zeeman + H_hyperfine
		self.Hamiltonian = H
		
	def plot_spectrum(self, b_start, b_stop, **kwargs):
		'''
		Plot the spectrum for a single phosphorus donor as a function of magnetic field
		Inputs:
			b_start - magnetic field value to start from
			b_stop - magnetic field value to end at
			num_points (optional) - number of points to evaluate at (default is 1000)
		Outputs: 
			None
		Todo:
			- Add plots of projection on to high-field eigenstates?
		'''
		num_points = kwargs.get('num_points', 1000)
		b_sweep = np.linspace(b_start, b_stop, num_points)
		# Pre-allocate numpy array for eigenvalues
		self.eigenvals = np.zeros([len(b_sweep), 4])
		projvecs = np.zeros([len(b_sweep), 4])
		for ind,val in enumerate(b_sweep):
			self.build_hamiltonian(static_mag=val)
			self.eigenvals[ind], temp_vecs = self.Hamiltonian.eigenstates()
			projvecs[ind] = self.project_eigenvecs(temp_vecs)
		plt.figure(1)
		plt.plot(b_sweep, self.eigenvals/(2*np.pi*1e9))
		plt.xlabel('Magnetic field (T)')
		plt.ylabel('Energy (GHz)')
		# plt.figure(2)
		# plt.plot(b_sweep, projvecs[:,0:2], '.')
		# plt.xlabel('Magnetic field (T)')
		# plt.ylabel('Projection')
		# plt.legend(['E down, N up', 'E down, N down'])
		# plt.figure(3)
		# plt.plot(b_sweep, projvecs[:,2:4], '.')
		# plt.xlabel('Magnetic field (T)')
		# plt.ylabel('Projection')
		# plt.legend(['E up, N down', 'E up, N up'])
		plt.show(block=False)
		
	def project_eigenvecs(self, vecs):
		'''
		Helper function to project eigenvectors onto spin up and down states which are eigenvectors 
		in a large magnetic field. So I am projecting each eigenstate onto the state it will become at high
		magnetic field.
		'''
		proj_vector = np.zeros(4)
		proj_vector[:] = np.abs([vecs[0].overlap(self.dnup), vecs[1].overlap(self.dndn), 
		vecs[2].overlap(self.updn), vecs[3].overlap(self.upup)])
		return proj_vector
	
	
	def do_spin_resonance(self, b1=0.01, bz=1.0, psi0=dnup, transition='esr1', detuning=0, **kwargs):
		'''
		Perform an ESR/NMR control operation on the system. The simulation works with the full Hamiltonian of both
		the electron and the nucleus, so to choose whether to do an ESR or NMR experiment you just change the drive
		frequency to match the appropriate resonance. Currently only sine wave drive supported.
		Inputs:
			b1 - amplitude of ac magnetic drive (default 0.01T)
			bz - static mag field (default 1.0T)
			psi0 - initial state (default dnup state)
			transition - select the base transition frequency that you want to drive at. Default is dnup to upup freq
							options are esr1, esr2, nmr1, nmr2
			detuning - detuning from resonance frequency of transition you selected (in Hz). Default is 0
			kwargs:
			tmax (optional) - Simulation time (default is approximately a pi/2 pulse)
			n_steps (optional) - number of steps in simulation time vector. Default is 1000
		Outputs:
			output - Expectation values of electron and nuclear spin operators (ordered as Sx, Sy, Sz, Ix, Iy, Iz)
		Todo:
			- Have solver options as kwargs input
			- Plot expectation values as a function of time
			- Include decoherence channels
		'''
		self.build_hamiltonian(static_mag=bz) 	# Update the static Hamiltonian
		esr1_freq, esr2_freq, nmr1_freq, nmr2_freq = self.calculate_resonance_freqs() # Get the transition frequencies
		# Set ESR drive frequency based on input options
		if transition == 'esr1':
			freq = esr1_freq + detuning		# AC drive frequency, defaults to energy splitting between dnup and upup
			esr_flag = True
		elif transition == 'esr2':
			freq = esr2_freq + detuning
			esr_flag = True
		elif transition == 'nmr1':
			freq = nmr1_freq + detuning
			esr_flag = False
		elif transition == 'nmr2':
			freq = nmr2_freq + detuning
			esr_flag = False
		else:
			raise ValueError('Undefined spin resonance transition - should be esr1, esr2, nmr1 or nmr2')
		
		if esr_flag:
			tmax = kwargs.get('tmax', np.pi/(self.electron_gyro*b1)) 	# Default should be approximately a pi/2 pulse
		else:
			tmax = kwargs.get('tmax', np.pi/(self.nuclear_gyro*b1)) 	# Default should be approximately a pi/2 pulse
		n_steps = kwargs.get('n_steps', 1000)
		tlist = np.linspace(0,tmax,n_steps) 	# Time vector for simulation
		# Operator list to be evaluated at each time step
		e_ops_list = [self.Sx_operator, self.Sy_operator, self.Sz_operator, self.Ix_operator, 
						self.Iy_operator, self.Iz_operator]
		drive_args = {'b1':b1, 'freq':freq, 'phase':0}
		esr_Hamiltonian = 0.5*self.electron_gyro*self.Sx_operator 	# ESR Hamiltonian
		nmr_Hamiltonian = 0.5*self.nuclear_gyro*self.Ix_operator 	# NMR Hamiltonian
		full_H = [self.Hamiltonian, [esr_Hamiltonian+nmr_Hamiltonian, rabi_drive]]
		opts = qu.Options(nsteps=5000)
		output = qu.mesolve(full_H, psi0, tlist, c_ops=[], e_ops=e_ops_list, args=drive_args, options=opts)
		self.plot_trajectories(output.times, np.array(output.expect[0:3]), 'Electron state')
		self.plot_trajectories(output.times, np.array(output.expect[3:6]), 'Nucleus state')
		return output
		
	def plot_trajectories(self, times, trajs, plot_title=''):
		'''
		Plot output of a simulation result on the Bloch sphere
		'''
		f = plt.figure(figsize=(5,5), facecolor='white')
		f.suptitle(plot_title, x=0.2)
		# plt.title(plot_title)
		xp2 = qu.rx(np.pi*0.5)
		yp2 = qu.ry(np.pi*0.5)
		zp2 = qu.rz(np.pi*0.5)
		xpi = qu.sigmax()
		ypi = qu.sigmay()
		zpi = qu.sigmaz()
		up = qu.basis(2,0)
		dn = qu.basis(2,1)
		# ax = f.add_subplot(1, 1, 1, axisbg='red')
		
		# p1 = f.add_subplot(2,2,1)
		# plt.plot(times,trajs[0])
		# legend(['OneX','OneY','OneZ'],loc='best')
		
		# p2 = f.add_subplot(2,2,3)
		# pure = norm(trajs[0],axis=1)
		# plt.plot(times,pure)
		# legend(['Purity'],loc='best')
		# ylim(-0.1,1.1)
		# p3 = f.add_subplot(1,2,2, projection='3d')
		b = qu.Bloch(fig=f)#,axes=p3)
		b.zlabel = [r"$\left|1\right\rangle $",r"$\left|0\right\rangle$"]
		b.xlabel = [r"$ X $",r""]
		b.ylabel = [r"$ Y $",r""]
		b.vector_color = sb.color_palette()
		b.add_states([yp2*up,xp2*xpi*up,up])
		b.point_color = sb.color_palette('dark')[3:4]
		b.add_points(trajs,'l')
		# b.add_points(trajs[0][0].transpose(),'s')
		b.font_size = 30
		b.sphere_color = '000000'
		b.render(f)
		b.show()
		

	def calculate_resonance_freqs(self):
		'''
		Calculate what the ESR and NMR resonance frequencies should be from current Hamiltonian eigenstates
		Inputs: 
			None
		Outputs:
			esr1_freq - ESR transition frequency for dnup to upup (ordering electron, nucleus)
			esr2_freq - ESR transition frequency for dndn to updn
			nmr1_freq - NMR transition frequency for dnup to dndn
			nmr2_freq - NMR transition frequency for updn to upup
		'''
		# Get eigenvalues and eigenvectors of static part of current Hamiltonian
		vals, vecs = self.Hamiltonian.eigenstates()
		esr1_freq = (vals[3] - vals[0])/(2*np.pi)
		esr2_freq = (vals[2] - vals[1])/(2*np.pi)
		nmr1_freq = (vals[1] - vals[0])/(2*np.pi)
		nmr2_freq = (vals[3] - vals[2])/(2*np.pi)
		return (esr1_freq, esr2_freq, nmr1_freq, nmr2_freq)
		
		
def rabi_drive(t, args):
	'''
	Defining the function for the time dependence.
	'''
	return args['b1']*np.cos(2*np.pi*args['freq']*t + args['phase'])
		
def detuning_pulse(t, args):
	'''
	Detuning pulse function for time-dependent Hamiltonian solver. Can use this to simulate e.g. pulsing a gate.
	Just does a linear ramp for now
	'''
	return (args['stop']-args['start'])*t/args['tmax']
	
		
class SingletTriplet():
	'''
	Class defining two coupled spins as in a singlet triplet spin system.
	'''
	
	# Define some physical constants
	_mu_b = const.physical_constants['Bohr magneton'][0]
	_mu_n = const.physical_constants['nuclear magneton'][0]
	
	# Define basis vectors
	S11 = qu.basis(5,0)
	Tm = qu.basis(5,1)
	T0 = qu.basis(5,2)
	Tp = qu.basis(5,3)
	S20 = qu.basis(5,4)
	# Also updn and dnup basis - I believe I have these the correct way round....
	updn = (T0 - S11).unit()
	dnup = (T0 + S11).unit()
	
	def __init__(self, b_z, delta_Bz, tc, alpha, **kwargs):
		'''
		Docstring
		'''
		g_factor = kwargs.get('electron_g_factor', 2)
		ng_factor = kwargs.get('nuclear_g_factor', 2.2632)
		self.hyperfine = kwargs.get('default_hyperfine', 2*np.pi*114e6)
		detuning = kwargs.get('detuning', 0)
		self.dBx = kwargs.get('delta_Bx', 0)
		
		self.electron_gyro = g_factor*self._mu_b/const.hbar
		self.nuclear_gyro = ng_factor*self._mu_n/const.hbar
		
		self.b_z = b_z
		self.dBz = delta_Bz
		self.dBz_freq = self.electron_gyro*self.dBz/(2*np.pi)
		self.dBx_freq = self.electron_gyro*self.dBx/(2*np.pi)
		self.Zeeman = (self.electron_gyro*self.b_z)/(2*np.pi)
		self.tc = tc
		self.alpha = alpha
		
		self.build_hamiltonian(detuning)

	def build_hamiltonian(self, detuning=0):
		'''
		Build the singlet-triplet Hamiltonian
		Inputs:
			detuning (V) - detuning from the charge degeneracy point in volts
		Todo:
			- Check that detuning scaling is correct (factor of 2pi or not)
		'''
		# Convert Zeeman terms to frequency units
		detuning = self.alpha*detuning*const.eV/(const.h)	# Convert from voltage to frequency
		
		# Just directly writing out the 5 by 5 Hamiltonian (in units of angular frequency)
		# Defined in the basis [S11, T-, T0, T+, S20]
		H = qu.Qobj(2*np.pi*np.array([[detuning/2, self.dBx_freq, self.dBz_freq, 0, self.tc], 
										[self.dBx_freq, detuning/2-self.Zeeman, 0,0,self.dBx_freq], 
										[self.dBz_freq, 0, detuning/2,0,0], 
										[0,0,0,detuning/2+self.Zeeman,0], 
										[self.tc,self.dBx_freq,0,0,-detuning/2]]))
		self.Hamiltonian = H
		
		
	def build_full_hamiltonian(self, b_z=1, dBz=0.01, detuning=0, tc1=1e9, tc2=0.5e9, pauli_energy=50e9, alpha=0.1):
		'''
		NOTE - not ready for usage
		Attempting to build the full Hamiltonian, including the triplet (2,0) states.
		Make the full 8 by 8 spin Hamiltonian for singlet and triplet states on both dots
		TODO - fix for new philosphy of fixed parameters
		'''
		# In the basis (S11, T-11, T011, T+11, S02, T-02, T002, T+02)
		Zeeman = (0.5*self.electron_gyro*b_z)/(2*np.pi)
		dBz = 0.5*self.electron_gyro*dBz/(2*np.pi)
		detuning = alpha*detuning*const.eV/(2*np.pi*const.h)
		#print(Zeeman)
		#print(-detuning/2+Zeeman)
		H_array = np.array([[detuning/2, 0, dBz, 0, tc1, 0, 0, 0], [0,detuning/2-Zeeman,0,0,0,tc2,0,0], 
							[dBz,0,detuning/2,0,0,0,tc2,0], [0,0,0,detuning/2+Zeeman,0,0,0,tc2], 
							[tc1,0,0,0,-detuning/2,0,0,0], [0,tc2,0,0,0,-detuning/2-Zeeman+pauli_energy,0,0], 
							[0,0,tc2,0,0,0,-detuning/2+pauli_energy,0], [0,0,0,tc2,0,0,0,-detuning/2+Zeeman+pauli_energy]])
		#print(H_array.shape)
		#print('H shape is %s' % H_array.shape)
		H = qu.Qobj(H_array)
		return H
		
	def project_eigenvecs(self, psi):
		'''
		Project state psi onto the basis [S11, T-, T0, T+, S20] and [updn, dnup]
		'''
		S11_proj = np.abs(psi.overlap(self.S11))**2
		Tm_proj = np.abs(psi.overlap(self.Tm))**2
		T0_proj = np.abs(psi.overlap(self.T0))**2
		Tp_proj = np.abs(psi.overlap(self.Tp))**2
		S20_proj = np.abs(psi.overlap(self.S20))**2
		updn_proj = np.abs(psi.overlap(self.updn))**2
		dnup_proj = np.abs(psi.overlap(self.dnup))**2
		overlap_vec_ST = np.array([S11_proj, Tm_proj, T0_proj, Tp_proj, S20_proj])
		overlap_vec_UD = np.array([updn_proj, dnup_proj])
		return overlap_vec_ST, overlap_vec_UD
		
	def calculate_exchange(self, detuning, tc):
		'''
		Calculate the exchange energy as a function of detuning and tunnel coupling
		'''
		J = detuning/2 + np.sqrt((detuning/2)**2 + tc**2)
		return J
		
	
	def plot_spectrum_vs_detuning(self, start, stop, num_points=1000):
		'''
		'''
		self.current_detuning_sweep = np.linspace(start, stop, num_points)
		# self.eigenvals = np.zeros([num_points, 8]) 	# For when I was playing with full 8 by 8 Hamiltonian
		self.eigenvals = np.zeros([num_points, 5])
		# projvecs = np.zeros([len(b_sweep), 4])
		for ind,val in enumerate(self.current_detuning_sweep):
			self.build_hamiltonian(detuning=val)
			# self.Hamiltonian = self.build_full_hamiltonian(detuning=val, b_z=b_z, dBz=dBz, alpha=alpha)
			self.eigenvals[ind], temp_vecs = self.Hamiltonian.eigenstates()
			# projvecs[ind] = self.project_eigenvecs(temp_vecs)
		plt.figure(1)
		plt.plot(self.current_detuning_sweep, self.eigenvals)
		plt.xlabel('Detuning')
		plt.ylabel('Energy')
		# plt.figure(2)
		# plt.plot(b_sweep, projvecs[:,0:2], '+')
		# plt.xlabel('Magnetic field (T)')
		# plt.ylabel('Projection')
		# plt.legend(['E down, N up', 'E down, N down'])
		# plt.figure(3)
		# plt.plot(b_sweep, projvecs[:,2:4], '+')
		# plt.xlabel('Magnetic field (T)')
		# plt.ylabel('Projection')
		# plt.legend(['E up, N down', 'E up, N up'])
		# plt.ylim([-3.1e11,-3.0e11])
		plt.show(block=False)
		return plt
		
	def do_detuning_pulse(self, start, stop, psi0, ramp_time, num_points=1000, **kwargs):
		'''
		NOTE - this function is currently depreciated - to do a detuning pulse it is better to use the more
		flexible do_pulse_sequence method.
		Run an experiment that pulses the detuning from a start value to a stop value
		'''
		tlist = np.linspace(0, ramp_time, num_points)
		pulse_args = {'start':start, 'stop': stop, 'tmax': ramp_time}
		# tc = kwargs.get('tunnel_coupling', 500e6)
		opts = qu.Options(nsteps=100000)
		# alpha = kwargs.get('alpha', 0.1)
		# b_z = kwargs.get('b_z', 0.1)
		# dBz = kwargs.get('dBz', 0.005)
		self.build_hamiltonian(detuning=start)
		start_hamiltonian = self.Hamiltonian
		conversion = self.alpha*const.eV/const.h		# Convert voltage pulse to frequency units
		detuning_hamiltonian = conversion*qu.Qobj(np.array([[1/2, 0, 0, 0, 0], 
														[0, 1/2, 0,0,0], 
														[0, 0, 1/2,0,0], 
														[0,0,0,1/2,0], 
														[0,0,0,0,-1/2]]))
		full_H = [start_hamiltonian, [detuning_hamiltonian, detuning_pulse]]
		output = qu.mesolve(full_H, psi0, tlist, c_ops=[], e_ops=[], args=pulse_args, options=opts, progress_bar=True)
		return output, tlist
		
	def do_pulse_sequence(self, lePulse, psi0, **kwargs):
		'''
		Run a detuning pulse sequence
		NOTE - some weird compiler errors have been observed using the Qutip Cubic_Spline functionality - don't blame me,
		this runs fine on my computer!
		Inputs:
			lePulse - pulse object as defined by PulseSequence class
			psi0 - initial state before pulse happens
			nsteps (optional) - max number of steps the solver can take. Sometimes needs to be increased for long sims
		'''
		nsteps = kwargs.get('nsteps', 500000)
		opts = qu.Options(nsteps)
		
		self.build_hamiltonian(detuning=0)
		initial_hamiltonian = self.Hamiltonian
		conversion_factor = self.alpha*const.eV/const.h		# Convert voltage pulse to frequency units
		detuning_hamiltonian = 2*np.pi*conversion_factor*qu.Qobj(np.array([[1/2, 0, 0, 0, 0], 
																[0, 1/2, 0,0,0], 
																[0, 0, 1/2,0,0], 
																[0,0,0,1/2,0], 
																[0,0,0,0,-1/2]]))
		# Cubic_Spline interpolation is how Qutip handles arbitrary time dependence as input. Look it up in their docs
		# if you want to know more.
		qutip_time_dependence = qu.Cubic_Spline(lePulse.time_vec[0], lePulse.time_vec[-1], lePulse.simulation_waveform)
		full_H = [initial_hamiltonian, [detuning_hamiltonian, qutip_time_dependence]]
		output = qu.mesolve(full_H, psi0, tlist=lePulse.time_vec, c_ops=[], e_ops=[], options=opts, progress_bar=True)
		return output
		
	def extract_projections_from_sim_data(self, sim_data):
		'''
		Extract projections onto the singlet-triplet and up-down basis for simulation data as returned by Qutip
		mesolve function
		'''
		proj_ST = np.zeros((len(sim_data),5))
		proj_UD = np.zeros((len(sim_data),2))
		for ind,val in enumerate(sim_data):
			proj_ST[ind,:], proj_UD[ind,:] = self.project_eigenvecs(sim_data[ind])
		return proj_ST, proj_UD
		
	def plot_pulse_projections(self, lePulse, proj_ST, proj_UD):
		'''
		'''
		plt.figure(1)
		plt.plot(lePulse.time_vec, proj_ST)#, lePulse.time_vec, lePulse.simulation_waveform)
		plt.legend(['S11 proj', 'T- proj', 'T0 proj', 'T+ proj', 'S20 proj'])
		plt.ylim([-0.05,1.1])
		plt.show(block=False)
		
		plt.figure(2)
		plt.plot(lePulse.time_vec, proj_UD)#, lePulse.time_vec, lePulse.simulation_waveform)
		plt.legend(['UpDn proj', 'DnUp proj'])
		plt.ylim([-0.05,1.1])
		plt.show(block=False)
		
		plt.figure(3)
		plt.plot(lePulse.time_vec, lePulse.simulation_waveform)
		plt.xlabel('time')
		plt.ylabel('Detuning voltage')
		
		
class CoupledDonors(): 

	'''
	Class for a coupled donor system. This isn't ready. Yet......
	'''
	
	S11 = qu.basis(5,0)
	Tm = qu.basis(5,1)
	T0 = qu.basis(5,2)
	Tp = qu.basis(5,3)
	S20 = qu.basis(5,4)
	
	def __init__(self, Donor1, Donor2, b_z=1, dBz=0.01):
		'''
		Initialise coupled donor system
		'''
		self.Hamiltonian = self.build_hamiltonian(Donor1, Donor2)
		
	def build_hamiltonian(Donor1, Donor2):
		'''
		Build the Hamiltonian for coupled donor system
		'''
		H_uncoupled = qu.tensor(Donor1.Hamiltonian, qu.identity(2), qu.identity(2)) + qu.tensor(qu.identity(2), 
								qu.identity(2), Donor2.Hamiltonian)
		
		
		H = qu.Qobj(np.array([[detuning/2, 0, dBz, 0, tc], [0, detuning/2-Zeeman, 0,0,0], [dBz, 0, detuning/2,0,0], 
								[0,0,0,detuning/2+Zeeman,0], [tc,0,0,0,-detuning/2]]))


class PulseSequence():
	'''
	Class to encapsulate a pulsing object which can be used in a time-dependent simulation
	Currently only supports constant voltage levels or linear ramps
	'''
	
	def __init__(self, pulse_vertices, pulse_timings, dt):
		'''
		Inputs:
			pulse_vertices - voltage vertices of the pulse
			pulse_timings - timings of each pulse
			dt - waveform time increment
		Example:
			pulse_vertices = [0.2, 0.2, 0, 0, -0.1]
			pulse_timings = [1.0e-9, 10.0e-9, 6.5e-9, 20.0e-9]
			dt = 0.1e-9
			This would define a waveform that has a constant level at 0.2V for 1ns, then ramps from 0.2 to 0V in 
			10ns, then keeps a constant level at 0V for 6.5ns, then ramps to -0.1V in 20ns.
			The vertices define voltage values, and timings define the amount of time it takes to go between these
			vertices. Length of pulse_timings should be len(pulse_vertices)-1
		'''
		self.pulse_vertices = pulse_vertices
		self.pulse_times = np.cumsum([0] + pulse_timings)
		self.dt = dt
		self.make_waveform()
		
	def make_waveform(self):
		'''
		Make the waveform
		'''
		self.le_interpolator = interp.interp1d(self.pulse_times, self.pulse_vertices)
		self.time_vec = np.arange(self.pulse_times[0], self.pulse_times[-1], self.dt)
		self.simulation_waveform = self.le_interpolator(self.time_vec)
		
	def resample_waveform(self, new_dt):
		'''
		If you want to resample the same waveform at a different rate, you can do so.
		'''
		self.dt = new_dt
		self.make_waveform()
		
	def plot_waveform(self):
		'''
		Plot the current waveform
		'''
		plt.plot(self.time_vec, self.simulation_waveform, 'o')
		plt.show(block=False)
		
		
class PulseExperiment():
	'''
	Pulse experiment class to hold simulation data etc for a pulse sequence
	After writing this I'm starting to think it is unnecessarily complicated... Probably I will just use do_pulse_sequence 
	method of SingletTriplet class.
	'''
	
	def __init__(self, device, lePulse):
		'''
		'''
		self.device = device
		self.lePulse = lePulse
		self.simulation_result = None
	
	
	def do_pulse_sequence(self, psi0, **kwargs):
		'''
		Run a detuning pulse sequence
		'''
		opts = qu.Options(nsteps=100000)
		
		self.device.build_hamiltonian(detuning=0)
		initial_hamiltonian = self.device.Hamiltonian
		conversion_factor = self.device.alpha*const.eV/const.h		# Convert voltage pulse to frequency units
		detuning_hamiltonian = conversion_factor*qu.Qobj(np.array([[1/2, 0, 0, 0, 0], 
																[0, 1/2, 0,0,0], 
																[0, 0, 1/2,0,0], 
																[0,0,0,1/2,0], 
																[0,0,0,0,-1/2]]))
		qutip_time_dependence = qu.Cubic_Spline(self.lePulse.time_vec[0], self.lePulse.time_vec[-1], 
												self.lePulse.simulation_waveform)
		full_H = [initial_hamiltonian, [detuning_hamiltonian, qutip_time_dependence]]
		output = qu.mesolve(full_H, psi0, tlist=self.lePulse.time_vec, c_ops=[], e_ops=[], options=opts, progress_bar=True)
		self.simulation_result = output
	
	
	
	
	
	
	
	
