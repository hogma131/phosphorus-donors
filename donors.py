# Classes intended for simulating 31P donors in silicon

import scipy.constants as const
import matplotlib.pyplot as plt
import qutip as qu
import numpy as np
import seaborn as sb


class SingleDonor():
	'''
	This class creates a single 31P donor object
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
			electron_g_factor (optional) - electron g-factor, value is 2 if not specifiend
			nuclear_g_factor (optional) - nuclear g-factor
			hyperfine_coupling (optional) - hyperfine coupling constant between electron and nuclear spin. Default 
											value is bulk single donor
			mag_field (optional) - static magnetic field in z direction. Default is 1 Tesla if not specified.
		'''
		self._g_factor = kwargs.get('electron_g_factor', 2.0)
		self._ng_factor = kwargs.get('nuclear_g_factor', 2.2632)
		self.hyperfine = kwargs.get('hyperfine_coupling', 2*np.pi*114e6)
		self.b_z = kwargs.get('mag_field', 1.0)
		
		self.electron_gyro = self._g_factor*self._mu_b/const.hbar
		self.nuclear_gyro = self._ng_factor*self._mu_n/const.hbar
		self.Hamiltonian = self.build_hamiltonian()
		
	def build_hamiltonian(self, **kwargs):
		'''
		Construct static single donor Hamiltonian for given physical parameters
		Inputs:
			static_mag (optional) - static bz field, defaults to object attribute if not specified
		'''
		self.b_z = kwargs.get('static_mag', self.b_z)
		H_electron_zeeman = 0.5*self.electron_gyro*self.b_z*self.Sz_operator
		H_nuclear_zeeman = 0.5*self.nuclear_gyro*self.b_z*self.Iz_operator
		H_hyperfine = self.hyperfine*(qu.tensor(qu.sigmax(), qu.sigmax()) + qu.tensor(qu.sigmay(), qu.sigmay()) 
				+ qu.tensor(qu.sigmaz(), qu.sigmaz()))
		H = H_electron_zeeman - H_nuclear_zeeman + H_hyperfine
		return H
		
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
		self.eigenvals = np.zeros([len(b_sweep), 4])
		projvecs = np.zeros([len(b_sweep), 4])
		for ind,val in enumerate(b_sweep):
			self.Hamiltonian = self.build_hamiltonian(static_mag=val)
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
	
	
	def do_spin_resonance(self, b1=0.01, bz=1.0, psi0=dnup, **kwargs):
		'''
		Perform an ESR/NMR control operation on the system. The simulation works with the full Hamiltonian of both
		the electron and the nucleus, so to choose whether to do an ESR or NMR experiment you just change the drive
		frequency to match the appropriate resonance.
		Inputs:
			b1 - amplitude of ac magnetic drive (default 0.01T)
			bz - static mag field (default 1.0T)
			psi0 - initial state (default dnup state)
			kwargs:
			freq - Rabi drive frequency (default is energy splitting between dnup and upup states)
			tmax - Simulation time (default is approximately a pi/2 pulse for ESR drive)
		Outputs:
			output - Expectation values of electron and nuclear spin operators (ordered as Sx, Sy, Sz, Ix, Iy, Iz)
		Todo:
			- Have detuning from resonance as an input parameters
			- Come up with a more flexible way of choosing initial states and resonances to drive
		'''
		self.Hamiltonian = self.build_hamiltonian(static_mag=bz) 	# Update the static Hamiltonian
		esr1_freq, esr2_freq, nmr1_freq, nmr2_freq = self.calculate_resonance_freqs()
		freq = kwargs.get('freq', esr1_freq) 		# AC drive frequency, defaults to energy splitting between dnup and upup
		tmax = kwargs.get('tmax', np.pi/(self.electron_gyro*b1)) 	# Default should be approximately a pi/2 pulse
		n_steps = kwargs.get('n_steps', 1000)
		tlist = np.linspace(0,tmax,n_steps)
		e_ops_list = [self.Sx_operator, self.Sy_operator, self.Sz_operator, self.Ix_operator, 
						self.Iy_operator, self.Iz_operator]
		drive_args = {'b1':b1, 'freq':freq, 'phase':0}
		esr_Hamiltonian = 0.5*self.electron_gyro*self.Sx_operator
		nmr_Hamiltonian = 0.5*self.nuclear_gyro*self.Ix_operator
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
	
		
class DonorSingletTriplet(SingleDonor):
	'''
	Class defining a singlet triplet 31P donor spin system. Needs work to include the nuclei.
	'''
	
	# Define basis vectors
	S11 = qu.basis(5,0)
	Tm = qu.basis(5,1)
	T0 = qu.basis(5,2)
	Tp = qu.basis(5,3)
	S20 = qu.basis(5,4)
	# Also updn and dnup basis - not sure if I have these the correct way round
	updn = (T0 - S11).unit()
	dnup = (T0 + S11).unit()
	
	def __init__(self, **kwargs):
		'''
		Docstring
		'''
		g_factor = kwargs.get('electron_g_factor', 2)
		ng_factor = kwargs.get('nuclear_g_factor', 2.2632)
		self.hyperfine = kwargs.get('default_hyperfine', 2*np.pi*114e6)
		self.b_z = kwargs.get('default_mag_field', 1)
		self.dBz = kwargs.get('delta_bz', 0.01)
		
		self.electron_gyro = g_factor*self._mu_b/const.hbar
		self.nuclear_gyro = ng_factor*self._mu_n/const.hbar
		self.Hamiltonian = self.build_hamiltonian(b_z=self.b_z, dBz=self.dBz)

	def build_hamiltonian(self, b_z=1, dBz=0.01, detuning=0, tc=1e9, alpha=0.1):
		'''
		Build the singlet-triplet Hamiltonian
		Inputs:
			b_z (Tesla) - static magnetic field applied along the z direction
			dBz (Tesla) - delta Bz between the two qubit sites
			detuning (V) - detuning from the charge degeneracy point in volts
			tc (Hz)		- tunnel coupling (in frequency units)
			alpha 		- lever arm of gate that voltage is applied to (unitless)
		Todo:
			- Check that detuning scaling is correct (factor of 2pi or not)
		'''
		# Convert Zeeman terms to frequency units
		Zeeman = (0.5*self.electron_gyro*b_z)/(2*np.pi)
		dBz = 0.5*self.electron_gyro*dBz/(2*np.pi)
		detuning = alpha*detuning*const.eV/(2*np.pi*const.h)	# Convert from voltage to frequency
		# Zeeman term for electrons on left and right dots
		#~ H_zeeman_left = 0.5*self.electron_gyro*b_z*qu.tensor(qu.sigmaz(), qu.identity(2), qu.identity(2))
		#~ H_zeeman_right = 0.5*self.electron_gyro*(b_z+dBz)*qu.tensor(qu.identity(2), qu.sigmaz(), qu.identity(2))
		#~ J = self.calculate_exchange(detuning, tc)
		#~ H_exchange = J*(qu.tensor(qu.sigmax(), qu.sigmax(), qu.identity(2))
						#~ + qu.tensor(qu.sigmay(), qu.sigmay(), qu.identity(2))
						#~ + qu.tensor(qu.sigmaz(), qu.sigmaz(), qu.identity(2)))
		#~ H_charge = (detuning/2*(qu.tensor(qu.identity(2), qu.identity(2), qu.sigmaz())) + 
						#~ tc*qu.tensor(qu.identity(2), qu.identity(2), qu.sigmax()))
		#~ H = H_zeeman_left + H_zeeman_right + H_charge + H_exchange
		
		# Just taking 5 by 5 Hamiltonian 
		# Defined in the basis [S11, T-, T0, T+, S20]
		H = qu.Qobj(np.array([[detuning/2, 0, dBz, 0, tc], [0, detuning/2-Zeeman, 0,0,0], [dBz, 0, detuning/2,0,0], 
								[0,0,0,detuning/2+Zeeman,0], [tc,0,0,0,-detuning/2]]))
		return H
		
	def build_full_hamiltonian(self, b_z=1, dBz=0.01, detuning=0, tc1=1e9, tc2=0.5e9, pauli_energy=50e9, alpha=0.1):
		'''
		Make the full 8 by 8 spin Hamiltonian for singlet and triplet states on both dots
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
		Project state psi onto the basis [S11, T-, T0, T+, S20, updn, dnup]
		'''
		S11_proj = np.abs(psi.overlap(self.S11))**2
		Tm_proj = np.abs(psi.overlap(self.Tm))**2
		T0_proj = np.abs(psi.overlap(self.T0))**2
		Tp_proj = np.abs(psi.overlap(self.Tp))**2
		S20_proj = np.abs(psi.overlap(self.S20))**2
		updn_proj = np.abs(psi.overlap(self.updn))**2
		dnup_proj = np.abs(psi.overlap(self.dnup))**2
		overlap_vec = np.array([S11_proj, Tm_proj, T0_proj, Tp_proj, S20_proj, updn_proj, dnup_proj])
		return overlap_vec
		
	def calculate_exchange(self, detuning, tc):
		'''
		Calculate the exchange energy as a function of detuning and tunnel coupling
		'''
		J = detuning/2 + np.sqrt((detuning/2)**2 + tc**2)
		return J
		
	
	def plot_spectrum_vs_detuning(self, start, stop, num_points=1000, b_z=1, dBz=0.01, tc=1e9, alpha=0.1):
		'''
		'''
		detuning_sweep = np.linspace(start, stop, num_points)
		# self.eigenvals = np.zeros([num_points, 8])
		self.eigenvals = np.zeros([num_points, 5])
		# projvecs = np.zeros([len(b_sweep), 4])
		for ind,val in enumerate(detuning_sweep):
			self.Hamiltonian = self.build_hamiltonian(detuning=val, b_z=b_z, tc=tc, dBz=dBz, alpha=alpha)
			# self.Hamiltonian = self.build_full_hamiltonian(detuning=val, b_z=b_z, dBz=dBz, alpha=alpha)
			self.eigenvals[ind], temp_vecs = self.Hamiltonian.eigenstates()
			# projvecs[ind] = self.project_eigenvecs(temp_vecs)
		plt.figure(1)
		plt.plot(detuning_sweep, self.eigenvals)
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
		plt.show(block=False)
		return plt
		
	def do_detuning_pulse(self, start, stop, psi0, ramp_time, num_points=1000, **kwargs):
		'''
		Run an experiment that pulses the detuning from a start value to a stop value
		'''
		tlist = np.linspace(0, ramp_time, num_points)
		pulse_args = {'start':start, 'stop': stop, 'tmax': ramp_time}
		tc = kwargs.get('tunnel_coupling', 500e6)
		options = kwargs.get('options', None)
		alpha = kwargs.get('alpha', 0.1)
		b_z = kwargs.get('b_z', 0.1)
		dBz = kwargs.get('dBz', 0.005)
		start_hamiltonian = self.build_hamiltonian(detuning=start, b_z=b_z, dBz=dBz, tc=tc, alpha=alpha)
		conversion = alpha*const.eV/const.h		# Convert voltage pulse to frequency units
		detuning_hamiltonian = conversion*qu.Qobj(np.array([[1/2, 0, 0, 0, 0], 
														[0, 1/2, 0,0,0], 
														[0, 0, 1/2,0,0], 
														[0,0,0,1/2,0], 
														[0,0,0,0,-1/2]]))
		full_H = [start_hamiltonian, [detuning_hamiltonian, detuning_pulse]]
		output = qu.mesolve(full_H, psi0, tlist, c_ops=[], e_ops=[], args=pulse_args, options=options, progress_bar=True)
		return output, tlist
		
		
class CoupledDonors(): 

	'''
	Class for a coupled donor system
	'''
	
	def __init__(self, Donor1, Donor2):
		'''
		'''

class PulseSequence():
	'''
	Class to encapsulate a pulsing object which can be used in a time-dependent simulation
	'''
	
	def __init__(self):
		'''
		'''
		self.segments = []
		self.segment_names = []
		
	def add_segment(self, pulse_seg, name):
		'''
		Add a segment to the sequence
		'''
		assert (name not in self.segment_names), "Pulse segment with that name already exists!"
		self.segment_names.append(name)
		self.segments.append(pulse_seg)
		
	def make_waveform(self):
		'''
		Concatenate all the segments to make a waveform
		'''
		point_waveform = np.zeros(len(self.segments)/2 + 1)
		for ind,val in enumerate(self.segments):
			# point_waveform[ind] = 
		
	def resample_waveform(self):
		'''
		'''
		
		
class PulseSegment():
	'''
	Class for pulse segments, which can be stitched together to form a PulseSequence
	'''
	
	def __init__(self, pulse_type, duration, start_val, **kwargs):
		'''
		'''
		self.pulse_type = pulse_type
		self.duration = duration
		self.start_val = start_val
		self.end_val = kwargs.get('end_val', start_val)
		self.time_vector = np.array([0, self.duration])
		self.construct_pulse()
		
	def construct_pulse(self):
		'''
		'''
		if (self.pulse_type == 'level'):
			self.pulse_values = np.array([self.start_val, self.start_val])
		elif (self.pulse_type == 'ramp'):
			self.pulse_values = np.array([self.start_val, self.end_val])
			
	
	
	
	
	
	
	
	
	