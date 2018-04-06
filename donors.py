# Make some classes containing useful parameters regarding 31P donors in silicon

import scipy.constants as const
import matplotlib.pyplot as plt
import qutip as qu
import numpy as np
import seaborn as sb


class SingleDonor():
	'''
	This class creates a single 31P donor object
	'''
	
	# Define some constants
	_mu_b = const.physical_constants['Bohr magneton'][0]
	_mu_n = const.physical_constants['nuclear magneton'][0]
	# Set up the basis I'm going to use. I believe I have defined this correctly...
	upup = qu.tensor(qu.basis(2,0), qu.basis(2,0))
	updn = qu.tensor(qu.basis(2,0), qu.basis(2,1))
	dnup = qu.tensor(qu.basis(2,1), qu.basis(2,0))
	dndn = qu.tensor(qu.basis(2,1), qu.basis(2,1))
	
	# Defining electron and nuclear spin operators
	Sz_operator = qu.tensor(qu.sigmaz(), qu.identity(2))
	Sy_operator = qu.tensor(qu.sigmay(), qu.identity(2))
	Sx_operator = qu.tensor(qu.sigmax(), qu.identity(2))
	Iz_operator = qu.tensor(qu.identity(2), qu.sigmaz())
	Iy_operator = qu.tensor(qu.identity(2), qu.sigmay())
	Ix_operator = qu.tensor(qu.identity(2), qu.sigmax())
		
	def __init__(self, **kwargs):
		'''
		Docstring
		'''
		g_factor = kwargs.get('electron_g_factor', 2)
		ng_factor = kwargs.get('nuclear_g_factor', 2.2632)
		self.hyperfine = kwargs.get('default_hyperfine', 2*np.pi*114e6)
		self.b_z = kwargs.get('default_mag_field', 1)
		
		self.electron_gyro = g_factor*self._mu_b/const.hbar
		self.nuclear_gyro = ng_factor*self._mu_n/const.hbar
		self.Hamiltonian = self.build_hamiltonian()
		
	def build_hamiltonian(self, **kwargs):
		'''
		Set up the Hamiltonian for the single donor
		'''
		b_z = kwargs.get('static_mag', self.b_z)
		hyperfine = kwargs.get('hyperfine', self.hyperfine)
		H_electron_zeeman = 0.5*self.electron_gyro*b_z*qu.tensor(qu.sigmaz(), qu.identity(2))
		H_nuclear_zeeman = 0.5*self.nuclear_gyro*b_z*qu.tensor(qu.identity(2), qu.sigmaz())
		H_hyperfine = hyperfine*(qu.tensor(qu.sigmax(), qu.sigmax()) + qu.tensor(qu.sigmay(), qu.sigmay()) 
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
		plt.plot(b_sweep, self.eigenvals/(2*np.pi))
		plt.xlabel('Magnetic field (T)')
		plt.ylabel('Transition frequency (Hz)')
		plt.figure(2)
		plt.plot(b_sweep, projvecs[:,0:2], '+')
		plt.xlabel('Magnetic field (T)')
		plt.ylabel('Projection')
		plt.legend(['E down, N up', 'E down, N down'])
		plt.figure(3)
		plt.plot(b_sweep, projvecs[:,2:4], '+')
		plt.xlabel('Magnetic field (T)')
		plt.ylabel('Projection')
		plt.legend(['E up, N down', 'E up, N up'])
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
	
	# def do_esr(self, string='lost self', tlist=np.linspace(0,100e-12,1000)):
		# print(string)
	
	def do_spin_resonance(self, b1=0.01, bz=1, freq=28e9, psi0=dnup, tlist=np.linspace(0,1e-9,1000)):
		'''
		Perform an ESR/NMR control operation on the system. I hope to have this set up so that you do ESR
		by putting in a signal at the ESR frequency, and NMR by putting a signal in at NMR frequency
		'''
		e_ops_list = [self.Sx_operator, self.Sy_operator, self.Sz_operator, self.Ix_operator, 
						self.Iy_operator, self.Iz_operator]
		drive_args = {'b1':b1, 'freq':freq, 'phase':0}
		self.Hamiltonian = self.build_hamiltonian(static_mag=bz)
		esr_Hamiltonian = 0.5*self.electron_gyro*b1*self.Sx_operator
		nmr_Hamiltonian = 0.5*self.nuclear_gyro*b1*self.Ix_operator
		full_H = [self.Hamiltonian, [esr_Hamiltonian+nmr_Hamiltonian, rabi_drive]]
		output = qu.mesolve(full_H, psi0, tlist, c_ops=[], e_ops=e_ops_list, args=drive_args)
		self.plot_trajectories(output.times, np.array(output.expect[0:3]))
		self.plot_trajectories(output.times, np.array(output.expect[3:6]))
		return output
		
	def plot_trajectories(self, times, trajs):
		f = plt.figure(figsize=(5,5), facecolor='white')
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
		Defining the function for the time dependence. Not sure if this will work with the extra argument or not?
		'''
		return args['b1']*np.cos(2*np.pi*args['freq']*t + args['phase'])		
		
		
class DonorSingletTriplet(SingleDonor):
	'''
	Class defining a singlet triplet 31P donor spin system
	'''
	
	# Define basis vectors
	up = qu.basis(2,0)
	dn = qu.basis(2,1)
	left = qu.basis(2,0)
	right = qu.basis(2,1)
	
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

	def build_hamiltonian(self, b_z=1, dBz=0.01, detuning=0, tc=1e9):
		'''
		Build the singlet-triplet Hamiltonian
		'''
		# Zeeman term for electrons on left and right dots
		Zeeman = 0.5*self.electron_gyro*b_z
		# H_zeeman_left = 0.5*self.electron_gyro*b_z*qu.tensor(qu.sigmaz(), qu.identity(2), qu.identity(2))
		# H_zeeman_right = 0.5*self.electron_gyro*(b_z+dBz)*qu.tensor(qu.identity(2), qu.sigmaz(), qu.identity(2))
		# # J = self.calculate_exchange(detuning, tc)
		# # H_exchange = J*(qu.tensor(qu.sigmax(), qu.sigmax(), qu.identity(2)) + 
						# # qu.tensor(qu.sigmay(), qu.sigmay(), qu.identity(2)) +
						# # qu.tensor(qu.sigmaz(), qu.sigmaz(), qu.identity(2)))
		# H_charge = (detuning/2*(qu.tensor(qu.identity(2), qu.identity(2), qu.sigmaz())) + 
						# tc*qu.tensor(qu.identity(2), qu.identity(2), qu.sigmax()))
		# H = H_zeeman_left + H_zeeman_right + H_charge
		
		# Just taking 5 by 5 Hamiltonian 
		H = qu.Qobj(np.array([[detuning/2, 0, dBz, 0, tc], [0, detuning/2-Zeeman, 0,0,0], [dBz, 0, detuning/2,0,0], 
								[0,0,0,detuning/2+Zeeman,0], [tc,0,0,0,-detuning/2]]))
		return H
		
	def calculate_exchange(self, detuning, tc):
		'''
		Calculate the exchange energy as a function of detuning and tunnel coupling
		'''
		J = detuning/2 + np.sqrt((detuning/2)**2 + tc**2)
		return J
		
	# def plot_spectrum_vs_bfield(self, ):
	
	def plot_spectrum_vs_detuning(self, start, stop, num_points=1000, b_z=1, tc=1e9):
		'''
		'''
		detuning_sweep = np.linspace(start, stop, num_points)
		self.eigenvals = np.zeros([num_points, 5])
		# projvecs = np.zeros([len(b_sweep), 4])
		for ind,val in enumerate(detuning_sweep):
			self.Hamiltonian = self.build_hamiltonian(detuning=val, b_z=b_z, tc=tc)
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
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		