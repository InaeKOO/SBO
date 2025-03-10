import os
import torch
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union, Callable, Tuple, List, Any

# Primitives for execution
from qiskit_aer import AerSimulator, Aer
from qiskit_aer.primitives import Sampler, Estimator

# Minimum eigensolver from the new algorithms module
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

# Import optimizers from the updated location (choose your preferred optimizer)
from qiskit_algorithms.optimizers import SLSQP, SPSA
import qiskit_algorithms.optimizers.optimizer as qiskitopt

# Qiskit Nature: Updated second quantization driver for electronic structure.
# (ElectronicStructureMoleculeDriver is still available for legacy use,
#  but new drivers like PySCFDriver are recommended.)
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo

# Ground state eigensolver for chemistry problems remains in Qiskit Nature,
# and we also import the VQE factory.
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_algorithms import VQE

# The ansatz has been updated: use UCCSD (singles and doubles)
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

# Converter and mapper for transforming the fermionic problem to qubits
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper

# Define the electronic structure problem (same name, but under the updated package)
from qiskit_nature.second_q.problems import ElectronicStructureProblem

from qiskit_ionq import IonQProvider

provider = IonQProvider(token="12781dfb-bf64-46ec-96fe-40a7499f14f3")
#qpu_backend = provider.get_backend("qpu.aria-1")
print(provider.backends())

# Type alias for points (can be a float or an array)
POINT = Union[float, np.ndarray]
# Create an instance of AerSimulator (here, specifying GPU)
sim = AerSimulator(device="GPU")
qasm_backend = AerSimulator(method="automatic", shots=100)
# Now call available_devices() on the instance
print(sim.available_devices())

def scott_bandwidth(n: int, d: int) -> float:
    '''
    Scott's Rule per D.W. Scott,
    "Multivariate Density Estimation: Theory, Practice, and Visualization",
    John Wiley & Sons, New York, Chicester, 1992
    '''
    return n ** (-1. / (d + 4))

###############################################################################
# Custom Optimizer (a surrogate-based optimizer) using the new Qiskit optimizer API

class Optimizer(qiskitopt.Optimizer):
    def __init__(
        self,
        maxiter: int = 100,
        patch_size: float = 0.1,
        npoints_per_patch: int = 20,
        epsilon_i: float = 0.0,
        epsilon_int: float = 0.05,
        epsilon_f: float = 0.5,
        nfev_final_avg: int = 0,
    ) -> None:
        super().__init__()
        # general optimizer arguments
        self.maxiter = maxiter
        self.patch_size = patch_size
        self.npoints_per_patch = npoints_per_patch
        self.epsilon_i = epsilon_i
        self.epsilon_int = epsilon_int
        self.epsilon_f = epsilon_f
        self.nfev_final_avg = nfev_final_avg

    def get_support_level(self) -> Dict:
        """Get the support level dictionary."""
        return {
            "initial_point": qiskitopt.OptimizerSupportLevel.required,
            "gradient": qiskitopt.OptimizerSupportLevel.ignored,
            "bounds": qiskitopt.OptimizerSupportLevel.ignored,
        }
    
    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Optional[Callable[[POINT], POINT]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> qiskitopt.OptimizerResult:
         
        optimizer_iteration = OptimizerIteration()

        current_x = x0
        local_minima_found = []
        for i in range(self.maxiter):
            optimize_bounds_size = (
                self.patch_size
                * (1.0 - self.epsilon_i)
                * (1.0 - i / self.maxiter)
            )
            res = optimizer_iteration.minimize_kde(
                fun,
                current_x,
                self.patch_size,
                optimize_bounds_size,
                self.npoints_per_patch,
            )
            new_x = res.x
            distance = np.linalg.norm(new_x - current_x, ord=np.inf)
            #print("f(x) difference : ", abs(fun(new_x)-fun(current_x)))
            #print("distance : ",distance)
            #print("ratio : ", 0.1 / distance * abs(fun(new_x)-fun(current_x)))
            self.batch_size = distance / max(abs(fun(new_x)-fun(current_x), 0.01))
            print("batch_size : ",self.batch_size)
            #self.batch_size = distance * 100
            current_x = new_x
            #self.patch_size = distance * 
            if distance < (self.patch_size / 2) * (1 - self.epsilon_int):
                # local minimum found within this patch area
                local_minima_found.append(new_x)

        local_minima_near_current_x = [
            local_minimum
            for local_minimum in local_minima_found
            if (
                np.linalg.norm(local_minimum - current_x, ord=np.inf)
                < (self.patch_size / 2) * self.epsilon_f
            )
        ]
        optimal_x = (
            np.mean(local_minima_near_current_x, axis=0)
            if local_minima_near_current_x
            else current_x
        )
        result = qiskitopt.OptimizerResult()
        result.nfev = (
            (self.maxiter * self.npoints_per_patch)
            + self.nfev_final_avg
        )
        result.nit = self.maxiter

        result.x = optimal_x
        if self.nfev_final_avg > 0:
            result.fun = np.mean(
                [fun(optimal_x) for _ in range(self.nfev_final_avg)]
            )
        else:
            result.fun = (
                'final function value not evaluated '
                + 'because nfev_final_avg == 0'
            )

        return result

class OptimizerIteration:
    '''
    Implements a single iteration of surrogate-based optimization using
    a Gaussian kernel.
    '''
    def __init__(self) -> None:
        pass

    def get_conditional_expectation_with_gradient(
        self,
        training_data: np.ndarray,
        x: np.ndarray,
        bandwidth_function: Callable = scott_bandwidth,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # normalize training data coordinates
        training_x = training_data[:, :-1]
        training_x_mean = np.mean(training_x, axis=0)
        training_x_std = np.std(training_x, axis=0)
        training_x_std[training_x_std == 0.0] = 1.0
        training_x_normalized = (training_x - training_x_mean) / training_x_std

        # normalize input coordinates
        x_normalized = (x - training_x_mean) / training_x_std

        # normalize training data z-values
        training_z = training_data[:, -1]
        training_z_mean = np.mean(training_z)
        training_z_std = np.std(training_z) or 1.0
        training_z_normalized = (training_z - training_z_mean) / training_z_std

        # get the normalized conditional expectation in z
        bandwidth = bandwidth_function(*training_x.shape)
        gaussians = np.exp(
            -1 / (2 * bandwidth**2)
            * np.linalg.norm((training_x_normalized - x_normalized), axis=1)**2
        )
        exp_z_normalized = (
            np.sum(training_z_normalized * gaussians) / np.sum(gaussians)
        )

        # calculate the gradients along each x coordinate
        grad_gaussians = np.array([
            (1 / (bandwidth**2))
            * (training_x_normalized[:, i] - x_normalized[i]) * gaussians
            for i in range(len(x_normalized))
        ])
        grad_exp_z_normalized = np.array([(
            np.sum(gaussians)
            * np.sum(training_z_normalized * grad_gaussians[i])
            - np.sum(training_z_normalized * gaussians)
            * np.sum(grad_gaussians[i])
        ) / (np.sum(gaussians)**2) for i in range(len(grad_gaussians))])

        # undo the normalization and return the expectation value and gradients
        exp_z = training_z_mean + training_z_std * exp_z_normalized
        grad_exp_z = training_z_std * grad_exp_z_normalized

        return exp_z, grad_exp_z
    
    def _minimize_kde(
        self,
        angles: np.ndarray,
        values: np.ndarray,
        patch_center_x: np.ndarray,
        optimize_bounds_size: float,
    ) -> Any:
        training = np.concatenate((angles, values), axis=1)
        num_angles = len(patch_center_x)
        bounds_limits = np.array([
            [
                patch_center_x[angle] - optimize_bounds_size / 2,
                patch_center_x[angle] + optimize_bounds_size / 2
            ]
            for angle in range(num_angles)
        ])
        bounds = optimize.Bounds(
            lb=bounds_limits[:, 0],
            ub=bounds_limits[:, 1],
            keep_feasible=True
        )
        return optimize.minimize(
            fun=lambda x:
                self.get_conditional_expectation_with_gradient(training, x),
            jac=True,
            x0=patch_center_x,
            bounds=bounds,
            method="L-BFGS-B",
        )
        
    def minimize_kde(
        self,
        f: Callable,
        patch_center_x: np.ndarray,
        patch_size: float,
        optimize_bounds_size: float,
        npoints_per_patch: int,
    ) -> Any:
        training_point_angles = self._generate_x_coords(
            patch_center_x, patch_size, npoints_per_patch)
        measured_values = np.atleast_2d(
            [f(x) for x in training_point_angles]
        ).T

        return self._minimize_kde(
            training_point_angles,
            measured_values,
            patch_center_x,
            optimize_bounds_size,
        )
    
    def _generate_x_coords(
        self,
        center: np.ndarray,
        patch_size: float,
        num_points: int = 40,
    ) -> np.ndarray:
        '''
        Generate num_points sample coordinates using Latin hypercube sampling.
        _lhsclassic copied from:
        https://github.com/tisimst/pyDOE/blob/master/pyDOE/doe_lhs.py#L123-L141
        '''
        def _lhsclassic(n: int, samples: int) -> np.ndarray:
            # Generate the intervals
            cut = np.linspace(0, 1, samples + 1)

            # Fill points uniformly in each interval
            u = np.random.rand(samples, n)
            a = cut[:samples]
            b = cut[1:samples + 1]
            rdpoints = np.zeros_like(u)
            for j in range(n):
                rdpoints[:, j] = u[:, j] * (b - a) + a

            # Make the random pairings
            H = np.zeros_like(rdpoints)
            for j in range(n):
                order = np.random.permutation(range(samples))
                H[:, j] = rdpoints[order, j]

            return H

        n_dim = len(center)
        lhs_points = _lhsclassic(n_dim, num_points)
        return np.array([
            ((point - 0.5) * 2 * patch_size) + center
            for point in lhs_points
        ])

###############################################################################
# Helper function to create an ideal (noiseless) quantum backend

def ideal_backend(nshots):
    # Construct an ideal simulator with only quantum shot noise.
    # Note: In the new API, we instantiate an AerSimulator and pass it to the Sampler.
    simulator = AerSimulator()
    return Sampler(backend=simulator, shots = nshots)

###############################################################################
# Function to solve for the ground state energy using a given solver.
# (Relies on global variables: `mapper` and `es_problem`.)
def solve(vqe):
    solver = GroundStateEigensolver(mapper, vqe)
    result = solver.solve(es_problem)
    return result.total_energies[0] 

def plot_energies(energies, label, marker, color):
    y = np.mean(list(energies.values()), axis=1)
    stds = np.std(list(energies.values()), axis=1)
    yerr = np.divide(stds, np.sqrt(len(stds)))
    linestyle_dotted = (0, (2, 3))
    plt.errorbar(
        distances, y, yerr, elinewidth=1.0, capsize=3.0,
        marker=marker, linestyle=linestyle_dotted,
        label=label, color=color)

def save_figure(filename, ext='png'):
    base = filename
    extension = '.' + ext if not filename.endswith(ext) else ''
    new_filename = base + extension
    counter = 1
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{extension}"
        counter += 1
    plt.savefig(new_filename)
    print("Saved to:", new_filename)

###############################################################################
# Main execution: Evaluate energies over a range of H–H distances

if __name__ == "__main__":
    distances = np.arange(0.2, 1.5, 0.05)   # true ground state ~0.735
    exact_energies = {}
    vqe_sbo_energies = {}
    vqe_spsa_energies = {}
    nshots = 100
    repetition_count = 5
    
    for distance in distances:
        exact_energies[distance] = []
        vqe_sbo_energies[distance] = []
        vqe_spsa_energies[distance] = []
        molecule = MoleculeInfo(
            ["H", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, distance)],
            charge=0,
            multiplicity=1
        )
        driver = PySCFDriver.from_molecule(molecule, basis="sto3g")
        es_problem = driver.run()
        mapper = JordanWignerMapper()

        backend_options = {"method": "automatic","device": "gpu"} if torch.cuda.is_available() else {"method": "automatic"}
        estimator = Estimator(backend_options=backend_options, run_options={
                        "shots": nshots
                })
        #print("Number of free parameters in UCCSD:", ansatz.num_parameters)

        for _ in range(repetition_count):
            
            # Exact solver
            numpy_solver = NumPyMinimumEigensolver()
            exact_energies[distance].append(solve(numpy_solver))

            # VQE using our custom surrogate-based optimizer (SBO)
            sbo_optimizer = Optimizer(
                maxiter=20,
                patch_size=0.15,
                npoints_per_patch=4,
                nfev_final_avg=4
            )
            ansatz = UCCSD(
                num_spatial_orbitals=2,
                num_particles=(1,1),
                qubit_mapper=mapper,
                initial_state=HartreeFock(
                    2,(1,1),mapper,
                )
            )
            vqe_solver = VQE(
                estimator, ansatz, optimizer=sbo_optimizer
            )
            vqe_sbo_energies[distance].append(solve(vqe_solver))

            # VQE with SPSA optimizer on ideal simulator, num function evals=20
            spsa_optimizer = SPSA(maxiter=10)
            ansatz = UCCSD(
                num_spatial_orbitals=2,
                num_particles=(1,1),
                qubit_mapper=mapper,
                initial_state=HartreeFock(
                    2,(1,1),mapper,
                )
            )
            vqe_solver = VQE(
                estimator, ansatz, optimizer=spsa_optimizer
            )
            vqe_spsa_energies[distance].append(solve(vqe_solver))

        print(f"Distance: {distance:.3f}, " + 
              f"SBO: {np.mean(vqe_sbo_energies[distance]):.6f}, " + 
              f"SPSA: {np.mean(vqe_spsa_energies[distance]):.6f}, " + 
              f"Exact Result: {np.mean(exact_energies[distance]):.6f}")
        
    plt.figure(figsize=(7, 5))
    plt.rcParams['font.size'] = 14
    plt.plot(distances, np.mean(list(exact_energies.values()), axis=1), 'k-', linewidth=1, label='exact')
    plot_energies(vqe_spsa_energies, f'SPSA', marker='^', color='C0')
    plot_energies(vqe_sbo_energies, f'SBO', marker='o', color='C1')

    plt.legend()
    plt.xlabel('interatomic distance')
    plt.ylabel('energy')
    plt.ylim(-1.3, 1.9)
    plt.title(r"H$_2$ VQE, shot noise only, K=100")
    save_figure("H2_VQE_4q")
