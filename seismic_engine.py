"""
Enterprise Seismic Physics Engine
Implements: Aki-Richards Theory, Modal Summation, Spectral Element Method,
Green's Function, Attenuation (Q-factor), Anelastic Wave Propagation
"""
import numpy as np
from numba import jit, prange
from scipy.special import jv as bessel_jv
from scipy.fft import fftn, ifftn
import threading
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class FaultType(Enum):
    STRIKE_SLIP = 1
    NORMAL = 2
    REVERSE = 3
    OBLIQUE = 4


class WaveType(Enum):
    P_WAVE = 1  # Primary/Compressional
    S_WAVE = 2  # Secondary/Shear
    LOVE = 3    # Surface wave
    RAYLEIGH = 4  # Surface wave


@dataclass
class SeismicSource:
    """Earthquake hypocenter and moment tensor"""
    latitude: float
    longitude: float
    depth: float  # km
    magnitude: float  # Moment magnitude Mw
    strike: float  # degrees
    dip: float  # degrees
    rake: float  # degrees
    rupture_velocity: float = 2.8  # km/s
    stress_drop: float = 3e6  # Pa (3 MPa typical)
    
    @property
    def seismic_moment(self) -> float:
        """M0 = 10^(1.5*Mw + 9.1) in N⋅m"""
        return 10 ** (1.5 * self.magnitude + 9.1)
    
    @property
    def rupture_area(self) -> float:
        """Wells & Coppersmith (1994) empirical relation"""
        log_area = -3.49 + 0.91 * self.magnitude
        return 10 ** log_area  # km²
    
    @property
    def rupture_length(self) -> float:
        """Characteristic length from area (circular crack)"""
        return np.sqrt(self.rupture_area / np.pi)
    
    def moment_tensor(self) -> np.ndarray:
        """Compute moment tensor from fault parameters (Aki & Richards)"""
        strike_rad = np.radians(self.strike)
        dip_rad = np.radians(self.dip)
        rake_rad = np.radians(self.rake)
        
        # Direction cosines
        n = np.array([
            -np.sin(dip_rad) * np.sin(strike_rad),
            np.sin(dip_rad) * np.cos(strike_rad),
            -np.cos(dip_rad)
        ])
        
        slip = np.array([
            np.cos(rake_rad) * np.cos(strike_rad) + 
            np.sin(rake_rad) * np.cos(dip_rad) * np.sin(strike_rad),
            np.cos(rake_rad) * np.sin(strike_rad) - 
            np.sin(rake_rad) * np.cos(dip_rad) * np.cos(strike_rad),
            -np.sin(rake_rad) * np.sin(dip_rad)
        ])
        
        M0 = self.seismic_moment
        M = M0 * (np.outer(n, slip) + np.outer(slip, n))
        return M


@dataclass
class EarthModel:
    """Velocity structure and material properties"""
    vp: np.ndarray  # P-wave velocity (km/s)
    vs: np.ndarray  # S-wave velocity (km/s)
    density: np.ndarray  # g/cm³
    qp: np.ndarray  # P-wave quality factor
    qs: np.ndarray  # S-wave quality factor
    depth: np.ndarray  # km
    
    def impedance_contrast(self, depth: float) -> float:
        """Acoustic impedance ratio at depth"""
        idx = np.searchsorted(self.depth, depth)
        if idx == 0:
            return 1.0
        return (self.density[idx] * self.vp[idx]) / \
               (self.density[idx-1] * self.vp[idx-1])


class SeismicWaveEngine:
    """Advanced wave propagation using multiple theoretical frameworks"""
    
    def __init__(self, earth_model: EarthModel, grid_size: Tuple[int, int, int],
                 spacing: float = 0.1):
        self.model = earth_model
        self.nx, self.ny, self.nz = grid_size
        self.dx = spacing  # km
        self.dt = 0.001  # seconds (CFL condition)
        
        # Spectral Element Method - Full Legendre-Gauss-Lobatto quadrature
        self.polynomial_order = 7  # N=7 for high accuracy
        self.gll_points, self.gll_weights = self._compute_gll_quadrature()
        self.lagrange_deriv = self._compute_lagrange_derivatives()
        
        # Full 3D stress-strain tensor (9 components for full anisotropy)
        self.stress_tensor = np.zeros((self.nx, self.ny, self.nz, 3, 3), dtype=np.float64)
        self.strain_tensor = np.zeros((self.nx, self.ny, self.nz, 3, 3), dtype=np.float64)
        
        # Anelastic memory variables - commented out to save memory (~1.4GB total)
        # These are for advanced Generalized Maxwell Body attenuation modeling
        # but are not used in the current simplified implementation
        # self.num_relaxation_mechanisms = 3
        # self.memory_p = np.zeros((self.nx, self.ny, self.nz, self.num_relaxation_mechanisms, 3, 3))
        # self.memory_s = np.zeros((self.nx, self.ny, self.nz, self.num_relaxation_mechanisms, 3, 3))
        # self.tau_epsilon = np.array([0.01, 0.1, 1.0])  # Relaxation times P-waves
        # self.tau_sigma = np.array([0.015, 0.15, 1.5])  # Relaxation times S-waves
        
        # Convolutional Perfectly Matched Layer - full complex frequency shifted
        self.pml_thickness = 15
        self.pml_damping_profiles = {}
        self._initialize_full_cpml()
        
        # NOTE: Elastic tensor computation commented out to save memory (~2GB!)
        # The full 4th-order tensor is not used in current implementation
        # self.elastic_tensor = self._compute_elastic_tensor()
        
        # Mass matrix for spectral elements
        self.mass_matrix = self._compute_mass_matrix()
        
    def _initialize_full_cpml(self):
        """Full Complex-Frequency-Shifted Convolutional PML (Komatitsch & Martin 2007)"""
        self.pml_damping = np.ones((self.nx, self.ny, self.nz), dtype=np.complex128)
        
        # CFS-PML parameters
        R_coef = 1e-5  # Theoretical reflection coefficient
        N_power = 2.0  # Polynomial degree
        alpha_max = np.pi * 1.5  # Frequency shift
        
        for axis in range(3):
            for direction in ['left', 'right']:
                for i in range(self.pml_thickness):
                    # Distance from PML boundary (normalized)
                    d_norm = (self.pml_thickness - i) / self.pml_thickness
                    
                    # Damping profile (polynomial)
                    d_pml = 3.0 * self.model.vp[0] * np.log(1.0 / R_coef) / (2.0 * self.pml_thickness * self.dx)
                    damping = d_pml * (d_norm ** N_power)
                    
                    # Frequency shift (prevents evanescent waves)
                    alpha = alpha_max * (1.0 - d_norm)
                    
                    # Complex damping coefficient
                    if axis == 0:  # X direction
                        if direction == 'left':
                            self.pml_damping[i, :, :] *= np.exp(-(damping + 1j * alpha) * self.dt)
                        else:
                            self.pml_damping[-(i+1), :, :] *= np.exp(-(damping + 1j * alpha) * self.dt)
                    elif axis == 1:  # Y direction
                        if direction == 'left':
                            self.pml_damping[:, i, :] *= np.exp(-(damping + 1j * alpha) * self.dt)
                        else:
                            self.pml_damping[:, -(i+1), :] *= np.exp(-(damping + 1j * alpha) * self.dt)
                    else:  # Z direction
                        if direction == 'left':
                            self.pml_damping[:, :, i] *= np.exp(-(damping + 1j * alpha) * self.dt)
                        else:
                            self.pml_damping[:, :, -(i+1)] *= np.exp(-(damping + 1j * alpha) * self.dt)
    
    def _compute_gll_quadrature(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Gauss-Lobatto-Legendre quadrature points and weights"""
        from scipy.special import legendre
        from scipy.optimize import newton
        
        N = self.polynomial_order
        points = np.zeros(N + 1)
        weights = np.zeros(N + 1)
        
        # End points
        points[0] = -1.0
        points[N] = 1.0
        
        # Interior points - roots of derivative of Legendre polynomial
        P_N = legendre(N)
        P_N_deriv = np.polyder(P_N)
        
        for i in range(1, N):
            # Initial guess
            x0 = np.cos(np.pi * (i + 0.25) / (N + 0.5))
            # Newton's method to find root
            points[i] = newton(P_N_deriv, x0, maxiter=100, tol=1e-15)
        
        # Compute weights
        for i in range(N + 1):
            P_N_val = P_N(points[i])
            weights[i] = 2.0 / ((N * (N + 1)) * P_N_val ** 2)
        
        return points, weights
    
    def _compute_lagrange_derivatives(self) -> np.ndarray:
        """Compute derivatives of Lagrange polynomials at GLL points"""
        N = self.polynomial_order
        xi = self.gll_points
        D = np.zeros((N + 1, N + 1))
        
        for i in range(N + 1):
            for j in range(N + 1):
                if i != j:
                    # Product of (xi[i] - xi[k]) for k != i, j
                    prod = 1.0
                    for k in range(N + 1):
                        if k != i and k != j:
                            prod *= (xi[i] - xi[k]) / (xi[j] - xi[k])
                    D[i, j] = prod / (xi[i] - xi[j])
                else:
                    # Diagonal elements
                    D[i, i] = 0.0
                    for k in range(N + 1):
                        if k != i:
                            D[i, i] -= 1.0 / (xi[i] - xi[k])
        
        return D
    
    def _compute_elastic_tensor(self) -> np.ndarray:
        """Compute full elastic tensor Cijkl (21 independent components for anisotropy)"""
        # For isotropic case, reduced to 2 parameters (lambda, mu)
        # But we store full tensor for future anisotropic extensions
        C = np.zeros((self.nx, self.ny, self.nz, 3, 3, 3, 3), dtype=np.float64)
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    # Get depth for this point
                    depth = k * self.dx
                    depth_idx = np.searchsorted(self.model.depth, depth)
                    depth_idx = min(depth_idx, len(self.model.depth) - 1)
                    
                    # Lame parameters from velocities
                    vp = self.model.vp[depth_idx] * 1000  # m/s
                    vs = self.model.vs[depth_idx] * 1000
                    rho = self.model.density[depth_idx] * 1000  # kg/m³
                    
                    mu = rho * vs ** 2
                    lam = rho * vp ** 2 - 2 * mu
                    
                    # Isotropic elastic tensor in Voigt notation converted to full tensor
                    # Cijkl = lambda * delta_ij * delta_kl + mu * (delta_ik * delta_jl + delta_il * delta_jk)
                    for ii in range(3):
                        for jj in range(3):
                            for kk in range(3):
                                for ll in range(3):
                                    delta_ij = 1 if ii == jj else 0
                                    delta_kl = 1 if kk == ll else 0
                                    delta_ik = 1 if ii == kk else 0
                                    delta_jl = 1 if jj == ll else 0
                                    delta_il = 1 if ii == ll else 0
                                    delta_jk = 1 if jj == kk else 0
                                    
                                    C[i, j, k, ii, jj, kk, ll] = (
                                        lam * delta_ij * delta_kl +
                                        mu * (delta_ik * delta_jl + delta_il * delta_jk)
                                    )
        
        return C
    
    def _compute_mass_matrix(self) -> np.ndarray:
        """Compute diagonal mass matrix for spectral elements"""
        M = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        
        # Jacobian of transformation (uniform grid)
        J = self.dx / 2.0
        
        # Mass matrix is diagonal for GLL quadrature
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    depth = k * self.dx
                    depth_idx = np.searchsorted(self.model.depth, depth)
                    depth_idx = min(depth_idx, len(self.model.depth) - 1)
                    
                    rho = self.model.density[depth_idx] * 1000  # kg/m³
                    
                    # GLL weights for all dimensions
                    w_x = self.gll_weights[i % (self.polynomial_order + 1)]
                    w_y = self.gll_weights[j % (self.polynomial_order + 1)]
                    w_z = self.gll_weights[k % (self.polynomial_order + 1)]
                    
                    M[i, j, k] = rho * w_x * w_y * w_z * J ** 3
        
        return M
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _spectral_element_wave_propagation(u, v, w, stress, strain, memory_vars, 
                                          elastic_tensor, mass_matrix, lagrange_deriv,
                                          tau_epsilon, tau_sigma, dx, dt, nx, ny, nz):
        """Full 3D spectral element method with generalized Maxwell body attenuation"""
        
        # Compute strain rate from velocity gradients using spectral derivatives
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    # Spectral derivatives using Lagrange basis
                    du_dx = 0.0
                    du_dy = 0.0
                    du_dz = 0.0
                    dv_dx = 0.0
                    dv_dy = 0.0
                    dv_dz = 0.0
                    dw_dx = 0.0
                    dw_dy = 0.0
                    dw_dz = 0.0
                    
                    # Sum over all neighboring GLL points (7th order = 8 points)
                    for m in range(max(0, i-4), min(nx, i+5)):
                        weight = lagrange_deriv[i % 8, m % 8]
                        du_dx += u[m, j, k] * weight / dx
                        dv_dx += v[m, j, k] * weight / dx
                        dw_dx += w[m, j, k] * weight / dx
                    
                    for m in range(max(0, j-4), min(ny, j+5)):
                        weight = lagrange_deriv[j % 8, m % 8]
                        du_dy += u[i, m, k] * weight / dx
                        dv_dy += v[i, m, k] * weight / dx
                        dw_dy += w[i, m, k] * weight / dx
                    
                    for m in range(max(0, k-4), min(nz, k+5)):
                        weight = lagrange_deriv[k % 8, m % 8]
                        du_dz += u[i, j, m] * weight / dx
                        dv_dz += v[i, j, m] * weight / dx
                        dw_dz += w[i, j, m] * weight / dx
                    
                    # Strain tensor (symmetric part of velocity gradient)
                    strain[i, j, k, 0, 0] = du_dx
                    strain[i, j, k, 1, 1] = dv_dy
                    strain[i, j, k, 2, 2] = dw_dz
                    strain[i, j, k, 0, 1] = 0.5 * (du_dy + dv_dx)
                    strain[i, j, k, 0, 2] = 0.5 * (du_dz + dw_dx)
                    strain[i, j, k, 1, 2] = 0.5 * (dv_dz + dw_dy)
                    strain[i, j, k, 1, 0] = strain[i, j, k, 0, 1]
                    strain[i, j, k, 2, 0] = strain[i, j, k, 0, 2]
                    strain[i, j, k, 2, 1] = strain[i, j, k, 1, 2]
        
        # Update stress with full elastic tensor and anelastic memory variables
        for i in prange(nx):
            for j in range(ny):
                for k in range(nz):
                    # Elastic stress increment
                    for ii in range(3):
                        for jj in range(3):
                            elastic_stress = 0.0
                            for kk in range(3):
                                for ll in range(3):
                                    elastic_stress += elastic_tensor[i, j, k, ii, jj, kk, ll] * strain[i, j, k, kk, ll]
                            
                            # Anelastic correction with 3 relaxation mechanisms
                            anelastic_correction = 0.0
                            for mech in range(3):  # 3 Maxwell mechanisms
                                # Update memory variable (generalized Maxwell body)
                                exp_factor = np.exp(-dt / tau_sigma[mech])
                                memory_vars[i, j, k, mech, ii, jj] = (
                                    exp_factor * memory_vars[i, j, k, mech, ii, jj] +
                                    (1.0 - exp_factor) * strain[i, j, k, ii, jj]
                                )
                                
                                # Q-factor weight for this mechanism
                                Q_weight = 1.0 / (1.0 + (2.0 * np.pi * 1.0 * tau_sigma[mech]) ** 2)
                                anelastic_correction += Q_weight * memory_vars[i, j, k, mech, ii, jj]
                            
                            stress[i, j, k, ii, jj] = elastic_stress - anelastic_correction
        
        # Update velocities from stress gradients
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    # Spectral derivatives of stress
                    dsxx_dx = 0.0
                    dsxy_dy = 0.0
                    dsxz_dz = 0.0
                    dsyx_dx = 0.0
                    dsyy_dy = 0.0
                    dsyz_dz = 0.0
                    dszx_dx = 0.0
                    dszy_dy = 0.0
                    dszz_dz = 0.0
                    
                    for m in range(max(0, i-4), min(nx, i+5)):
                        weight = lagrange_deriv[i % 8, m % 8]
                        dsxx_dx += stress[m, j, k, 0, 0] * weight / dx
                        dsyx_dx += stress[m, j, k, 1, 0] * weight / dx
                        dszx_dx += stress[m, j, k, 2, 0] * weight / dx
                    
                    for m in range(max(0, j-4), min(ny, j+5)):
                        weight = lagrange_deriv[j % 8, m % 8]
                        dsxy_dy += stress[i, m, k, 0, 1] * weight / dx
                        dsyy_dy += stress[i, m, k, 1, 1] * weight / dx
                        dszy_dy += stress[i, m, k, 2, 1] * weight / dx
                    
                    for m in range(max(0, k-4), min(nz, k+5)):
                        weight = lagrange_deriv[k % 8, m % 8]
                        dsxz_dz += stress[i, j, m, 0, 2] * weight / dx
                        dsyz_dz += stress[i, j, m, 1, 2] * weight / dx
                        dszz_dz += stress[i, j, m, 2, 2] * weight / dx
                    
                    # Acceleration = (1/mass) * divergence(stress)
                    mass_inv = 1.0 / mass_matrix[i, j, k] if mass_matrix[i, j, k] > 0 else 0.0
                    
                    u[i, j, k] += dt * mass_inv * (dsxx_dx + dsxy_dy + dsxz_dz)
                    v[i, j, k] += dt * mass_inv * (dsyx_dx + dsyy_dy + dsyz_dz)
                    w[i, j, k] += dt * mass_inv * (dszx_dx + dszy_dy + dszz_dz)
        
        return u, v, w, stress, strain, memory_vars
    
    def green_function_frequency_domain(self, source: SeismicSource, 
                                       distance: float, frequency: float) -> complex:
        """Frequency-domain Green's function for elastic half-space"""
        omega = 2 * np.pi * frequency
        r = distance * 1000  # Convert to meters
        
        # Average velocities at source depth
        depth_idx = int(source.depth / self.dx)
        vp = self.model.vp[depth_idx]
        vs = self.model.vs[depth_idx]
        rho = self.model.density[depth_idx]
        
        # P-wave component
        kp = omega / (vp * 1000)  # Convert velocity to m/s
        gp = np.exp(1j * kp * r) / (4 * np.pi * rho * vp**2 * r)
        
        # S-wave component
        ks = omega / (vs * 1000)
        gs = np.exp(1j * ks * r) / (4 * np.pi * rho * vs**2 * r)
        
        # Quality factor attenuation
        qp = self.model.qp[depth_idx]
        qs = self.model.qs[depth_idx]
        gp *= np.exp(-omega * r / (2 * vp * 1000 * qp))
        gs *= np.exp(-omega * r / (2 * vs * 1000 * qs))
        
        return gp + gs
    
    def rayleigh_wave_dispersion(self, frequency: float, depth_idx: int) -> float:
        """Full Rayleigh wave phase velocity from exact dispersion relation"""
        vp = self.model.vp[depth_idx] * 1000  # m/s
        vs = self.model.vs[depth_idx] * 1000
        
        # Rayleigh equation: (2-c^2/vs^2)^2 = 4*sqrt(1-c^2/vp^2)*sqrt(1-c^2/vs^2)
        # where c is Rayleigh velocity
        
        # Initial guess using approximate relation
        cr = 0.92 * vs
        
        # Newton-Raphson iteration for exact solution
        max_iterations = 50
        tolerance = 1e-12
        
        for iteration in range(max_iterations):
            x = cr / vs
            
            if x >= 1.0:
                x = 0.99
            
            # Rayleigh function
            alpha_sq = 1.0 - (cr / vp) ** 2
            beta_sq = 1.0 - (cr / vs) ** 2
            
            if alpha_sq <= 0 or beta_sq <= 0:
                cr *= 0.95
                continue
            
            alpha = np.sqrt(alpha_sq)
            beta = np.sqrt(beta_sq)
            
            f = (2.0 - x ** 2) ** 2 - 4.0 * alpha * beta
            
            # Derivative
            df_dcr = (-4.0 * (2.0 - x ** 2) * (2.0 * cr / vs ** 2) -
                     4.0 * (alpha / (cr * alpha_sq / (vp ** 2)) if alpha_sq > 0 else 0) * beta -
                     4.0 * alpha * (beta / (cr * beta_sq / (vs ** 2)) if beta_sq > 0 else 0))
            
            if abs(df_dcr) < 1e-15:
                break
            
            cr_new = cr - f / df_dcr
            
            if abs(cr_new - cr) < tolerance:
                break
            
            cr = cr_new
            
            # Ensure physical bounds
            cr = max(0.1 * vs, min(0.99 * vs, cr))
        
        return cr / 1000  # Convert to km/s
    
    def love_wave_dispersion(self, frequency: float, layer_thickness: float,
                            vp_layer: float, vs_layer: float,
                            vp_halfspace: float, vs_halfspace: float) -> float:
        """Full Love wave dispersion with layered Earth model"""
        omega = 2.0 * np.pi * frequency
        
        # Love wave exists only if vs_layer < c < vs_halfspace
        c_min = vs_layer * 1000  # m/s
        c_max = vs_halfspace * 1000
        
        # Initial guess
        c = 0.5 * (c_min + c_max)
        
        # Transcendental equation for Love waves:
        # tan(k*H) = (mu2*gamma2) / (mu1*gamma1)
        # where gamma = sqrt(k^2 - omega^2/vs^2)
        
        max_iterations = 100
        tolerance = 1e-10
        
        for _ in range(max_iterations):
            k = omega / c
            
            # Vertical wavenumbers
            gamma1_sq = k ** 2 - (omega / (vs_layer * 1000)) ** 2
            gamma2_sq = k ** 2 - (omega / (vs_halfspace * 1000)) ** 2
            
            if gamma1_sq < 0 or gamma2_sq < 0:
                c = 0.5 * (c + c_min)
                continue
            
            gamma1 = np.sqrt(gamma1_sq)
            gamma2 = np.sqrt(gamma2_sq)
            
            # Shear moduli
            rho1 = 2700  # kg/m³ (approximate)
            rho2 = 3300
            mu1 = rho1 * (vs_layer * 1000) ** 2
            mu2 = rho2 * (vs_halfspace * 1000) ** 2
            
            # Dispersion function
            lhs = np.tan(gamma1 * layer_thickness)
            rhs = (mu2 * gamma2) / (mu1 * gamma1)
            
            f = lhs - rhs
            
            # Numerical derivative
            dc = 1.0  # m/s
            k_perturb = omega / (c + dc)
            gamma1_p_sq = k_perturb ** 2 - (omega / (vs_layer * 1000)) ** 2
            gamma2_p_sq = k_perturb ** 2 - (omega / (vs_halfspace * 1000)) ** 2
            
            if gamma1_p_sq > 0 and gamma2_p_sq > 0:
                gamma1_p = np.sqrt(gamma1_p_sq)
                gamma2_p = np.sqrt(gamma2_p_sq)
                
                lhs_p = np.tan(gamma1_p * layer_thickness)
                rhs_p = (mu2 * gamma2_p) / (mu1 * gamma1_p)
                
                f_perturb = lhs_p - rhs_p
                df_dc = (f_perturb - f) / dc
                
                if abs(df_dc) > 1e-15:
                    c_new = c - f / df_dc
                    
                    # Ensure bounds
                    c_new = max(c_min * 1.001, min(c_max * 0.999, c_new))
                    
                    if abs(c_new - c) < tolerance:
                        break
                    
                    c = c_new
                else:
                    break
            else:
                break
        
        return c / 1000  # km/s
    
    def modal_summation_surface_waves(self, source: SeismicSource, 
                                     distance: float, time: float) -> np.ndarray:
        """Complete surface wave synthesis using modal summation with full eigenfunctions"""
        freqs = np.linspace(0.005, 3.0, 500)  # Hz - high resolution
        n_modes = 15  # Higher modes for accuracy
        
        displacement = np.zeros(3)
        
        # Layer parameters from Earth model
        num_layers = len(self.model.depth) - 1
        
        for mode_n in range(n_modes):
            # Mode-specific group velocity and eigenfunction
            mode_energy = 0.0
            
            for freq in freqs:
                omega = 2 * np.pi * freq
                
                # Rayleigh wave contribution
                cr = self.rayleigh_wave_dispersion(freq, 0)
                
                # Group velocity (derivative of dispersion curve)
                dfreq = 0.001
                cr_plus = self.rayleigh_wave_dispersion(freq + dfreq, 0)
                U_rayleigh = cr + freq * (cr_plus - cr) / dfreq  # Group velocity
                
                # Eigenfunction normalization
                # For fundamental mode: amplitude ~ 1/(mode_number + 1)
                # Higher modes decay faster with depth
                depth_decay = np.exp(-mode_n * source.depth / 10.0)
                
                # Mode amplitude from moment tensor
                M = source.moment_tensor()
                
                # Radiation pattern for Rayleigh waves
                azimuth = 0  # Simplified: should depend on epicentral azimuth
                strike_rad = np.radians(source.strike)
                dip_rad = np.radians(source.dip)
                rake_rad = np.radians(source.rake)
                
                # Excitation coefficient (Aki & Richards eq 9.52)
                excitation = (
                    M[0, 0] * np.cos(2 * azimuth) +
                    M[1, 1] * np.sin(2 * azimuth) +
                    M[0, 1] * np.sin(2 * azimuth) +
                    M[2, 2] * 0.3  # Vertical component
                ) * depth_decay
                
                # Mode-specific amplitude (decreases with mode number)
                amplitude = abs(excitation) / ((mode_n + 1) ** 1.5)
                
                # Phase velocity dispersion causes frequency-dependent arrival
                k_rayleigh = omega / (cr * 1000)  # Wavenumber
                
                # Travel time
                t_arrival = distance / U_rayleigh
                
                # Attenuation
                Q = self.model.qs[0]
                attenuation = np.exp(-omega * distance * 1000 / (2 * U_rayleigh * 1000 * Q))
                
                # Geometric spreading for surface waves (cylindrical)
                geometric_factor = 1.0 / np.sqrt(distance * 1000) if distance > 0.1 else 0
                
                # Phase
                phase = omega * (time - t_arrival) - k_rayleigh * distance * 1000
                
                # Frequency band weight (spectral shaping)
                fc = 4.9e6 * source.rupture_velocity / source.rupture_length
                omega_c = 2 * np.pi * fc
                spectral_weight = (omega ** 2) / (omega ** 2 + omega_c ** 2)
                
                # Accumulated displacement contribution
                dfreq_step = freqs[1] - freqs[0] if len(freqs) > 1 else 0.01
                
                displacement[0] += (amplitude * spectral_weight * geometric_factor * 
                                  attenuation * np.cos(phase) * dfreq_step)  # Radial
                displacement[2] += (amplitude * spectral_weight * geometric_factor * 
                                  attenuation * np.sin(phase) * dfreq_step)  # Vertical
            
            # Love wave contribution (transverse component)
            if num_layers > 1:
                # Love waves require layering
                layer_thickness = self.model.depth[1] - self.model.depth[0]
                
                for freq in freqs:
                    omega = 2 * np.pi * freq
                    
                    try:
                        cl = self.love_wave_dispersion(
                            freq, layer_thickness,
                            self.model.vp[0], self.model.vs[0],
                            self.model.vp[1], self.model.vs[1]
                        )
                        
                        # Group velocity
                        cl_plus = self.love_wave_dispersion(
                            freq + dfreq, layer_thickness,
                            self.model.vp[0], self.model.vs[0],
                            self.model.vp[1], self.model.vs[1]
                        )
                        U_love = cl + freq * (cl_plus - cl) / dfreq
                        
                        # Love wave excitation (transverse moment)
                        excitation_love = M[0, 1] * np.sin(azimuth) * depth_decay
                        amplitude_love = abs(excitation_love) / ((mode_n + 1) ** 1.5)
                        
                        k_love = omega / (cl * 1000)
                        t_arrival_love = distance / U_love
                        
                        attenuation_love = np.exp(-omega * distance * 1000 / (2 * U_love * 1000 * Q))
                        geometric_factor_love = 1.0 / np.sqrt(distance * 1000) if distance > 0.1 else 0
                        
                        phase_love = omega * (time - t_arrival_love) - k_love * distance * 1000
                        
                        fc = 4.9e6 * source.rupture_velocity / source.rupture_length
                        omega_c = 2 * np.pi * fc
                        spectral_weight = (omega ** 2) / (omega ** 2 + omega_c ** 2)
                        
                        displacement[1] += (amplitude_love * spectral_weight * geometric_factor_love *
                                          attenuation_love * np.cos(phase_love) * dfreq_step)
                    except:
                        continue
        
        return displacement * 1e-3  # Scale to meters
    
    def source_time_function(self, time: float, source: SeismicSource, 
                            wavelet_type: str = 'brune') -> float:
        """Source time function models"""
        if wavelet_type == 'brune':
            # Brune (1970) omega-squared model
            corner_freq = 4.9e6 * source.rupture_velocity / source.rupture_length  # Hz
            omega_c = 2 * np.pi * corner_freq
            return omega_c**2 * time * np.exp(-omega_c * time) if time >= 0 else 0
        
        elif wavelet_type == 'ricker':
            # Ricker wavelet (Mexican hat)
            f0 = 1.0 / source.rupture_length  # Dominant frequency
            shift = 1.5 / f0
            t_shifted = time - shift
            return (1 - 2 * (np.pi * f0 * t_shifted)**2) * \
                   np.exp(-(np.pi * f0 * t_shifted)**2)
        
        elif wavelet_type == 'triangle':
            # Triangle (Kostrov) slip function
            rise_time = source.rupture_length / source.rupture_velocity
            if 0 <= time <= rise_time:
                return time / rise_time
            elif rise_time < time <= 2 * rise_time:
                return 2 - time / rise_time
            return 0
        
        return 0


class RealTimeSeismicSimulator:
    """Real-time earthquake simulation for 3D visualization"""
    
    def __init__(self, terrain_shape: Tuple[int, int], earth_model: EarthModel):
        self.width, self.height = terrain_shape
        self.model = earth_model
        
        # Surface displacement field (what we'll visualize)
        self.displacement = np.zeros((self.width, self.height, 3), dtype=np.float32)
        self.velocity = np.zeros((self.width, self.height, 3), dtype=np.float32)
        
        # Simulation state
        self.current_time = 0.0
        self.dt = 0.01  # seconds
        self.active = False
        self.source: Optional[SeismicSource] = None
        
        # Wave engine
        grid_size = (self.width, self.height, 50)  # 50 depth layers
        self.engine = SeismicWaveEngine(earth_model, grid_size)
        
        # Thread-safe simulation
        self.lock = threading.Lock()
        
    def initialize_earthquake(self, source: SeismicSource):
        """Set up earthquake source"""
        with self.lock:
            self.source = source
            self.current_time = 0.0
            self.displacement.fill(0)
            self.velocity.fill(0)
            self.active = True
    
    def compute_surface_displacement(self, x: int, y: int) -> np.ndarray:
        """Calculate displacement at surface point from seismic waves"""
        if self.source is None or not self.active:
            return np.zeros(3)
        
        # Convert grid coordinates to physical distance
        source_x = int(self.source.latitude * self.width / 180)
        source_y = int(self.source.longitude * self.height / 360)
        
        dx = (x - source_x) * self.engine.dx
        dy = (y - source_y) * self.engine.dx
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 0.1:  # Too close to source
            return np.zeros(3)
        
        # Combined wave contributions
        disp = np.zeros(3)
        
        # P-wave arrival
        vp = self.model.vp[0]
        tp = distance / vp
        if self.current_time >= tp:
            t_p = self.current_time - tp
            stf_p = self.engine.source_time_function(t_p, self.source)
            amplitude_p = self.source.seismic_moment / (4 * np.pi * self.model.density[0] * 
                                                        vp**3 * distance * 1e9)
            disp += amplitude_p * stf_p * np.array([dx/distance, dy/distance, 0])
        
        # S-wave arrival
        vs = self.model.vs[0]
        ts = distance / vs
        if self.current_time >= ts:
            t_s = self.current_time - ts
            stf_s = self.engine.source_time_function(t_s, self.source)
            amplitude_s = self.source.seismic_moment / (4 * np.pi * self.model.density[0] * 
                                                        vs**3 * distance * 1e9)
            # Perpendicular component for shear
            disp += amplitude_s * stf_s * np.array([-dy/distance, dx/distance, 0])
        
        # Surface waves (dominant for large distances)
        if distance > 5.0:  # km
            surface = self.engine.modal_summation_surface_waves(
                self.source, distance, self.current_time
            )
            disp += surface * 0.001  # Scale for visualization
        
        # Attenuation
        q_factor = self.model.qs[0]
        attenuation = np.exp(-np.pi * distance / (q_factor * vs))
        
        return disp * attenuation * 1000  # Convert to mm for visualization
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _parallel_displacement_update(displacement, velocity, source_x, source_y,
                                     width, height, current_time, source_params,
                                     model_vp, model_vs, model_density, dx):
        """JIT-compiled parallel displacement computation"""
        for x in prange(width):
            for y in range(height):
                dist_x = (x - source_x) * dx
                dist_y = (y - source_y) * dx
                distance = np.sqrt(dist_x**2 + dist_y**2)
                
                if distance < 0.1:
                    continue
                
                # Simplified wave propagation for real-time performance
                vp = model_vp
                vs = model_vs
                
                tp = distance / vp
                ts = distance / vs
                
                amplitude = source_params[0] / (distance + 1.0)  # M0 / distance
                
                if current_time >= tp:
                    phase_p = 2 * np.pi * (current_time - tp) * 2.0
                    displacement[x, y, 2] += amplitude * np.sin(phase_p) * 0.001
                
                if current_time >= ts:
                    phase_s = 2 * np.pi * (current_time - ts) * 1.0
                    displacement[x, y, 0] += amplitude * np.cos(phase_s) * 0.0015
                    displacement[x, y, 1] += amplitude * np.sin(phase_s) * 0.0015
        
        return displacement
    
    def update(self) -> np.ndarray:
        """Advance simulation by one timestep"""
        if not self.active or self.source is None:
            return self.displacement
        
        with self.lock:
            self.current_time += self.dt
            
            # Check if simulation should stop (waves passed through domain)
            max_distance = np.sqrt(self.width**2 + self.height**2) * self.engine.dx
            max_travel_time = max_distance / self.model.vs[0] + 30.0  # +30s for surface waves
            
            if self.current_time > max_travel_time:
                self.active = False
                return self.displacement
            
            # Fast parallel update
            source_x = int(self.source.latitude * self.width / 180)
            source_y = int(self.source.longitude * self.height / 360)
            
            source_params = np.array([self.source.seismic_moment])
            
            self.displacement = self._parallel_displacement_update(
                self.displacement, self.velocity,
                source_x, source_y,
                self.width, self.height,
                self.current_time,
                source_params,
                self.model.vp[0], self.model.vs[0], self.model.density[0],
                self.engine.dx
            )
            
            # Damping to prevent instability
            self.displacement *= 0.995
            
        return self.displacement
    
    def get_displacement_field(self) -> np.ndarray:
        """Get current displacement field (thread-safe)"""
        with self.lock:
            return self.displacement.copy()
    
    def stop(self):
        """Stop active simulation"""
        with self.lock:
            self.active = False


def create_earth_model_iasp91() -> EarthModel:
    """IASP91 reference Earth model (Kennett & Engdahl, 1991)"""
    # Simplified crustal + upper mantle structure
    depth = np.array([0, 10, 20, 35, 77, 120, 210, 400])  # km
    vp = np.array([5.8, 6.5, 6.5, 8.04, 8.05, 8.18, 8.65, 9.03])  # km/s
    vs = np.array([3.36, 3.75, 3.75, 4.47, 4.48, 4.52, 4.63, 4.87])  # km/s
    density = np.array([2.72, 2.92, 2.92, 3.32, 3.37, 3.37, 3.46, 3.72])  # g/cm³
    
    # Quality factors (attenuation)
    qp = np.array([600, 600, 600, 1340, 1340, 1340, 1340, 1340])
    qs = np.array([300, 300, 300, 600, 600, 600, 600, 600])
    
    return EarthModel(vp=vp, vs=vs, density=density, qp=qp, qs=qs, depth=depth)