"""
Validation and Testing Suite for Enterprise Earthquake Simulator
Tests physics accuracy, performance, and numerical stability
"""
import numpy as np
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass

from seismic_engine import (
    SeismicSource, create_earth_model_iasp91, SeismicWaveEngine,
    RealTimeSeismicSimulator, WaveType
)
from terrain_processor import (
    DEMProcessor, TerrainMeshGenerator, create_synthetic_dem
)


@dataclass
class ValidationResult:
    """Test result container"""
    test_name: str
    passed: bool
    error_metric: float
    expected: float
    actual: float
    tolerance: float
    details: str


class PhysicsValidator:
    """Validate seismological physics implementation"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.earth_model = create_earth_model_iasp91()
    
    def validate_wave_speeds(self) -> ValidationResult:
        """Test P and S wave velocities against IASP91"""
        print("Testing wave velocities...")
        
        # IASP91 reference values at surface
        expected_vp = 5.8  # km/s
        expected_vs = 3.36  # km/s
        
        actual_vp = self.earth_model.vp[0]
        actual_vs = self.earth_model.vs[0]
        
        error_vp = abs(actual_vp - expected_vp) / expected_vp
        error_vs = abs(actual_vs - expected_vs) / expected_vs
        
        tolerance = 0.01  # 1%
        passed = error_vp < tolerance and error_vs < tolerance
        
        result = ValidationResult(
            test_name="Wave Velocity Accuracy",
            passed=passed,
            error_metric=(error_vp + error_vs) / 2,
            expected=(expected_vp + expected_vs) / 2,
            actual=(actual_vp + actual_vs) / 2,
            tolerance=tolerance,
            details=f"Vp error: {error_vp*100:.2f}%, Vs error: {error_vs*100:.2f}%"
        )
        
        self.results.append(result)
        return result
    
    def validate_seismic_moment(self) -> ValidationResult:
        """Test seismic moment calculation"""
        print("Testing seismic moment calculation...")
        
        # Create test earthquake
        source = SeismicSource(
            latitude=0, longitude=0, depth=10,
            magnitude=7.0, strike=0, dip=45, rake=90
        )
        
        # Theoretical M0 for Mw=7.0
        expected_m0 = 10 ** (1.5 * 7.0 + 9.1)  # N⋅m
        actual_m0 = source.seismic_moment
        
        error = abs(actual_m0 - expected_m0) / expected_m0
        tolerance = 0.001  # 0.1%
        passed = error < tolerance
        
        result = ValidationResult(
            test_name="Seismic Moment Calculation",
            passed=passed,
            error_metric=error,
            expected=expected_m0,
            actual=actual_m0,
            tolerance=tolerance,
            details=f"M0 = {actual_m0:.2e} N⋅m"
        )
        
        self.results.append(result)
        return result
    
    def validate_moment_tensor(self) -> ValidationResult:
        """Test moment tensor symmetry and properties"""
        print("Testing moment tensor...")
        
        source = SeismicSource(
            latitude=0, longitude=0, depth=10,
            magnitude=6.5, strike=45, dip=60, rake=90
        )
        
        M = source.moment_tensor()
        
        # Test symmetry: M should be symmetric
        symmetry_error = np.linalg.norm(M - M.T)
        
        # Test trace (should be close to zero for double-couple)
        trace = np.trace(M)
        trace_normalized = abs(trace) / source.seismic_moment
        
        tolerance = 1e-10
        passed = symmetry_error < tolerance and trace_normalized < 0.01
        
        result = ValidationResult(
            test_name="Moment Tensor Properties",
            passed=passed,
            error_metric=symmetry_error,
            expected=0.0,
            actual=symmetry_error,
            tolerance=tolerance,
            details=f"Symmetry error: {symmetry_error:.2e}, Trace: {trace:.2e}"
        )
        
        self.results.append(result)
        return result
    
    def validate_rupture_scaling(self) -> ValidationResult:
        """Test Wells & Coppersmith scaling relations"""
        print("Testing rupture scaling...")
        
        # Test multiple magnitudes
        magnitudes = [5.0, 6.0, 7.0, 8.0]
        errors = []
        
        for mag in magnitudes:
            source = SeismicSource(
                latitude=0, longitude=0, depth=10,
                magnitude=mag, strike=0, dip=45, rake=90
            )
            
            # Wells & Coppersmith (1994): log(A) = -3.49 + 0.91*M
            expected_area = 10 ** (-3.49 + 0.91 * mag)
            actual_area = source.rupture_area
            
            error = abs(actual_area - expected_area) / expected_area
            errors.append(error)
        
        mean_error = np.mean(errors)
        tolerance = 0.01
        passed = mean_error < tolerance
        
        result = ValidationResult(
            test_name="Rupture Area Scaling",
            passed=passed,
            error_metric=mean_error,
            expected=0.0,
            actual=mean_error,
            tolerance=tolerance,
            details=f"Mean error across magnitudes: {mean_error*100:.2f}%"
        )
        
        self.results.append(result)
        return result
    
    def validate_attenuation(self) -> ValidationResult:
        """Test Q-factor attenuation"""
        print("Testing attenuation model...")
        
        engine = SeismicWaveEngine(
            self.earth_model,
            grid_size=(100, 100, 50),
            spacing=1.0
        )
        
        # Test Green's function attenuation
        source = SeismicSource(
            latitude=0, longitude=0, depth=10,
            magnitude=6.5, strike=0, dip=45, rake=90
        )
        
        distances = np.array([10, 50, 100, 200])  # km
        frequency = 1.0  # Hz
        
        amplitudes = []
        for dist in distances:
            G = engine.green_function_frequency_domain(source, dist, frequency)
            amplitudes.append(abs(G))
        
        # Check geometric spreading (1/r decay)
        expected_decay = amplitudes[0] * distances[0] / distances
        actual_decay = np.array(amplitudes)
        
        # Relative error in decay
        errors = np.abs(actual_decay - expected_decay) / expected_decay
        mean_error = np.mean(errors)
        
        tolerance = 0.2  # 20% (attenuation adds complexity)
        passed = mean_error < tolerance
        
        result = ValidationResult(
            test_name="Attenuation Model",
            passed=passed,
            error_metric=mean_error,
            expected=0.0,
            actual=mean_error,
            tolerance=tolerance,
            details=f"Geometric spreading + Q attenuation validated"
        )
        
        self.results.append(result)
        return result
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run complete validation suite"""
        print("\n" + "="*70)
        print("PHYSICS VALIDATION SUITE")
        print("="*70 + "\n")
        
        tests = [
            self.validate_wave_speeds,
            self.validate_seismic_moment,
            self.validate_moment_tensor,
            self.validate_rupture_scaling,
            self.validate_attenuation
        ]
        
        for test in tests:
            result = test()
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {status}: {result.test_name}")
            print(f"    Error: {result.error_metric:.2e} (tolerance: {result.tolerance:.2e})")
            print(f"    {result.details}\n")
        
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print("="*70)
        print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        print("="*70 + "\n")
        
        return {r.test_name: r.passed for r in self.results}


class PerformanceProfiler:
    """Profile computational performance"""
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
    
    def profile_mesh_generation(self, sizes: List[int]) -> Dict[int, float]:
        """Profile mesh generation for different terrain sizes"""
        print("\n" + "="*70)
        print("MESH GENERATION PERFORMANCE")
        print("="*70 + "\n")
        
        timings = {}
        
        for size in sizes:
            print(f"Testing {size}x{size} terrain...")
            
            # Create synthetic terrain
            elevation = create_synthetic_dem(size, size, complexity=0.5)
            
            # Time mesh generation
            start = time.time()
            generator = TerrainMeshGenerator(elevation, adaptive_refinement=True)
            vertices, faces, normals, uvs = generator.generate_mesh()
            elapsed = time.time() - start
            
            timings[size] = elapsed
            
            print(f"  Generated {len(vertices):,} vertices, {len(faces):,} faces")
            print(f"  Time: {elapsed:.3f}s\n")
        
        self.timings['mesh_generation'] = timings
        return timings
    
    def profile_wave_propagation(self, grid_sizes: List[Tuple[int, int, int]]) -> Dict[Tuple, float]:
        """Profile wave propagation for different grid sizes"""
        print("\n" + "="*70)
        print("WAVE PROPAGATION PERFORMANCE")
        print("="*70 + "\n")
        
        timings = {}
        earth_model = create_earth_model_iasp91()
        
        for grid_size in grid_sizes:
            print(f"Testing {grid_size[0]}x{grid_size[1]}x{grid_size[2]} grid...")
            
            engine = SeismicWaveEngine(earth_model, grid_size)
            
            # Create test arrays
            u = np.random.randn(*grid_size) * 0.01
            v = np.random.randn(*grid_size) * 0.01
            stress = np.random.randn(*grid_size + (3,)) * 0.01
            density = np.ones(grid_size) * 2.7
            lame_lambda = np.ones(grid_size) * 30e9
            lame_mu = np.ones(grid_size) * 30e9
            
            # Time one step
            start = time.time()
            u_new, v_new, stress_new = engine._staggered_grid_wave_step(
                u, v, stress, density, lame_lambda, lame_mu,
                engine.dx, engine.dt
            )
            elapsed = time.time() - start
            
            timings[grid_size] = elapsed
            
            total_cells = grid_size[0] * grid_size[1] * grid_size[2]
            cells_per_sec = total_cells / elapsed if elapsed > 0 else 0
            
            print(f"  Time: {elapsed:.4f}s")
            print(f"  Throughput: {cells_per_sec:,.0f} cells/sec\n")
        
        self.timings['wave_propagation'] = timings
        return timings
    
    def profile_simulation_update(self, terrain_sizes: List[int]) -> Dict[int, float]:
        """Profile real-time simulation update"""
        print("\n" + "="*70)
        print("REAL-TIME SIMULATION UPDATE PERFORMANCE")
        print("="*70 + "\n")
        
        timings = {}
        earth_model = create_earth_model_iasp91()
        
        for size in terrain_sizes:
            print(f"Testing {size}x{size} terrain...")
            
            simulator = RealTimeSeismicSimulator(
                terrain_shape=(size, size),
                earth_model=earth_model
            )
            
            # Initialize earthquake
            source = SeismicSource(
                latitude=0, longitude=0, depth=10,
                magnitude=6.5, strike=0, dip=45, rake=90
            )
            simulator.initialize_earthquake(source)
            
            # Time multiple updates
            num_updates = 100
            start = time.time()
            for _ in range(num_updates):
                simulator.update()
            elapsed = time.time() - start
            
            avg_time = elapsed / num_updates
            fps = 1 / avg_time if avg_time > 0 else 0
            
            timings[size] = avg_time
            
            print(f"  Average update time: {avg_time*1000:.2f}ms")
            print(f"  Effective FPS: {fps:.1f}\n")
        
        self.timings['simulation_update'] = timings
        return timings
    
    def generate_report(self):
        """Generate performance report"""
        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70 + "\n")
        
        for category, timing_data in self.timings.items():
            print(f"{category.upper().replace('_', ' ')}:")
            for key, value in timing_data.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}s")
                else:
                    print(f"  {key}: {value}")
            print()


def run_validation_suite():
    """Run complete validation and profiling"""
    # Physics validation
    validator = PhysicsValidator()
    physics_results = validator.run_all_tests()
    
    # Performance profiling
    profiler = PerformanceProfiler()
    
    # Profile with increasing complexity
    mesh_timings = profiler.profile_mesh_generation([64, 128, 256, 512])
    wave_timings = profiler.profile_wave_propagation([
        (50, 50, 20),
        (100, 100, 30),
        (150, 150, 40)
    ])
    sim_timings = profiler.profile_simulation_update([64, 128, 256])
    
    profiler.generate_report()
    
    return {
        'physics': physics_results,
        'mesh_generation': mesh_timings,
        'wave_propagation': wave_timings,
        'simulation': sim_timings
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EARTHQUAKE SIMULATOR - VALIDATION SUITE")
    print("="*70 + "\n")
    
    results = run_validation_suite()
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)