"""
Enterprise Earthquake Simulator - Main Application
Unified entry point with advanced configuration and parameter management
"""
import sys
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import numpy as np

# Import core modules
from seismic_engine import (
    SeismicSource, EarthModel, create_earth_model_iasp91,
    RealTimeSeismicSimulator, FaultType
)
from terrain_processor import (
    DEMProcessor, TerrainMeshGenerator, TerrainSeismicIntegrator,
    create_synthetic_dem
)
from visualization_engine import (
    EarthquakeVisualization3D, create_visualization_from_dem
)


@dataclass
class SimulationConfig:
    """Complete simulation configuration"""
    # Terrain parameters
    dem_path: Optional[str] = None
    terrain_resolution: float = 30.0  # meters
    terrain_size: tuple = (256, 256)  # if synthetic
    vertical_exaggeration: float = 2.0
    mesh_simplification: float = 0.5
    
    # Earthquake parameters
    magnitude: float = 6.5
    depth: float = 10.0  # km
    latitude: float = 0.0
    longitude: float = 0.0
    strike: float = 0.0  # degrees
    dip: float = 45.0
    rake: float = 90.0
    rupture_velocity: float = 2.8  # km/s
    stress_drop: float = 3e6  # Pa
    
    # Earth model
    earth_model_type: str = "iasp91"  # iasp91, prem, ak135
    
    # Simulation parameters
    time_step: float = 0.01  # seconds
    max_simulation_time: float = 120.0  # seconds
    use_gpu_deformation: bool = True
    
    # Visualization parameters
    window_size: tuple = (1920, 1080)
    camera_distance: float = 25.0
    enable_shadows: bool = True
    enable_fog: bool = True
    
    # Advanced physics
    enable_anelastic_attenuation: bool = True
    enable_surface_waves: bool = True
    enable_modal_summation: bool = True
    quality_factor_scaling: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """Create from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: str) -> 'SimulationConfig':
        """Load from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_json(self, path: str):
        """Save to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class EarthquakeSimulatorApp:
    """Main application orchestrator"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.earth_model: Optional[EarthModel] = None
        self.terrain_elevation: Optional[np.ndarray] = None
        self.terrain_mesh: Optional[Dict[str, np.ndarray]] = None
        self.simulator: Optional[RealTimeSeismicSimulator] = None
        self.visualization: Optional[EarthquakeVisualization3D] = None
        
        print("=" * 70)
        print("ENTERPRISE EARTHQUAKE SIMULATOR")
        print("Advanced Seismic Wave Propagation & Terrain Deformation")
        print("=" * 70)
        print()
    
    def initialize(self):
        """Initialize all components"""
        print("Initializing simulation components...")
        
        # Load Earth model
        print(f"  ✓ Loading {self.config.earth_model_type.upper()} Earth model...")
        self.earth_model = self._create_earth_model()
        
        # Load or create terrain
        print("  ✓ Processing terrain data...")
        self.terrain_elevation = self._load_terrain()
        
        # Generate mesh
        print("  ✓ Generating 3D mesh...")
        self.terrain_mesh = self._generate_mesh()
        
        # Initialize seismic simulator
        print("  ✓ Initializing seismic wave engine...")
        self.simulator = RealTimeSeismicSimulator(
            terrain_shape=self.terrain_elevation.shape,
            earth_model=self.earth_model
        )
        
        print()
        print("Initialization complete!")
        print(f"  • Terrain size: {self.terrain_elevation.shape}")
        print(f"  • Mesh vertices: {len(self.terrain_mesh['vertices'])}")
        print(f"  • Mesh faces: {len(self.terrain_mesh['faces'])}")
        print(f"  • Earth model layers: {len(self.earth_model.depth)}")
        print()
    
    def _create_earth_model(self) -> EarthModel:
        """Create Earth velocity model"""
        if self.config.earth_model_type == "iasp91":
            model = create_earth_model_iasp91()
        elif self.config.earth_model_type == "prem":
            # PREM model (simplified)
            depth = np.array([0, 15, 24.4, 40, 60, 80, 115, 150, 220, 400])
            vp = np.array([5.8, 6.8, 6.8, 8.11, 8.15, 8.18, 8.35, 8.56, 8.85, 9.13])
            vs = np.array([3.2, 3.9, 3.9, 4.49, 4.51, 4.52, 4.61, 4.78, 4.94, 5.21])
            density = np.array([2.6, 2.9, 2.9, 3.38, 3.41, 3.44, 3.52, 3.66, 3.85, 4.08])
            qp = np.array([600, 600, 600, 1340, 1340, 1340, 1340, 1340, 1340, 1340])
            qs = np.array([300, 300, 300, 600, 600, 600, 600, 600, 600, 600])
            model = EarthModel(vp=vp, vs=vs, density=density, qp=qp, qs=qs, depth=depth)
        elif self.config.earth_model_type == "ak135":
            # AK135 model (simplified)
            depth = np.array([0, 20, 35, 77.5, 120, 165, 210, 260, 310, 410])
            vp = np.array([5.8, 6.5, 8.04, 8.05, 8.18, 8.37, 8.60, 8.85, 9.03, 9.36])
            vs = np.array([3.46, 3.85, 4.48, 4.49, 4.52, 4.60, 4.70, 4.82, 4.91, 5.08])
            density = np.array([2.72, 2.92, 3.32, 3.37, 3.37, 3.46, 3.54, 3.63, 3.72, 3.85])
            qp = np.array([600, 600, 1340, 1340, 1340, 1340, 1340, 1340, 1340, 1340])
            qs = np.array([300, 300, 600, 600, 600, 600, 600, 600, 600, 600])
            model = EarthModel(vp=vp, vs=vs, density=density, qp=qp, qs=qs, depth=depth)
        else:
            print(f"Warning: Unknown model '{self.config.earth_model_type}', using IASP91")
            model = create_earth_model_iasp91()
        
        # Apply quality factor scaling
        if self.config.quality_factor_scaling != 1.0:
            model.qp *= self.config.quality_factor_scaling
            model.qs *= self.config.quality_factor_scaling
        
        return model
    
    def _load_terrain(self) -> np.ndarray:
        """Load terrain from DEM or create synthetic"""
        if self.config.dem_path and Path(self.config.dem_path).exists():
            print(f"    Loading DEM: {self.config.dem_path}")
            processor = DEMProcessor(
                self.config.dem_path,
                target_resolution=self.config.terrain_resolution
            )
            elevation = processor.load_dem()
            
            if processor.metadata:
                print(f"    Elevation range: {processor.metadata.elevation_range[0]:.1f}m - "
                      f"{processor.metadata.elevation_range[1]:.1f}m")
                print(f"    Mean slope: {processor.metadata.mean_slope:.2f}°")
        else:
            print(f"    Creating synthetic terrain: {self.config.terrain_size}")
            elevation = create_synthetic_dem(
                self.config.terrain_size[0],
                self.config.terrain_size[1],
                complexity=0.6
            )
        
        return elevation
    
    def _generate_mesh(self) -> Dict[str, np.ndarray]:
        """Generate 3D mesh from elevation data"""
        generator = TerrainMeshGenerator(
            self.terrain_elevation,
            vertical_exaggeration=self.config.vertical_exaggeration,
            adaptive_refinement=True  # Use adaptive refinement based on terrain complexity
        )
        
        vertices, faces, normals, uvs = generator.generate_mesh()
        
        return {
            'vertices': vertices,
            'faces': faces,
            'normals': normals,
            'uvs': uvs
        }
    
    def create_earthquake_source(self) -> SeismicSource:
        """Create earthquake source from configuration"""
        source = SeismicSource(
            latitude=self.config.latitude,
            longitude=self.config.longitude,
            depth=self.config.depth,
            magnitude=self.config.magnitude,
            strike=self.config.strike,
            dip=self.config.dip,
            rake=self.config.rake,
            rupture_velocity=self.config.rupture_velocity,
            stress_drop=self.config.stress_drop
        )
        
        print()
        print("Earthquake Source Parameters:")
        print(f"  • Magnitude (Mw): {source.magnitude}")
        print(f"  • Depth: {source.depth} km")
        print(f"  • Location: ({source.latitude:.2f}°, {source.longitude:.2f}°)")
        print(f"  • Fault geometry: Strike={source.strike:.0f}° Dip={source.dip:.0f}° Rake={source.rake:.0f}°")
        print(f"  • Seismic moment: {source.seismic_moment:.2e} N⋅m")
        print(f"  • Rupture area: {source.rupture_area:.1f} km²")
        print(f"  • Rupture length: {source.rupture_length:.1f} km")
        print()
        
        return source
    
    def run_visualization(self):
        """Launch 3D visualization"""
        print("Launching 3D visualization...")
        print()
        print("Controls:")
        print("  • SPACE: Trigger earthquake")
        print("  • R: Reset simulation")
        print("  • G: Toggle grid")
        print("  • WASD: Move camera")
        print("  • Mouse: Rotate view")
        print("  • Scroll: Zoom")
        print()
        print("Starting Ursina engine...")
        print("=" * 70)
        print()
        
        self.visualization = EarthquakeVisualization3D(
            terrain_vertices=self.terrain_mesh['vertices'],
            terrain_faces=self.terrain_mesh['faces'],
            terrain_normals=self.terrain_mesh['normals'],
            terrain_uvs=self.terrain_mesh['uvs'],
            seismic_simulator=self.simulator,
            window_size=self.config.window_size
        )
        
        self.visualization.run()
    
    def run_batch_simulation(self, output_dir: str = "./output"):
        """Run headless batch simulation"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Running batch simulation (headless mode)...")
        
        # Create earthquake source
        source = self.create_earthquake_source()
        
        # Initialize simulation
        self.simulator.initialize_earthquake(source)
        
        # Time-stepping loop
        time_steps = int(self.config.max_simulation_time / self.config.time_step)
        output_interval = max(1, time_steps // 100)  # 100 outputs max
        
        print(f"Simulating {time_steps} time steps ({self.config.max_simulation_time}s)...")
        
        for step in range(time_steps):
            displacement = self.simulator.update()
            
            # Save outputs periodically
            if step % output_interval == 0:
                progress = (step / time_steps) * 100
                print(f"  Progress: {progress:.1f}% (t={step * self.config.time_step:.2f}s)")
                
                # Save displacement field
                output_file = output_path / f"displacement_{step:06d}.npy"
                np.save(output_file, displacement)
        
        print()
        print(f"Simulation complete! Outputs saved to: {output_path}")
        print(f"Total output files: {len(list(output_path.glob('displacement_*.npy')))}")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Enterprise Earthquake Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default synthetic terrain
  python main.py
  
  # Run with DEM file
  python main.py --dem terrain.tif --resolution 30
  
  # Run with custom earthquake parameters
  python main.py --magnitude 7.5 --depth 20 --strike 45 --dip 60 --rake -90
  
  # Run batch simulation (headless)
  python main.py --batch --output ./results
  
  # Load configuration from file
  python main.py --config simulation.json
        """
    )
    
    # Configuration
    parser.add_argument('--config', type=str, help='Load configuration from JSON file')
    
    # Terrain
    parser.add_argument('--dem', type=str, help='Path to DEM file (.tif)')
    parser.add_argument('--resolution', type=float, default=30.0, help='Terrain resolution (meters)')
    parser.add_argument('--terrain-size', type=int, nargs=2, default=[256, 256], 
                       help='Synthetic terrain size (if no DEM)')
    
    # Earthquake
    parser.add_argument('--magnitude', type=float, default=6.5, help='Moment magnitude (Mw)')
    parser.add_argument('--depth', type=float, default=10.0, help='Hypocenter depth (km)')
    parser.add_argument('--strike', type=float, default=0.0, help='Fault strike (degrees)')
    parser.add_argument('--dip', type=float, default=45.0, help='Fault dip (degrees)')
    parser.add_argument('--rake', type=float, default=90.0, help='Fault rake (degrees)')
    
    # Earth model
    parser.add_argument('--earth-model', type=str, default='iasp91',
                       choices=['iasp91', 'prem', 'ak135'],
                       help='Earth velocity model')
    
    # Simulation mode
    parser.add_argument('--batch', action='store_true', help='Run batch simulation (headless)')
    parser.add_argument('--output', type=str, default='./output', help='Output directory for batch mode')
    parser.add_argument('--max-time', type=float, default=120.0, help='Maximum simulation time (seconds)')
    
    # Visualization
    parser.add_argument('--window-size', type=int, nargs=2, default=[1920, 1080],
                       help='Window size (width height)')
    parser.add_argument('--gpu', action='store_true', default=True, 
                       help='Use GPU vertex shaders (default: True)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU-based deformation')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Load or create configuration
    if args.config:
        config = SimulationConfig.from_json(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        config = SimulationConfig(
            dem_path=args.dem,
            terrain_resolution=args.resolution,
            terrain_size=tuple(args.terrain_size),
            magnitude=args.magnitude,
            depth=args.depth,
            strike=args.strike,
            dip=args.dip,
            rake=args.rake,
            earth_model_type=args.earth_model,
            max_simulation_time=args.max_time,
            window_size=tuple(args.window_size),
            use_gpu_deformation=not args.cpu
        )
    
    # Create application
    app = EarthquakeSimulatorApp(config)
    
    try:
        # Initialize
        app.initialize()
        
        # Run appropriate mode
        if args.batch:
            app.run_batch_simulation(output_dir=args.output)
        else:
            app.run_visualization()
    
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()