"""
Ursina 3D Earthquake Visualization Engine
Real-time rendering with vertex shaders for seismic deformation
"""
from ursina import *
from ursina.shaders import lit_with_shadows_shader, basic_lighting_shader
import numpy as np
from pathlib import Path
import threading
import time


class SeismicTerrainEntity(Entity):
    """Custom terrain entity with dynamic vertex deformation"""
    
    def __init__(self, vertices, faces, normals, uvs, **kwargs):
        super().__init__()
        
        # Store base mesh data
        self.base_vertices = vertices.copy()
        self.current_vertices = vertices.copy()
        self.faces = faces
        self.normals = normals
        self.uvs = uvs
        
        # Create mesh
        self._create_mesh()
        
        # Shader for dynamic deformation
        self.shader = self._create_deformation_shader()
        
        # Properties
        self.model_initialized = True
        self.collider = 'mesh'
        
        # Apply any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def _create_mesh(self):
        """Create Ursina mesh from numpy arrays"""
        from ursina import Mesh
        
        # Convert faces to list format Ursina expects
        triangles = []
        for face in self.faces:
            triangles.append((int(face[0]), int(face[1]), int(face[2])))
        
        # Create mesh
        self.mesh = Mesh(
            vertices=self.current_vertices.tolist(),
            triangles=triangles,
            normals=self.normals.tolist(),
            uvs=self.uvs.tolist()
        )
        
        self.model = self.mesh
    
    def _create_deformation_shader(self):
        """Create physically-based vertex shader with full wave equations"""
        vertex_shader = """
        #version 330
        
        uniform mat4 p3d_ModelViewProjectionMatrix;
        uniform mat4 p3d_ModelViewMatrix;
        uniform mat4 p3d_ModelMatrix;
        uniform mat3 p3d_NormalMatrix;
        uniform vec4 p3d_ColorScale;
        
        in vec4 p3d_Vertex;
        in vec3 p3d_Normal;
        in vec2 p3d_MultiTexCoord0;
        in vec4 p3d_Color;
        
        out vec3 vpos;
        out vec3 vnormal;
        out vec3 world_pos;
        out vec2 texcoords;
        out vec4 vcolor;
        out float wave_amplitude;
        
        // Full seismic wave parameters
        uniform float time;
        uniform vec3 epicenter;
        uniform float magnitude;
        uniform float depth;
        uniform float vp;  // P-wave velocity km/s
        uniform float vs;  // S-wave velocity km/s
        uniform float Q_factor;  // Quality factor
        uniform float source_freq;  // Source frequency Hz
        uniform int wave_active;
        
        // Constants
        const float PI = 3.14159265359;
        const float TWO_PI = 6.28318530718;
        
        // Bessel function approximation for surface waves
        float bessel_j0(float x) {
            if (abs(x) < 3.0) {
                float y = x * x;
                return 1.0 - y/4.0 + y*y/64.0 - y*y*y/2304.0;
            } else {
                float z = 3.0 / x;
                return sqrt(2.0 / (PI * x)) * cos(x - PI/4.0);
            }
        }
        
        void main() {
            vec4 vertex = p3d_Vertex;
            vec3 normal = p3d_Normal;
            wave_amplitude = 0.0;
            
            if (wave_active > 0) {
                vec3 world_vertex = (p3d_ModelMatrix * vertex).xyz;
                
                // Horizontal distance from epicenter
                float dx = world_vertex.x - epicenter.x;
                float dz = world_vertex.z - epicenter.z;
                float r = sqrt(dx*dx + dz*dz);
                
                if (r > 0.01) {
                    // Moment magnitude to seismic moment
                    float M0 = pow(10.0, 1.5 * magnitude + 9.1);
                    
                    // P-wave arrival and propagation
                    float t_p = r / vp;
                    if (time >= t_p) {
                        float omega_p = TWO_PI * source_freq * 3.0;  // Higher frequency for P
                        float k_p = omega_p / vp;
                        float phase_p = omega_p * (time - t_p) - k_p * r;
                        
                        // Attenuation
                        float atten_p = exp(-omega_p * r / (2.0 * vp * Q_factor));
                        
                        // Geometric spreading
                        float geom_p = 1.0 / (r + 1.0);
                        
                        // Amplitude
                        float amp_p = M0 * 1e-22 * geom_p * atten_p;
                        
                        // P-wave motion (radial and vertical)
                        float p_wave = amp_p * sin(phase_p);
                        
                        // Radial component
                        vertex.x += (dx / r) * p_wave * 0.3;
                        vertex.z += (dz / r) * p_wave * 0.3;
                        // Vertical component (smaller)
                        vertex.y += p_wave * 0.15;
                        
                        wave_amplitude += abs(p_wave);
                    }
                    
                    // S-wave arrival and propagation
                    float t_s = r / vs;
                    if (time >= t_s) {
                        float omega_s = TWO_PI * source_freq * 1.5;
                        float k_s = omega_s / vs;
                        float phase_s = omega_s * (time - t_s) - k_s * r;
                        
                        float atten_s = exp(-omega_s * r / (2.0 * vs * Q_factor));
                        float geom_s = 1.0 / (r + 1.0);
                        float amp_s = M0 * 1e-22 * geom_s * atten_s;
                        
                        // S-wave motion (transverse and vertical)
                        float s_wave = amp_s * sin(phase_s);
                        
                        // Transverse components (perpendicular to radial)
                        vertex.x += (-dz / r) * s_wave * 0.6;
                        vertex.z += (dx / r) * s_wave * 0.6;
                        vertex.y += s_wave * 0.3;
                        
                        wave_amplitude += abs(s_wave);
                    }
                    
                    // Rayleigh surface waves (dominant for large distances)
                    if (r > 5.0) {
                        float c_r = vs * 0.92;  // Rayleigh velocity
                        float t_r = r / c_r;
                        
                        if (time >= t_r) {
                            float omega_r = TWO_PI * source_freq;
                            float k_r = omega_r / c_r;
                            float phase_r = omega_r * (time - t_r) - k_r * r;
                            
                            // Cylindrical spreading for surface waves
                            float geom_r = 1.0 / sqrt(r + 1.0);
                            float atten_r = exp(-omega_r * r / (2.0 * c_r * Q_factor * 0.5));
                            float amp_r = M0 * 1e-21 * geom_r * atten_r;
                            
                            // Rayleigh wave has both radial and vertical motion
                            float rayleigh = amp_r * sin(phase_r);
                            
                            // Retrograde elliptical motion
                            vertex.x += (dx / r) * rayleigh * 0.7;
                            vertex.z += (dz / r) * rayleigh * 0.7;
                            vertex.y += rayleigh * 1.2;  // Dominant vertical
                            
                            wave_amplitude += abs(rayleigh) * 1.5;
                        }
                    }
                    
                    // Love waves (transverse horizontal)
                    if (r > 5.0 && depth < 50.0) {  // Love waves require layering
                        float c_l = vs * 1.05;  // Love wave slightly faster than S
                        float t_l = r / c_l;
                        
                        if (time >= t_l) {
                            float omega_l = TWO_PI * source_freq * 0.8;
                            float k_l = omega_l / c_l;
                            float phase_l = omega_l * (time - t_l) - k_l * r;
                            
                            float geom_l = 1.0 / sqrt(r + 1.0);
                            float atten_l = exp(-omega_l * r / (2.0 * c_l * Q_factor * 0.5));
                            float amp_l = M0 * 1e-21 * geom_l * atten_l * 0.8;
                            
                            // Pure horizontal transverse motion
                            float love = amp_l * sin(phase_l);
                            
                            vertex.x += (-dz / r) * love;
                            vertex.z += (dx / r) * love;
                            
                            wave_amplitude += abs(love);
                        }
                    }
                    
                    // Normal perturbation for lighting
                    normal.y += wave_amplitude * 10.0;
                    normal = normalize(normal);
                }
            }
            
            gl_Position = p3d_ModelViewProjectionMatrix * vertex;
            vpos = (p3d_ModelViewMatrix * vertex).xyz;
            world_pos = (p3d_ModelMatrix * vertex).xyz;
            vnormal = normalize(p3d_NormalMatrix * normal);
            texcoords = p3d_MultiTexCoord0;
            vcolor = p3d_Color * p3d_ColorScale;
        }
        """
        
        fragment_shader = """
        #version 330
        
        uniform sampler2D p3d_Texture0;
        uniform vec4 p3d_ColorScale;
        uniform vec3 p3d_LightSource0_position;
        uniform vec3 p3d_LightSource0_diffuse;
        uniform vec3 p3d_LightSource0_specular;
        uniform vec3 camera_position;
        
        in vec3 vpos;
        in vec3 vnormal;
        in vec3 world_pos;
        in vec2 texcoords;
        in vec4 vcolor;
        in float wave_amplitude;
        
        out vec4 fragColor;
        
        const float PI = 3.14159265359;
        
        // Fresnel-Schlick approximation
        vec3 fresnelSchlick(float cosTheta, vec3 F0) {
            return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
        }
        
        // GGX/Trowbridge-Reitz normal distribution
        float DistributionGGX(vec3 N, vec3 H, float roughness) {
            float a = roughness * roughness;
            float a2 = a * a;
            float NdotH = max(dot(N, H), 0.0);
            float NdotH2 = NdotH * NdotH;
            
            float nom = a2;
            float denom = (NdotH2 * (a2 - 1.0) + 1.0);
            denom = PI * denom * denom;
            
            return nom / denom;
        }
        
        // Smith's Schlick-GGX geometry function
        float GeometrySchlickGGX(float NdotV, float roughness) {
            float r = (roughness + 1.0);
            float k = (r * r) / 8.0;
            
            float nom = NdotV;
            float denom = NdotV * (1.0 - k) + k;
            
            return nom / denom;
        }
        
        float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
            float NdotV = max(dot(N, V), 0.0);
            float NdotL = max(dot(N, L), 0.0);
            float ggx2 = GeometrySchlickGGX(NdotV, roughness);
            float ggx1 = GeometrySchlickGGX(NdotL, roughness);
            
            return ggx1 * ggx2;
        }
        
        void main() {
            vec3 N = normalize(vnormal);
            vec3 V = normalize(camera_position - world_pos);
            
            // Material properties (terrain-specific)
            vec3 albedo = vcolor.rgb;
            float metallic = 0.05;  // Terrain is mostly non-metallic
            float roughness = 0.7 - wave_amplitude * 0.3;  // Smoother during waves
            float ao = 1.0;  // Ambient occlusion
            
            // Fresnel F0 for dielectrics
            vec3 F0 = vec3(0.04);
            F0 = mix(F0, albedo, metallic);
            
            // Reflectance equation
            vec3 Lo = vec3(0.0);
            
            // Main directional light
            vec3 L = normalize(p3d_LightSource0_position);
            vec3 H = normalize(V + L);
            vec3 radiance = p3d_LightSource0_diffuse;
            
            // Cook-Torrance BRDF
            float NDF = DistributionGGX(N, H, roughness);
            float G = GeometrySmith(N, V, L, roughness);
            vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
            
            vec3 numerator = NDF * G * F;
            float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
            vec3 specular = numerator / denominator;
            
            vec3 kS = F;
            vec3 kD = vec3(1.0) - kS;
            kD *= 1.0 - metallic;
            
            float NdotL = max(dot(N, L), 0.0);
            Lo += (kD * albedo / PI + specular) * radiance * NdotL;
            
            // Ambient lighting with hemisphere
            vec3 ambient_sky = vec3(0.4, 0.5, 0.7);
            vec3 ambient_ground = vec3(0.2, 0.15, 0.1);
            float hemisphere_mix = dot(N, vec3(0, 1, 0)) * 0.5 + 0.5;
            vec3 ambient = mix(ambient_ground, ambient_sky, hemisphere_mix) * albedo * ao;
            
            vec3 color = ambient + Lo;
            
            // HDR tonemapping (Reinhard)
            color = color / (color + vec3(1.0));
            
            // Gamma correction
            color = pow(color, vec3(1.0 / 2.2));
            
            // Wave visualization overlay
            if (wave_amplitude > 0.001) {
                vec3 wave_color = vec3(1.0, 0.3, 0.0) * wave_amplitude * 2.0;
                color = mix(color, wave_color, clamp(wave_amplitude * 0.5, 0.0, 0.3));
            }
            
            fragColor = vec4(color, 1.0);
        }
        """
        
        shader = Shader(
            vertex=vertex_shader,
            fragment=fragment_shader,
            default_input={
                'time': 0.0,
                'epicenter': Vec3(0, 0, 0),
                'magnitude': 0.0,
                'depth': 10.0,
                'vp': 6.0,
                'vs': 3.5,
                'Q_factor': 300.0,
                'source_freq': 1.0,
                'wave_active': 0,
                'camera_position': Vec3(0, 15, -25)
            }
        )
        
        return shader
    
    def update_vertices(self, new_vertices: np.ndarray):
        """Update mesh vertices (CPU-based approach)"""
        self.current_vertices = new_vertices
        
        # Update mesh
        if hasattr(self, 'mesh'):
            self.mesh.vertices = new_vertices.tolist()
            self.mesh.generate()
    
    def set_shader_params(self, time: float, epicenter: tuple, 
                         magnitude: float, depth: float, active: bool,
                         vp: float = 6.0, vs: float = 3.5, Q: float = 300.0):
        """Update shader parameters for GPU-based full wave deformation"""
        if self.shader:
            self.shader.default_input['time'] = time
            self.shader.default_input['epicenter'] = Vec3(*epicenter)
            self.shader.default_input['magnitude'] = magnitude
            self.shader.default_input['depth'] = depth
            self.shader.default_input['vp'] = vp
            self.shader.default_input['vs'] = vs
            self.shader.default_input['Q_factor'] = Q
            self.shader.default_input['source_freq'] = 1.0  # Hz
            self.shader.default_input['wave_active'] = 1 if active else 0
            self.shader.default_input['camera_position'] = Vec3(camera.position.x, 
                                                                camera.position.y, 
                                                                camera.position.z)


class EarthquakeVisualization3D:
    """Main 3D visualization system"""
    
    def __init__(self, terrain_vertices, terrain_faces, terrain_normals, terrain_uvs,
                 seismic_simulator, window_size=(1920, 1080)):
        
        # Initialize Ursina
        self.app = Ursina(
            title="Enterprise Earthquake Simulator",
            borderless=False,
            fullscreen=False,
            size=window_size,
            vsync=True
        )
        
        # Store references
        self.seismic_simulator = seismic_simulator
        self.terrain_data = {
            'vertices': terrain_vertices,
            'faces': terrain_faces,
            'normals': terrain_normals,
            'uvs': terrain_uvs
        }
        
        # Simulation state
        self.simulation_active = False
        self.simulation_time = 0.0
        self.use_gpu_deformation = True  # Toggle between GPU/CPU
        
        # Setup scene
        self._setup_scene()
        self._setup_lighting()
        self._setup_camera()
        self._setup_ui()
        
        # Background simulation thread
        self.simulation_thread = None
        self.running = True
    
    def _setup_scene(self):
        """Create 3D scene elements"""
        # Sky
        self.sky = Sky(texture='sky_sunset')
        
        # Terrain
        self.terrain = SeismicTerrainEntity(
            vertices=self.terrain_data['vertices'],
            faces=self.terrain_data['faces'],
            normals=self.terrain_data['normals'],
            uvs=self.terrain_data['uvs'],
            color=color.rgb(139, 119, 101),  # Earth tone
            position=(0, -0.5, 0),
            scale=(20, 1, 20)
        )
        
        # Epicenter marker
        self.epicenter_marker = Entity(
            model='sphere',
            color=color.red,
            scale=0.3,
            position=(0, 100, 0),  # Hidden initially
            visible=False
        )
        
        # Grid reference
        self.grid = Entity(
            model=Grid(30, 30),
            scale=50,
            position=(0, -1, 0),
            color=color.rgba(255, 255, 255, 50),
            visible=False  # Hidden by default
        )
    
    def _setup_lighting(self):
        """Configure scene lighting"""
        # Main directional light (sun)
        self.sun = DirectionalLight(
            position=(10, 20, 10),
            rotation=(45, -45, 0),
            shadows=True
        )
        
        # Ambient light
        scene.fog_density = 0.02
        scene.fog_color = color.rgb(200, 200, 200)
        
        # Additional point lights for dramatic effect
        self.accent_lights = []
        for pos in [(5, 3, 5), (-5, 3, -5)]:
            light = Entity(
                model='sphere',
                position=pos,
                scale=0.1,
                color=color.white,
                unlit=False
            )
            self.accent_lights.append(light)
    
    def _setup_camera(self):
        """Configure camera with controls"""
        camera.position = (0, 15, -25)
        camera.rotation_x = -30
        
        # EditorCamera for free movement
        self.camera_controller = EditorCamera()
        self.camera_controller.rotation_speed = 100
        self.camera_controller.move_speed = 10
        
        # Camera shake for earthquake effect
        self.camera_shake_active = False
        self.camera_base_position = camera.position
    
    def _setup_ui(self):
        """Create user interface"""
        # Info panel
        self.info_text = Text(
            text="Earthquake Simulator - Ready\nPress SPACE to trigger earthquake",
            position=(-0.85, 0.45),
            scale=1.5,
            origin=(-0.5, 0.5),
            background=True
        )
        
        # Parameters panel
        self.params_text = Text(
            text="",
            position=(-0.85, 0.35),
            scale=1.2,
            origin=(-0.5, 0.5),
            background=True
        )
        
        # Status indicator
        self.status_text = Text(
            text="Status: Idle",
            position=(-0.85, -0.45),
            scale=1.2,
            origin=(-0.5, -0.5),
            background=True,
            color=color.green
        )
        
        # Buttons
        self.trigger_button = Button(
            text='Trigger Earthquake',
            position=(-0.7, -0.35),
            scale=(0.2, 0.05),
            on_click=self.trigger_earthquake
        )
        
        self.reset_button = Button(
            text='Reset',
            position=(-0.45, -0.35),
            scale=(0.15, 0.05),
            on_click=self.reset_simulation
        )
        
        self.toggle_grid_button = Button(
            text='Toggle Grid',
            position=(-0.25, -0.35),
            scale=(0.15, 0.05),
            on_click=self.toggle_grid
        )
        
        # Magnitude slider
        self.magnitude_slider = Slider(
            min=4.0, max=9.0, default=6.5, step=0.1,
            position=(0.5, 0.4),
            width=0.3,
            height=0.03
        )
        
        Text(
            text='Magnitude:',
            position=(0.35, 0.42),
            scale=1.2
        )
        
        self.magnitude_display = Text(
            text='6.5',
            position=(0.82, 0.42),
            scale=1.2
        )
        
        # Depth slider
        self.depth_slider = Slider(
            min=0, max=100, default=10, step=1,
            position=(0.5, 0.3),
            width=0.3,
            height=0.03
        )
        
        Text(
            text='Depth (km):',
            position=(0.35, 0.32),
            scale=1.2
        )
        
        self.depth_display = Text(
            text='10',
            position=(0.82, 0.32),
            scale=1.2
        )
    
    def trigger_earthquake(self):
        """Start earthquake simulation"""
        if self.simulation_active:
            return
        
        from seismic_engine import SeismicSource, FaultType
        
        # Get parameters from UI
        magnitude = self.magnitude_slider.value
        depth = self.depth_slider.value
        
        # Create earthquake source
        source = SeismicSource(
            latitude=0,  # Center of terrain
            longitude=0,
            depth=depth,
            magnitude=magnitude,
            strike=np.random.uniform(0, 360),
            dip=np.random.uniform(30, 90),
            rake=np.random.uniform(-180, 180),
            rupture_velocity=3.0,
            stress_drop=3e6
        )
        
        # Initialize simulation
        self.seismic_simulator.initialize_earthquake(source)
        self.simulation_active = True
        self.simulation_time = 0.0
        
        # Show epicenter marker
        self.epicenter_marker.position = (0, 0, 0)
        self.epicenter_marker.visible = True
        
        # Start camera shake
        self.camera_shake_active = True
        self.camera_base_position = camera.position
        
        # Update UI
        self.status_text.text = "Status: Simulating..."
        self.status_text.color = color.orange
        self.info_text.text = f"Earthquake Mw {magnitude:.1f} - Depth {depth}km\nSimulating wave propagation..."
        
        # Update shader parameters
        if self.use_gpu_deformation:
            self.terrain.set_shader_params(
                time=0,
                epicenter=(0, 0, 0),
                magnitude=magnitude,
                depth=depth,  # Add missing depth parameter
                active=True
            )
    
    def reset_simulation(self):
        """Reset to initial state"""
        self.simulation_active = False
        self.simulation_time = 0.0
        self.camera_shake_active = False
        
        # Reset terrain
        if not self.use_gpu_deformation:
            self.terrain.update_vertices(self.terrain_data['vertices'])
        else:
            self.terrain.set_shader_params(
                time=0, 
                epicenter=(0, 0, 0), 
                magnitude=0, 
                depth=10,  # Default depth
                active=False
            )
        
        # Hide epicenter
        self.epicenter_marker.visible = False
        
        # Reset seismic simulator
        self.seismic_simulator.stop()
        
        # Update UI
        self.status_text.text = "Status: Idle"
        self.status_text.color = color.green
        self.info_text.text = "Earthquake Simulator - Ready\nPress SPACE to trigger earthquake"
    
    def toggle_grid(self):
        """Toggle reference grid visibility"""
        self.grid.visible = not self.grid.visible
    
    def update(self):
        """Main update loop (called by Ursina)"""
        # Update magnitude display
        self.magnitude_display.text = f"{self.magnitude_slider.value:.1f}"
        self.depth_display.text = f"{self.depth_slider.value:.0f}"
        
        # Handle keyboard input
        if held_keys['space'] and not self.simulation_active:
            self.trigger_earthquake()
        
        if held_keys['r']:
            self.reset_simulation()
        
        if held_keys['g']:
            self.toggle_grid()
        
        # Update simulation
        if self.simulation_active:
            self.simulation_time += time.dt
            
            if self.use_gpu_deformation:
                # GPU-based: update shader parameters with full physics
                earth_vp = self.seismic_simulator.model.vp[0]
                earth_vs = self.seismic_simulator.model.vs[0]
                earth_Q = self.seismic_simulator.model.qs[0]
                
                self.terrain.set_shader_params(
                    time=self.simulation_time,
                    epicenter=(0, 0, 0),
                    magnitude=self.magnitude_slider.value,
                    depth=self.depth_slider.value,
                    active=True,
                    vp=earth_vp,
                    vs=earth_vs,
                    Q=earth_Q
                )
            else:
                # CPU-based: update vertex positions
                displacement = self.seismic_simulator.get_displacement_field()
                
                from terrain_processor import TerrainSeismicIntegrator
                integrator = TerrainSeismicIntegrator(
                    self.terrain_data['vertices'],
                    self.seismic_simulator.model.vp  # Dummy elevation
                )
                
                new_vertices = integrator.apply_displacement_fast(
                    self.terrain_data['vertices'],
                    displacement,
                    self.seismic_simulator.width,
                    self.seismic_simulator.height,
                    0.01
                )
                
                self.terrain.update_vertices(new_vertices)
            
            # Camera shake effect
            if self.camera_shake_active:
                shake_magnitude = self.magnitude_slider.value * 0.1
                shake_decay = np.exp(-self.simulation_time * 0.5)
                
                camera.position = self.camera_base_position + Vec3(
                    np.random.uniform(-1, 1) * shake_magnitude * shake_decay,
                    np.random.uniform(-1, 1) * shake_magnitude * shake_decay * 0.5,
                    np.random.uniform(-1, 1) * shake_magnitude * shake_decay
                )
            
            # Check if simulation should stop
            if not self.seismic_simulator.active:
                self.simulation_active = False
                self.camera_shake_active = False
                self.status_text.text = "Status: Complete"
                self.status_text.color = color.green
                self.info_text.text = "Simulation complete\nPress SPACE for new earthquake"
    
    def run(self):
        """Start the visualization"""
        self.app.run()
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join()
        application.quit()


def create_visualization_from_dem(dem_path: str = None):
    """Create visualization from DEM file or synthetic terrain"""
    from terrain_processor import DEMProcessor, TerrainMeshGenerator, create_synthetic_dem
    from seismic_engine import create_earth_model_iasp91, RealTimeSeismicSimulator
    
    # Load or create terrain
    if dem_path and Path(dem_path).exists():
        processor = DEMProcessor(dem_path, target_resolution=30.0)
        elevation = processor.load_dem()
    else:
        print("Creating synthetic terrain (no DEM provided)...")
        elevation = create_synthetic_dem(256, 256, complexity=0.6)
    
    # Generate mesh
    mesh_generator = TerrainMeshGenerator(
        elevation,
        vertical_exaggeration=2.0,
        simplification_factor=0.5
    )
    vertices, faces, normals, uvs = mesh_generator.generate_mesh()
    
    # Create Earth model
    earth_model = create_earth_model_iasp91()
    
    # Create seismic simulator
    simulator = RealTimeSeismicSimulator(
        terrain_shape=elevation.shape,
        earth_model=earth_model
    )
    
    # Create visualization
    viz = EarthquakeVisualization3D(
        terrain_vertices=vertices,
        terrain_faces=faces,
        terrain_normals=normals,
        terrain_uvs=uvs,
        seismic_simulator=simulator
    )
    
    return viz