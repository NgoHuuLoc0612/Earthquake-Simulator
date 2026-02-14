"""
Advanced Terrain Processing System
Handles 16-bit DEM files with rasterio, generates high-quality 3D meshes,
applies geomorphological analysis, and integrates with seismic simulation
"""
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import gaussian_filter, sobel, generic_filter
from scipy.spatial import Delaunay
from numba import jit, prange
from dataclasses import dataclass
from typing import Tuple, Optional, List
import struct


@dataclass
class TerrainMetadata:
    """Comprehensive terrain characteristics"""
    bounds: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y
    crs: str  # Coordinate reference system
    resolution: Tuple[float, float]  # x_res, y_res in meters
    nodata_value: Optional[float]
    elevation_range: Tuple[float, float]  # min, max elevation
    mean_slope: float  # degrees
    mean_aspect: float  # degrees
    roughness: float  # terrain complexity metric


class DEMProcessor:
    """Enterprise-grade DEM processing with rasterio"""
    
    def __init__(self, dem_path: str, target_resolution: Optional[float] = None):
        self.dem_path = dem_path
        self.target_resolution = target_resolution
        self.elevation: Optional[np.ndarray] = None
        self.metadata: Optional[TerrainMetadata] = None
        self.transform = None
        self.crs = None
        
    def load_dem(self) -> np.ndarray:
        """Load and preprocess 16-bit DEM with rasterio"""
        with rasterio.open(self.dem_path) as src:
            # Read all bands (usually just 1 for DEM)
            elevation = src.read(1, masked=True)
            
            self.transform = src.transform
            self.crs = src.crs
            nodata = src.nodata
            
            # Convert to float64 FIRST (before handling nodata)
            elevation = elevation.astype(np.float64)
            
            # Handle nodata values - now safe because it's float64
            if nodata is not None:
                if hasattr(elevation, 'mask'):
                    # If masked array, fill masked values with NaN
                    elevation = np.ma.filled(elevation, fill_value=np.nan)
                else:
                    # Replace nodata value with NaN
                    elevation[elevation == nodata] = np.nan
            
            # Resample if target resolution specified
            if self.target_resolution is not None:
                elevation = self._resample_dem(src, elevation)
            
            # Fill NaN values with interpolation
            elevation = self._fill_missing_data(elevation)
            
            # Compute metadata
            self.metadata = self._compute_metadata(elevation, src.bounds, src.crs)
            
            self.elevation = elevation
            return elevation
    
    def _resample_dem(self, src: rasterio.DatasetReader, 
                     data: np.ndarray) -> np.ndarray:
        """High-quality resampling using Lanczos"""
        scale = src.res[0] / self.target_resolution
        
        new_width = int(src.width * scale)
        new_height = int(src.height * scale)
        
        # Calculate new transform
        transform, width, height = calculate_default_transform(
            src.crs, src.crs, new_width, new_height, *src.bounds,
            resolution=self.target_resolution
        )
        
        # Create output array
        resampled = np.empty((height, width), dtype=data.dtype)
        
        # Reproject with Lanczos resampling (high quality)
        reproject(
            source=data,
            destination=resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=src.crs,
            resampling=Resampling.lanczos
        )
        
        self.transform = transform
        return resampled
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _fill_missing_data_parallel(data: np.ndarray) -> np.ndarray:
        """Fast NaN filling using neighborhood averaging"""
        height, width = data.shape
        result = data.copy()
        
        for i in prange(1, height - 1):
            for j in range(1, width - 1):
                if np.isnan(data[i, j]):
                    # Average of valid neighbors
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            val = data[i + di, j + dj]
                            if not np.isnan(val):
                                neighbors.append(val)
                    
                    if len(neighbors) > 0:
                        result[i, j] = np.mean(np.array(neighbors))
        
        return result
    
    def _fill_missing_data(self, data: np.ndarray) -> np.ndarray:
        """Iterative filling of missing data"""
        max_iterations = 10
        for _ in range(max_iterations):
            nan_count = np.sum(np.isnan(data))
            if nan_count == 0:
                break
            data = self._fill_missing_data_parallel(data)
        
        # If still NaN, fill with mean
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, nan=np.nanmean(data))
        
        return data
    
    def _compute_metadata(self, elevation: np.ndarray, bounds, crs) -> TerrainMetadata:
        """Extract comprehensive terrain characteristics"""
        # Elevation statistics
        elev_min, elev_max = np.min(elevation), np.max(elevation)
        
        # Slope calculation (Horn's method - 3rd order finite difference)
        slope = self.compute_slope(elevation)
        mean_slope = np.mean(slope)
        
        # Aspect calculation
        aspect = self.compute_aspect(elevation)
        mean_aspect = np.mean(aspect)
        
        # Roughness (standard deviation of elevation in moving window)
        roughness = self.compute_roughness(elevation)
        
        resolution = (self.transform[0], -self.transform[4])
        
        return TerrainMetadata(
            bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
            crs=str(crs),
            resolution=resolution,
            nodata_value=None,
            elevation_range=(elev_min, elev_max),
            mean_slope=mean_slope,
            mean_aspect=mean_aspect,
            roughness=roughness
        )
    
    @staticmethod
    def compute_slope(elevation: np.ndarray, cell_size: float = 30.0) -> np.ndarray:
        """Compute slope in degrees using Horn's method"""
        dz_dx = sobel(elevation, axis=1) / (8 * cell_size)
        dz_dy = sobel(elevation, axis=0) / (8 * cell_size)
        
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        return np.degrees(slope_rad)
    
    @staticmethod
    def compute_aspect(elevation: np.ndarray) -> np.ndarray:
        """Compute aspect (direction of slope) in degrees"""
        dz_dx = sobel(elevation, axis=1)
        dz_dy = sobel(elevation, axis=0)
        
        aspect_rad = np.arctan2(-dz_dy, dz_dx)
        aspect_deg = np.degrees(aspect_rad)
        
        # Convert to compass bearing (0 = North, 90 = East)
        aspect_deg = 90 - aspect_deg
        aspect_deg[aspect_deg < 0] += 360
        
        return aspect_deg
    
    @staticmethod
    def compute_roughness(elevation: np.ndarray, window_size: int = 5) -> float:
        """Terrain roughness index (TRI)"""
        def std_func(values):
            return np.std(values)
        
        tri = generic_filter(elevation, std_func, size=window_size)
        return np.mean(tri)
    
    def apply_geomorphological_smoothing(self, iterations: int = 3):
        """Advanced smoothing preserving geomorphological features"""
        if self.elevation is None:
            raise ValueError("DEM not loaded")
        
        # Anisotropic diffusion (Perona-Malik)
        for _ in range(iterations):
            self.elevation = self._perona_malik_step(self.elevation)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _perona_malik_step(img: np.ndarray, kappa: float = 50.0, 
                          gamma: float = 0.2) -> np.ndarray:
        """Perona-Malik anisotropic diffusion step"""
        h, w = img.shape
        result = img.copy()
        
        for i in prange(1, h - 1):
            for j in range(1, w - 1):
                # Gradients in 4 directions
                grad_n = img[i-1, j] - img[i, j]
                grad_s = img[i+1, j] - img[i, j]
                grad_e = img[i, j+1] - img[i, j]
                grad_w = img[i, j-1] - img[i, j]
                
                # Diffusion coefficients
                c_n = np.exp(-(grad_n / kappa)**2)
                c_s = np.exp(-(grad_s / kappa)**2)
                c_e = np.exp(-(grad_e / kappa)**2)
                c_w = np.exp(-(grad_w / kappa)**2)
                
                # Update
                result[i, j] = img[i, j] + gamma * (
                    c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w
                )
        
        return result


class TerrainMeshGenerator:
    """Generate production-grade 3D mesh with advanced algorithms"""
    
    def __init__(self, elevation: np.ndarray, 
                 vertical_exaggeration: float = 2.0,
                 adaptive_refinement: bool = True):
        self.elevation = elevation
        self.height, self.width = elevation.shape
        self.vertical_exaggeration = vertical_exaggeration
        self.adaptive_refinement = adaptive_refinement
        
        # Mesh data
        self.vertices: Optional[np.ndarray] = None
        self.faces: Optional[np.ndarray] = None
        self.normals: Optional[np.ndarray] = None
        self.uvs: Optional[np.ndarray] = None
        self.vertex_colors: Optional[np.ndarray] = None
        
        # Adaptive refinement data structures
        self.error_metric: Optional[np.ndarray] = None
        self.refinement_levels: Optional[np.ndarray] = None
        
    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate complete mesh with adaptive refinement based on terrain complexity"""
        
        if self.adaptive_refinement:
            # Compute error metric for adaptive refinement
            self.error_metric = self._compute_curvature_error()
            self.refinement_levels = self._determine_refinement_levels()
            
            # Generate adaptively refined mesh
            vertices, faces = self._generate_adaptive_mesh()
        else:
            # Standard regular grid mesh
            vertices = self._generate_vertices(self.elevation)
            faces = self._generate_faces_constrained_delaunay(self.height, self.width)
        
        # Compute per-vertex normals using angle-weighted method
        normals = self._compute_angle_weighted_normals(vertices, faces)
        
        # Generate UVs with proper texture coordinate mapping
        uvs = self._generate_uvs_with_aspect_correction(vertices, faces)
        
        # Generate vertex colors based on elevation and slope
        self.vertex_colors = self._generate_vertex_colors(vertices, normals)
        
        self.vertices = vertices
        self.faces = faces
        self.normals = normals
        self.uvs = uvs
        
        return vertices, faces, normals, uvs
    
    def _compute_curvature_error(self) -> np.ndarray:
        """Compute Gaussian curvature for error-driven refinement"""
        # First fundamental form coefficients
        dz_dx = sobel(self.elevation, axis=1)
        dz_dy = sobel(self.elevation, axis=0)
        
        # Second fundamental form coefficients
        d2z_dx2 = sobel(dz_dx, axis=1)
        d2z_dy2 = sobel(dz_dy, axis=0)
        d2z_dxdy = sobel(dz_dx, axis=0)
        
        # Gaussian curvature K = (Lxx * Lyy - Lxy^2) / (1 + fx^2 + fy^2)^2
        denominator = (1 + dz_dx**2 + dz_dy**2) ** 2
        numerator = d2z_dx2 * d2z_dy2 - d2z_dxdy ** 2
        
        curvature = np.abs(numerator / (denominator + 1e-10))
        
        # Normalize to [0, 1]
        curvature = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-10)
        
        return curvature
    
    def _determine_refinement_levels(self) -> np.ndarray:
        """Determine refinement levels based on error metric"""
        refinement = np.zeros_like(self.error_metric, dtype=np.int32)
        
        # Thresholds for different refinement levels
        thresholds = [0.1, 0.3, 0.6, 0.85]
        
        for level, threshold in enumerate(thresholds):
            refinement[self.error_metric > threshold] = level + 1
        
        return refinement
    
    def _generate_adaptive_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mesh with adaptive refinement using quadtree subdivision"""
        vertices_list = []
        faces_list = []
        vertex_map = {}
        
        def add_vertex(i, j, level_offset=0):
            """Add vertex with level-dependent position"""
            key = (i, j, level_offset)
            if key in vertex_map:
                return vertex_map[key]
            
            h, w = self.elevation.shape
            x = (j / (w - 1)) * 2 - 1
            y = (i / (h - 1)) * 2 - 1
            
            elev = self.elevation[min(i, h-1), min(j, w-1)]
            elev_norm = (elev - self.elevation.min()) / (self.elevation.max() - self.elevation.min() + 1e-10)
            z = elev_norm * self.vertical_exaggeration
            
            idx = len(vertices_list)
            vertices_list.append([x, z, y])
            vertex_map[key] = idx
            return idx
        
        # Subdivide based on refinement levels
        h, w = self.elevation.shape
        
        for i in range(0, h - 1):
            for j in range(0, w - 1):
                level = self.refinement_levels[i, j]
                # Use float to allow negative exponents, then convert to int
                # Higher levels = finer detail = smaller steps
                step = max(1, int(2.0 ** (3 - level)))  # Inverse refinement
                
                # Ensure we don't go out of bounds
                i_next = min(i + step, h - 1)
                j_next = min(j + step, w - 1)
                
                # Create quad and split into two triangles
                v0 = add_vertex(i, j)
                v1 = add_vertex(i, j_next)
                v2 = add_vertex(i_next, j)
                v3 = add_vertex(i_next, j_next)
                
                # Triangle 1
                faces_list.append([v0, v2, v1])
                # Triangle 2
                faces_list.append([v1, v2, v3])
        
        vertices = np.array(vertices_list, dtype=np.float32)
        faces = np.array(faces_list, dtype=np.int32)
        
        return vertices, faces
    
    def _generate_faces_constrained_delaunay(self, h: int, w: int) -> np.ndarray:
        """Generate faces using constrained Delaunay triangulation"""
        # Create point cloud
        points = []
        point_indices = {}
        
        idx = 0
        for i in range(h):
            for j in range(w):
                points.append([j, i])
                point_indices[(i, j)] = idx
                idx += 1
        
        points = np.array(points)
        
        # Perform Delaunay triangulation
        tri = Delaunay(points)
        
        # Extract faces
        faces = tri.simplices
        
        # Constrain edges (ensure grid connectivity)
        # Add explicit edges for grid structure
        constrained_faces = []
        
        for face in faces:
            # Check if this triangle respects grid topology
            # (vertices should be close in grid space)
            p0, p1, p2 = face
            
            # Find grid coordinates
            coords = []
            for p in [p0, p1, p2]:
                i, j = p // w, p % w
                coords.append((i, j))
            
            # Check edge lengths in grid space
            max_dist = 0
            for k1 in range(3):
                for k2 in range(k1 + 1, 3):
                    dist = abs(coords[k1][0] - coords[k2][0]) + abs(coords[k1][1] - coords[k2][1])
                    max_dist = max(max_dist, dist)
            
            # Only accept triangles with reasonable edge lengths
            if max_dist <= 2:
                constrained_faces.append(face)
        
        return np.array(constrained_faces, dtype=np.int32)
    
    def _generate_vertices(self, elevation: np.ndarray) -> np.ndarray:
        """Create vertex array from elevation data"""
        h, w = elevation.shape
        vertices = np.zeros((h * w, 3), dtype=np.float32)
        
        # Normalize coordinates to [-1, 1] for x and y
        # Z is elevation with vertical exaggeration
        elev_min, elev_max = elevation.min(), elevation.max()
        elev_range = elev_max - elev_min if elev_max != elev_min else 1.0
        
        idx = 0
        for i in range(h):
            for j in range(w):
                x = (j / (w - 1)) * 2 - 1  # [-1, 1]
                y = (i / (h - 1)) * 2 - 1  # [-1, 1]
                z = ((elevation[i, j] - elev_min) / elev_range) * self.vertical_exaggeration
                
                vertices[idx] = [x, z, y]  # Note: y and z swapped for Ursina
                idx += 1
        
        return vertices
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _generate_faces_parallel(h: int, w: int) -> np.ndarray:
        """Fast face generation using numba"""
        num_faces = (h - 1) * (w - 1) * 2
        faces = np.zeros((num_faces, 3), dtype=np.int32)
        
        face_idx = 0
        for i in prange(h - 1):
            for j in range(w - 1):
                # Calculate vertex indices
                v0 = i * w + j
                v1 = v0 + 1
                v2 = v0 + w
                v3 = v2 + 1
                
                # First triangle
                faces[face_idx * 2] = [v0, v2, v1]
                # Second triangle
                faces[face_idx * 2 + 1] = [v1, v2, v3]
                
                face_idx += 1
        
        return faces
    
    def _generate_faces(self, h: int, w: int) -> np.ndarray:
        """Generate triangle faces for mesh"""
        faces = []
        
        for i in range(h - 1):
            for j in range(w - 1):
                # Vertex indices for quad
                v0 = i * w + j
                v1 = v0 + 1
                v2 = v0 + w
                v3 = v2 + 1
                
                # Two triangles per quad
                faces.append([v0, v2, v1])
                faces.append([v1, v2, v3])
        
        return np.array(faces, dtype=np.int32)
    
    def _compute_angle_weighted_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Compute angle-weighted vertex normals (Max 1999)"""
        normals = np.zeros_like(vertices)
        
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            
            # Edge vectors
            e1 = v1 - v0
            e2 = v2 - v0
            e3 = v2 - v1
            
            # Face normal (cross product)
            face_normal = np.cross(e1, e2)
            face_area = np.linalg.norm(face_normal)
            
            if face_area > 1e-10:
                face_normal /= face_area
                
                # Compute angles at each vertex
                # Angle at v0
                cos_angle0 = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10)
                cos_angle0 = np.clip(cos_angle0, -1.0, 1.0)
                angle0 = np.arccos(cos_angle0)
                
                # Angle at v1
                cos_angle1 = np.dot(-e1, e3) / (np.linalg.norm(e1) * np.linalg.norm(e3) + 1e-10)
                cos_angle1 = np.clip(cos_angle1, -1.0, 1.0)
                angle1 = np.arccos(cos_angle1)
                
                # Angle at v2
                cos_angle2 = np.dot(-e2, -e3) / (np.linalg.norm(e2) * np.linalg.norm(e3) + 1e-10)
                cos_angle2 = np.clip(cos_angle2, -1.0, 1.0)
                angle2 = np.arccos(cos_angle2)
                
                # Weight by angle
                normals[face[0]] += face_normal * angle0
                normals[face[1]] += face_normal * angle1
                normals[face[2]] += face_normal * angle2
        
        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normals /= norms
        
        return normals
    
    def _generate_uvs_with_aspect_correction(self, vertices: np.ndarray, 
                                            faces: np.ndarray) -> np.ndarray:
        """Generate texture coordinates with aspect ratio correction"""
        n_verts = len(vertices)
        uvs = np.zeros((n_verts, 2), dtype=np.float32)
        
        # Find bounding box in XZ plane
        x_coords = vertices[:, 0]
        z_coords = vertices[:, 2]
        
        x_min, x_max = x_coords.min(), x_coords.max()
        z_min, z_max = z_coords.min(), z_coords.max()
        
        x_range = x_max - x_min
        z_range = z_max - z_min
        
        # Compute aspect ratio
        aspect = x_range / (z_range + 1e-10)
        
        for i in range(n_verts):
            # Normalize to [0, 1]
            u = (vertices[i, 0] - x_min) / (x_range + 1e-10)
            v = (vertices[i, 2] - z_min) / (z_range + 1e-10)
            
            # Apply aspect correction
            if aspect > 1.0:
                v /= aspect
            else:
                u *= aspect
            
            uvs[i] = [u, v]
        
        return uvs
    
    def _generate_vertex_colors(self, vertices: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Generate vertex colors based on elevation and terrain features"""
        colors = np.zeros((len(vertices), 4), dtype=np.float32)
        
        # Extract elevation (y-component)
        elevations = vertices[:, 1]
        elev_min, elev_max = elevations.min(), elevations.max()
        elev_norm = (elevations - elev_min) / (elev_max - elev_min + 1e-10)
        
        for i in range(len(vertices)):
            h = elev_norm[i]
            
            # Color gradient based on elevation
            if h < 0.2:  # Low elevation - greenish
                r, g, b = 0.2 + h, 0.5 + h * 0.5, 0.2
            elif h < 0.5:  # Mid elevation - brown/tan
                t = (h - 0.2) / 0.3
                r, g, b = 0.6 + t * 0.2, 0.5 + t * 0.2, 0.3
            elif h < 0.8:  # High elevation - gray
                t = (h - 0.5) / 0.3
                r, g, b = 0.5 + t * 0.3, 0.5 + t * 0.3, 0.5 + t * 0.3
            else:  # Very high - white (snow)
                t = (h - 0.8) / 0.2
                r, g, b = 0.8 + t * 0.2, 0.8 + t * 0.2, 0.8 + t * 0.2
            
            # Modulate by slope (from normal)
            slope_factor = abs(normals[i, 1])  # Vertical component
            r *= (0.7 + 0.3 * slope_factor)
            g *= (0.7 + 0.3 * slope_factor)
            b *= (0.7 + 0.3 * slope_factor)
            
            colors[i] = [r, g, b, 1.0]
        
        return colors
    
    def apply_continuous_lod(self, camera_position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Continuous Level-of-Detail using view-dependent mesh refinement"""
        if self.vertices is None or self.faces is None:
            raise ValueError("Mesh not generated yet")
        
        # Compute distance from camera to each vertex
        distances = np.linalg.norm(self.vertices - camera_position, axis=1)
        
        # Determine per-vertex detail level based on distance and curvature
        detail_levels = np.zeros(len(self.vertices), dtype=np.float32)
        
        for i, dist in enumerate(distances):
            # Distance-based LOD
            if dist < 10:
                base_level = 1.0
            elif dist < 30:
                base_level = 0.7
            elif dist < 60:
                base_level = 0.4
            else:
                base_level = 0.2
            
            # Modulate by local curvature if available
            if self.error_metric is not None:
                # Map vertex back to grid
                x_norm = (self.vertices[i, 0] + 1) * 0.5
                z_norm = (self.vertices[i, 2] + 1) * 0.5
                
                grid_i = int(z_norm * (self.height - 1))
                grid_j = int(x_norm * (self.width - 1))
                
                grid_i = np.clip(grid_i, 0, self.height - 1)
                grid_j = np.clip(grid_j, 0, self.width - 1)
                
                curvature_factor = self.error_metric[grid_i, grid_j]
                detail_levels[i] = base_level * (1.0 + curvature_factor)
            else:
                detail_levels[i] = base_level
        
        # Filter faces based on vertex detail levels
        filtered_faces = []
        for face in self.faces:
            # Average detail level for this face
            avg_detail = np.mean([detail_levels[v] for v in face])
            
            # Randomly keep face based on detail level
            if np.random.random() < avg_detail:
                filtered_faces.append(face)
        
        return self.vertices, np.array(filtered_faces, dtype=np.int32)


class TerrainSeismicIntegrator:
    """Integrate terrain mesh with seismic displacement"""
    
    def __init__(self, base_vertices: np.ndarray, elevation_2d: np.ndarray):
        self.base_vertices = base_vertices.copy()
        self.elevation_2d = elevation_2d
        self.height, self.width = elevation_2d.shape
        
    def apply_seismic_displacement(self, displacement_field: np.ndarray) -> np.ndarray:
        """Apply 3D displacement field to terrain mesh"""
        # displacement_field shape: (width, height, 3)
        # Need to map to vertices
        
        modified_vertices = self.base_vertices.copy()
        
        # Interpolate displacement to vertex positions
        for i in range(len(modified_vertices)):
            # Get grid position from vertex
            grid_x = int((modified_vertices[i, 0] + 1) * 0.5 * (self.width - 1))
            grid_y = int((modified_vertices[i, 2] + 1) * 0.5 * (self.height - 1))
            
            grid_x = np.clip(grid_x, 0, self.width - 1)
            grid_y = np.clip(grid_y, 0, self.height - 1)
            
            # Apply displacement
            disp = displacement_field[grid_x, grid_y]
            modified_vertices[i, 0] += disp[0] * 0.001  # Scale for visibility
            modified_vertices[i, 1] += disp[2] * 0.001  # Vertical
            modified_vertices[i, 2] += disp[1] * 0.001
        
        return modified_vertices
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def apply_displacement_fast(vertices: np.ndarray, displacement: np.ndarray,
                               width: int, height: int, scale: float) -> np.ndarray:
        """Fast displacement application"""
        result = vertices.copy()
        n_verts = len(vertices)
        
        for i in prange(n_verts):
            # Map vertex to grid
            grid_x = int((vertices[i, 0] + 1) * 0.5 * (width - 1))
            grid_y = int((vertices[i, 2] + 1) * 0.5 * (height - 1))
            
            if 0 <= grid_x < width and 0 <= grid_y < height:
                result[i, 0] += displacement[grid_x, grid_y, 0] * scale
                result[i, 1] += displacement[grid_x, grid_y, 2] * scale
                result[i, 2] += displacement[grid_x, grid_y, 1] * scale
        
        return result


def create_synthetic_dem(width: int, height: int, 
                        complexity: float = 0.5) -> np.ndarray:
    """Generate realistic synthetic DEM for testing"""
    # Multi-scale Perlin-like noise
    elevation = np.zeros((height, width), dtype=np.float64)
    
    # Multiple octaves
    for octave in range(6):
        frequency = 2 ** octave
        amplitude = complexity ** octave
        
        # Generate noise for this octave
        noise = np.random.randn(height // frequency + 1, width // frequency + 1)
        noise = gaussian_filter(noise, sigma=2)
        
        # Upsample to full resolution
        from scipy.ndimage import zoom
        noise_full = zoom(noise, frequency, order=3)
        
        # Crop to exact size
        noise_full = noise_full[:height, :width]
        
        elevation += noise_full * amplitude * 1000
    
    # Add some mountain ranges
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    mountain = 3000 * np.exp(-((X-0.3)**2 + (Y-0.2)**2) / 0.1)
    elevation += mountain
    
    # Ensure all positive
    elevation -= elevation.min()
    
    return elevation