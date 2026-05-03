const PI: f32 = 3.14159265359;

struct CameraUniforms {
    position: vec3<f32>,
    pitch: f32,
    yaw: f32,
    fov: f32,
    aspect: f32,
    time: f32,
    resolution: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(0) @binding(1)
var voxel_texture: texture_3d<u32>;

struct Material {
    base_color: vec3<f32>,
    roughness: f32,
    pattern_scale: f32,
    ao_strength: f32,
};

// --- Hash & Noise Functions ---
fn hash1(n: f32) -> f32 {
    return fract(sin(n * 127.1) * 43758.5453);
}

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn hash3(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

fn hash3v(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        hash3(p + vec3<f32>(0.0, 0.0, 0.0)),
        hash3(p + vec3<f32>(57.0, 13.0, 89.0)),
        hash3(p + vec3<f32>(91.0, 37.0, 11.0))
    );
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    var f = fract(p);
    f = f * f * (3.0 - 2.0 * f); // smoothstep
    
    let n000 = hash3(i + vec3<f32>(0.0, 0.0, 0.0));
    let n001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let n010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let n011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let n100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let n101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let n110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let n111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));
    
    let nx00 = mix(n000, n100, f.x);
    let nx01 = mix(n001, n101, f.x);
    let nx10 = mix(n010, n110, f.x);
    let nx11 = mix(n011, n111, f.x);
    
    let nxy0 = mix(nx00, nx10, f.y);
    let nxy1 = mix(nx01, nx11, f.y);
    
    return mix(nxy0, nxy1, f.z);
}

fn fbm3(p: vec3<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var freq = 1.0;
    for (var i = 0; i < octaves; i = i + 1) {
        value = value + amplitude * value_noise(p * freq);
        amplitude = amplitude * 0.5;
        freq = freq * 2.0;
    }
    return value;
}

// --- Procedural Material Textures ---
fn get_stone_brick_color(world_pos: vec3<f32>, base: vec3<f32>) -> vec3<f32> {
    let brick_size = 4.0;
    let mortar = 0.15;
    let bp = world_pos / brick_size;
    let local = fract(bp);
    
    let bx = abs(local.x - 0.5);
    let by = abs(local.y - 0.5);
    let bz = abs(local.z - 0.5);
    
    // Stagger every other row
    var stagger = 0.0;
    if floor(bp.y) % 2.0 == 0.0 {
        stagger = 0.5;
    }
    let staggered_x = fract(bp.x + stagger);
    let sx = abs(staggered_x - 0.5);
    
    let edge_dist = min(min(sx, by), bz);
    let is_mortar = edge_dist < mortar;
    
    // Noise variation
    let n = fbm3(world_pos * 0.8, 2);
    let color_var = base * (0.85 + n * 0.3);
    
    let mortar_color = vec3<f32>(0.45, 0.42, 0.38) * (0.8 + n * 0.2);
    
    return select(mortar_color, color_var, !is_mortar);
}

fn get_cobblestone_color(world_pos: vec3<f32>, base: vec3<f32>) -> vec3<f32> {
    let n = fbm3(world_pos * 1.2, 3);
    let n2 = fbm3(world_pos * 3.0 + vec3<f32>(5.0), 2);
    
    // Irregular stone shapes via thresholded noise
    let stone = step(0.35, n);
    let stone_color = base * (0.7 + n2 * 0.4);
    let gap_color = vec3<f32>(0.35, 0.33, 0.30) * (0.8 + n * 0.2);
    
    return mix(gap_color, stone_color, stone);
}

fn get_wood_color(world_pos: vec3<f32>, base: vec3<f32>) -> vec3<f32> {
    // Wood grain
    let grain = world_pos.x * 2.5 + fbm3(world_pos * vec3<f32>(3.0, 0.5, 0.5), 2) * 2.0;
    let grain_pattern = abs(sin(grain * PI));
    let grain_dark = pow(grain_pattern, 4.0);
    
    // Plank lines
    let plank = fract(world_pos.y / 3.0);
    let is_plank_edge = plank < 0.03 || plank > 0.97;
    
    let wood_color = base * (0.7 + grain_dark * 0.4);
    let plank_color = vec3<f32>(0.35, 0.25, 0.15);
    
    return mix(wood_color, plank_color, select(0.0, 1.0, is_plank_edge));
}

fn get_grass_color(world_pos: vec3<f32>, base: vec3<f32>) -> vec3<f32> {
    let n = fbm3(world_pos * 2.0, 2);
    let n2 = hash3(world_pos * 0.5);
    
    // Color variation
    var color = base * (0.8 + n * 0.4);
    
    // Add tiny flowers/patches
    let flower = step(0.97, hash3(floor(world_pos * 2.0)));
    let flower_color = select(
        vec3<f32>(0.9, 0.85, 0.3),
        vec3<f32>(0.7, 0.9, 0.3),
        n2 > 0.5
    );
    color = mix(color, flower_color, flower);
    
    // Moss patches on lower grass
    let moss = step(0.92, hash3(floor(world_pos * 0.7)));
    color = mix(color, vec3<f32>(0.25, 0.45, 0.15), moss * 0.5);
    
    return color;
}

fn get_dirt_color(world_pos: vec3<f32>, base: vec3<f32>) -> vec3<f32> {
    let n = fbm3(world_pos * 1.5, 3);
    let pebbles = step(0.82, n);
    let pebble_color = vec3<f32>(0.5, 0.48, 0.42);
    return mix(base * (0.75 + n * 0.25), pebble_color, pebbles * 0.4);
}

fn get_mossy_stone_color(world_pos: vec3<f32>, base: vec3<f32>) -> vec3<f32> {
    let stone = get_stone_brick_color(world_pos, base);
    let moss = fbm3(world_pos * 0.7 + vec3<f32>(100.0), 3);
    let moss_mask = step(0.55, moss);
    let moss_color = vec3<f32>(0.25, 0.4, 0.2) * (0.7 + moss * 0.5);
    return mix(stone, moss_color, moss_mask * 0.7);
}

fn get_sand_color(world_pos: vec3<f32>, base: vec3<f32>) -> vec3<f32> {
    let n = fbm3(world_pos * 3.0, 2);
    return base * (0.85 + n * 0.3);
}

fn get_water_color(world_pos: vec3<f32>, base: vec3<f32>, time: f32) -> vec3<f32> {
    let n = fbm3(world_pos * 2.0 + vec3<f32>(0.0, time * 0.3, 0.0), 2);
    return base * (0.7 + n * 0.3) + vec3<f32>(0.05, 0.08, 0.1);
}

fn get_leaves_color(world_pos: vec3<f32>, base: vec3<f32>) -> vec3<f32> {
    let n = fbm3(world_pos * 1.5, 3);
    let n2 = fbm3(world_pos * 4.0 + vec3<f32>(10.0), 2);
    let density = step(0.3, n);
    let color = base * (0.6 + n2 * 0.4);
    // Darken interior
    return mix(vec3<f32>(0.08, 0.2, 0.05), color, density);
}

// --- Ambient Occlusion ---
fn get_voxel(vx: i32, vy: i32, vz: i32, grid_size: vec3<i32>) -> u32 {
    if vx < 0 || vx >= grid_size.x || vy < 0 || vy >= grid_size.y || vz < 0 || vz >= grid_size.z {
        return 0u;
    }
    let tex_coord = vec3<i32>(vx, grid_size.y - 1 - vy, vz);
    return textureLoad(voxel_texture, tex_coord, 0).r;
}

fn compute_ao(pos: vec3<i32>, normal: vec3<f32>, grid_size: vec3<i32>) -> f32 {
    var ao = 1.0;
    let tangent = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
    let bitangent = cross(normal, tangent);
    
    // Check neighbors in corner directions
    let corner_dirs = array<vec3<i32>, 4>(
        vec3<i32>(1, 0, 1),
        vec3<i32>(-1, 0, 1),
        vec3<i32>(1, 0, -1),
        vec3<i32>(-1, 0, -1),
    );
    
    // Simplified AO: check adjacent voxels
    let adj1 = get_voxel(pos.x + i32(normal.x), pos.y + i32(normal.y), pos.z + i32(normal.z), grid_size);
    let adj2 = get_voxel(pos.x + i32(sign(normal.x + 0.1)), pos.y + i32(sign(normal.y + 0.1)), pos.z + i32(sign(normal.z + 0.1)), grid_size);
    
    // If neighbors exist, darken
    if adj1 > 0u { ao = ao - 0.15; }
    if adj2 > 0u { ao = ao - 0.1; }
    
    return clamp(ao, 0.4, 1.0);
}

fn compute_edge_bevel(world_pos: vec3<f32>, voxel_pos: vec3<i32>, normal: vec3<f32>, grid_size: vec3<i32>) -> f32 {
    let local = fract(world_pos);
    let edge_dist = min(min(local.x, local.y), local.z);
    let edge_dist2 = min(min(1.0 - local.x, 1.0 - local.y), 1.0 - local.z);
    let min_edge = min(edge_dist, edge_dist2);
    
    // Subtle edge highlight
    let edge_glow = smoothstep(0.0, 0.08, min_edge);
    return mix(0.92, 1.0, edge_glow);
}

// --- Ray Generation ---
fn generate_ray_direction(uv: vec2<f32>, yaw: f32, pitch: f32, fov: f32, aspect: f32) -> vec3<f32> {
    let x = cos(yaw) * cos(pitch);
    let y = sin(pitch);
    let z = sin(yaw) * cos(pitch);
    let front = normalize(vec3<f32>(x, y, z));
    
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    let right = normalize(cross(world_up, front));
    let up = cross(front, right);
    
    let ndc = uv * 2.0 - 1.0;
    let tan_fov = tan(fov * 0.5);
    
    return normalize(front + ndc.x * aspect * tan_fov * right + ndc.y * tan_fov * up);
}

fn ray_box_intersection(origin: vec3<f32>, dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let inv_dir = 1.0 / dir;
    let t1 = (box_min - origin) * inv_dir;
    let t2 = (box_max - origin) * inv_dir;
    let t_min = min(t1, t2);
    let t_max = max(t1, t2);
    let t_enter = max(max(t_min.x, t_min.y), t_min.z);
    let t_exit = min(min(t_max.x, t_max.y), t_max.z);
    return vec2<f32>(t_enter, t_exit);
}

// --- DDA Marching with Micro-Detail ---
fn dda_march(origin: vec3<f32>, direction: vec3<f32>, grid_size: vec3<i32>, voxel_scale: f32) -> vec4<f32> {
    let box_min = vec3<f32>(0.0);
    let box_max = vec3<f32>(f32(grid_size.x), f32(grid_size.y), f32(grid_size.z)) * voxel_scale;
    
    let box_hit = ray_box_intersection(origin, direction, box_min, box_max);
    if box_hit.y < box_hit.x || box_hit.y < 0.0 {
        return vec4<f32>(0.0);
    }
    
    var t = max(box_hit.x, 0.0);
    let max_t = box_hit.y;
    
    var pos = origin + direction * (t + 0.001);
    
    let step_dir = vec3<i32>(
        select(-1, 1, direction.x > 0.0),
        select(-1, 1, direction.y > 0.0),
        select(-1, 1, direction.z > 0.0)
    );
    
    let inv_dir = 1.0 / max(abs(direction), vec3<f32>(0.0001));
    let delta = inv_dir * voxel_scale;
    
    var voxel = vec3<i32>(floor(pos / voxel_scale));
    
    let next_boundary = vec3<f32>(voxel + step_dir) * voxel_scale;
    var side_dist = abs((next_boundary - pos) / direction);
    
    let sun_dir = normalize(vec3<f32>(0.6, 0.4, 0.5));
    let time = camera.time;
    
    for (var steps = 0u; steps < 512u; steps = steps + 1u) {
        if voxel.x < 0 || voxel.x >= grid_size.x || 
           voxel.y < 0 || voxel.y >= grid_size.y || 
           voxel.z < 0 || voxel.z >= grid_size.z {
            break;
        }
        
        let voxel_value = get_voxel(voxel.x, voxel.y, voxel.z, grid_size);
        
        if voxel_value > 0u {
            // Determine normal from hit axis
            var normal: vec3<f32>;
            if side_dist.x < side_dist.y && side_dist.x < side_dist.z {
                normal = vec3<f32>(-f32(step_dir.x), 0.0, 0.0);
            } else if side_dist.y < side_dist.z {
                normal = vec3<f32>(0.0, -f32(step_dir.y), 0.0);
            } else {
                normal = vec3<f32>(0.0, 0.0, -f32(step_dir.z));
            }
            
            let world_pos = vec3<f32>(voxel) * voxel_scale;
            let local_pos = world_pos + fract(pos / voxel_scale) * voxel_scale;
            
            // Material definition
            var base_color: vec3<f32>;
            var roughness: f32 = 0.8;
            
            switch voxel_value {
                case 1u: { 
                    base_color = get_grass_color(world_pos, vec3<f32>(0.42, 0.58, 0.24));
                    roughness = 0.95;
                }
                case 2u: { 
                    base_color = get_dirt_color(world_pos, vec3<f32>(0.50, 0.38, 0.26));
                    roughness = 0.9;
                }
                case 3u: { 
                    base_color = get_stone_brick_color(world_pos, vec3<f32>(0.52, 0.50, 0.46));
                    roughness = 0.7;
                }
                case 4u: { 
                    base_color = get_water_color(world_pos, vec3<f32>(0.18, 0.35, 0.55), time);
                    roughness = 0.05;
                }
                case 5u: { 
                    base_color = get_wood_color(world_pos, vec3<f32>(0.52, 0.36, 0.20));
                    roughness = 0.85;
                }
                case 6u: { 
                    base_color = get_leaves_color(world_pos, vec3<f32>(0.28, 0.55, 0.18));
                    roughness = 0.9;
                }
                case 7u: { 
                    base_color = vec3<f32>(0.82, 0.88, 0.92);
                    roughness = 0.6;
                }
                case 8u: { 
                    base_color = get_sand_color(world_pos, vec3<f32>(0.76, 0.70, 0.52));
                    roughness = 0.9;
                }
                case 9u: { 
                    base_color = get_cobblestone_color(world_pos, vec3<f32>(0.48, 0.45, 0.42));
                    roughness = 0.75;
                }
                case 10u: { 
                    base_color = get_mossy_stone_color(world_pos, vec3<f32>(0.50, 0.48, 0.44));
                    roughness = 0.8;
                }
                case 11u: { 
                    base_color = get_wood_color(world_pos, vec3<f32>(0.72, 0.15, 0.12)); // red torii
                    roughness = 0.7;
                }
                case 12u: { 
                    base_color = vec3<f32>(0.95, 0.65, 0.75); // pink sakura blossom
                    roughness = 0.85;
                }
                case 13u: { 
                    base_color = get_grass_color(world_pos, vec3<f32>(0.82, 0.68, 0.22)); // golden wheat
                    roughness = 0.95;
                }
                case 14u: { 
                    base_color = vec3<f32>(0.90, 0.85, 0.78); // white plaster
                    roughness = 0.6;
                }
                case 15u: { 
                    base_color = vec3<f32>(0.92, 0.92, 0.88); // paper shoji
                    roughness = 0.4;
                }
                default: { 
                    base_color = vec3<f32>(0.8, 0.3, 0.3);
                }
            }
            
            // AO
            let ao = compute_ao(voxel, normal, grid_size);
            
            // Edge bevel
            let bevel = compute_edge_bevel(pos, voxel, normal, grid_size);
            
            // Diffuse lighting
            let diffuse = max(dot(normal, sun_dir), 0.0);
            let ambient = 0.35;
            let light = mix(ambient, 1.0, diffuse * 0.65);
            
            // Specular (simple Blinn-Phong)
            let half_vec = normalize(sun_dir - direction);
            let spec_angle = max(dot(normal, half_vec), 0.0);
            let specular = pow(spec_angle, mix(8.0, 64.0, 1.0 - roughness)) * (1.0 - roughness) * 0.4;
            
            var color = base_color * light * ao * bevel;
            color = color + vec3<f32>(specular);
            
            // Distance fog
            let dist = length(origin - world_pos);
            let fog = exp(-dist * 0.006);
            let sky_color = vec3<f32>(0.75, 0.55, 0.55); // warm sunset fog
            color = mix(sky_color * 0.85, color, fog);
            
            return vec4<f32>(color, 1.0);
        }
        
        // Step to next voxel
        if side_dist.x < side_dist.y && side_dist.x < side_dist.z {
            voxel.x = voxel.x + step_dir.x;
            side_dist.x = side_dist.x + delta.x;
        } else if side_dist.y < side_dist.z {
            voxel.y = voxel.y + step_dir.y;
            side_dist.y = side_dist.y + delta.y;
        } else {
            voxel.z = voxel.z + step_dir.z;
            side_dist.z = side_dist.z + delta.z;
        }
        
        let current_t = min(min(side_dist.x, side_dist.y), side_dist.z);
        if current_t > max_t {
            break;
        }
    }
    
    return vec4<f32>(0.0);
}

fn get_sky_color(ray_dir: vec3<f32>, time: f32) -> vec3<f32> {
    let t = ray_dir.y * 0.5 + 0.5;
    // Sunset palette: warm orange/pink/purple
    var sky = mix(vec3<f32>(0.95, 0.45, 0.35), vec3<f32>(0.65, 0.25, 0.45), t);
    sky = mix(sky, vec3<f32>(0.18, 0.12, 0.30), t * t * 0.5);
    
    // Horizon glow - warm golden
    let horizon = 1.0 - abs(ray_dir.y);
    sky = sky + vec3<f32>(0.55, 0.30, 0.15) * horizon * horizon * 0.6;
    
    // Soft clouds
    let cloud_n = fbm3(ray_dir * 3.0 + vec3<f32>(time * 0.02, 0.0, 0.0), 3);
    let clouds = smoothstep(0.45, 0.65, cloud_n);
    let cloud_color = vec3<f32>(0.95, 0.80, 0.75); // pink-tinged clouds
    
    return mix(sky, cloud_color, clouds * 0.3);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(vertex_index % 2u) * 4.0 - 1.0;
    let y = f32(vertex_index / 2u) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let resolution = camera.resolution;
    let uv = frag_coord.xy / resolution;
    
    let ray_dir = generate_ray_direction(
        uv, 
        camera.yaw, 
        camera.pitch, 
        camera.fov, 
        camera.aspect
    );
    
    let grid_size = vec3<i32>(256, 128, 256);
    let voxel_scale = 0.125;
    
    let result = dda_march(camera.position, ray_dir, grid_size, voxel_scale);
    
    if result.a > 0.0 {
        return vec4<f32>(result.rgb, 1.0);
    }
    
    let sky = get_sky_color(ray_dir, camera.time);
    return vec4<f32>(sky, 1.0);
}
