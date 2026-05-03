// Mesh Vertex/Fragment Shader
// Traditional rasterization with diffuse lighting + sky fog

struct Uniforms {
  viewProj: mat4x4<f32>,
  sunDir: vec3<f32>,
};

@binding(0) @group(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec4<f32>,
  @location(3) uv: vec2<f32>,
};

struct VertexOutput {
  @builtin(position) clipPosition: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.clipPosition = uniforms.viewProj * vec4<f32>(input.position, 1.0);
  output.worldPos = input.position;
  output.normal = input.normal;
  output.color = input.color;
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  // Diffuse lighting
  let sunDir = normalize(-uniforms.sunDir);
  let normal = normalize(input.normal);
  let NdotL = max(dot(normal, sunDir), 0.0);
  let ambient = 0.42;
  let diffuse = 0.70 * NdotL;
  let lighting = ambient + diffuse;

  // Fill light
  let fillDir = vec3<f32>(-0.5, 0.3, -0.5);
  let fill = 0.15 * max(dot(normal, normalize(fillDir)), 0.0);
  let finalLight = lighting + fill;

  var color = input.color.rgb * finalLight;
  let alpha = input.color.a;

  // Distance fog blending to sky
  let fogDist = length(input.worldPos - vec3<f32>(16.0, 8.0, 16.0));
  let fogAmount = smoothstep(40.0, 90.0, fogDist);
  let fogColor = vec3<f32>(0.4, 0.6, 0.9);
  color = mix(color, fogColor, fogAmount * 0.42);

  return vec4<f32>(color, alpha);
}
