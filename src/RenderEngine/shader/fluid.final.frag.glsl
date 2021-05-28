#version 440

in vec2		texcoord;
out vec4	fragOut;

layout(binding = 0) uniform sampler2D uFluidTex;
layout(binding = 1) uniform sampler2D uColorTex;

uniform int viewportWidth;
uniform int viewportHeight;

layout (std140, binding=0) uniform TransformUniformBlock
{
	mat4 model;
	mat4 view;
	mat4 proj;
} transform;

mat4 DCVCMatrix;
mat4 VCDCMatrix;

vec3 uvToEye(vec2 texCoordVal, float vcDepth)
{
  // need to convert depth back to DC, to use
  vec4 tmp = vec4(0.0, 0.0, vcDepth, 1.0);
  tmp = VCDCMatrix*tmp;
  tmp.x = texCoordVal.x * 2.0 - 1.0;
  tmp.y = texCoordVal.y * 2.0 - 1.0;
  tmp.z = tmp.z/tmp.w;
  tmp.w = 1.0;
  vec4 viewPos = DCVCMatrix * tmp;
  return viewPos.xyz / viewPos.w;
}

vec3 getSurfaceNormal()
{
  float x     = texcoord.x;
  float y     = texcoord.y;
  float depth = texture(uFluidTex, vec2(x, y)).r;

  float pixelWidth  = 1.0 / float(viewportWidth);
  float pixelHeight = 1.0 / float(viewportHeight);
  float xp          = texcoord.x + pixelWidth;
  float xn          = texcoord.x - pixelWidth;
  float yp          = texcoord.y + pixelHeight;
  float yn          = texcoord.y - pixelHeight;

  float depthxp = texture(uFluidTex, vec2(xp, y)).r;
  float depthxn = texture(uFluidTex, vec2(xn, y)).r;
  float depthyp = texture(uFluidTex, vec2(x, yp)).r;
  float depthyn = texture(uFluidTex, vec2(x, yn)).r;

  vec3 position   = uvToEye(vec2(x, y), depth);
  vec3 positionxp = uvToEye(vec2(xp, y), depthxp);
  vec3 positionxn = uvToEye(vec2(xn, y), depthxn);
  vec3 dxl        = position - positionxn;
  vec3 dxr        = positionxp - position;

  vec3 dx = (abs(dxr.z) < abs(dxl.z)) ? dxr : dxl;

  vec3 positionyp = uvToEye(vec2(x, yp), depthyp);
  vec3 positionyn = uvToEye(vec2(x, yn), depthyn);
  vec3 dyb        = position - positionyn;
  vec3 dyt        = positionyp - position;

  vec3 dy = (abs(dyt.z) < abs(dyb.z)) ? dyt : dyb;

  vec3 N = normalize(cross(dx, dy));
  if(isnan(N.x) || isnan(N.y) || isnan(N.y) ||
      isinf(N.x) || isinf(N.y) || isinf(N.z))
  {
    N = vec3(0, 0, 1);
  }

  return N;
}


subroutine void MainRoutine(void);
layout(location = 0) subroutine uniform MainRoutine _main;
void main(void) { 
	VCDCMatrix = transform.proj;
	DCVCMatrix = inverse(VCDCMatrix);
	_main();
} 

layout(index = 1) subroutine(MainRoutine) void blend(void)
{
	vec4 pFluid = texture(uFluidTex, texcoord);
	if(pFluid.a > 0.0)
	{
		fragOut = pFluid;
	}
	else
	{
		discard;
	}
}

uniform float refractionScale      = 0.1f;
uniform float attenuationScale     = 20.0f;
uniform float additionalReflection = 0.1f;
uniform float refractiveIndex      = 1.33f;

uniform vec3 fluidAttenuationColor = vec3(0.5f, 0.2f, 0.05f);

// This should not be changed
const float fresnelPower = 5.0f;

vec3 computeAttenuation(float thickness)
{
  return vec3(exp(-fluidAttenuationColor.r * thickness),
              exp(-fluidAttenuationColor.g * thickness),
              exp(-fluidAttenuationColor.b * thickness));
}

layout(index = 0) subroutine(MainRoutine) void render(void)
{
	vec4 pFluid = texture(uFluidTex, texcoord);
	vec4 pColor = texture(uColorTex, texcoord);

	fragOut = pColor;

	if(pFluid.b > 0.0)
	{
		float fdepth = pFluid.r;
		
		vec3 position = uvToEye(texcoord, fdepth);
		vec3 viewer   = normalize(-position.xyz);

		vec3 N = getSurfaceNormal(); 
		
		vec3  reflectionColor = vec3(1.0,1.0,1.0);

		float eta = 1.0 / refractiveIndex;          // Ratio of indices of refraction
		float F   = ((1.0 - eta) * (1.0 - eta)) / ((1.0 + eta) * (1.0 + eta));

		//Fresnel Reflection
		float fresnelRatio  = clamp(F + (1.0 - F) * pow((1.0 - dot(viewer, N)), fresnelPower), 0, 1);
		vec3  reflectionDir = reflect(-viewer, N);

		float fthick = pFluid.a * attenuationScale;
		vec3 volumeColor = computeAttenuation(fthick);

		vec3 refractionDir   = refract(-viewer, N, eta);
		vec3 refractionColor = volumeColor * 
			texture(uColorTex, texcoord + refractionDir.xy * refractionScale).xyz;

		fresnelRatio = mix(fresnelRatio, 1.0, additionalReflection);
		vec3 finalColor = mix(refractionColor, reflectionColor, fresnelRatio);		
		
		fragOut = vec4(finalColor, 1);
	}
}

