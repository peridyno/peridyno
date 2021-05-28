#version 440

in vec2		texcoord;
out vec4	fragOut;

layout(binding = 0) uniform sampler2D uFluidTex;

uniform int       viewportWidth;
uniform int       viewportHeight;

uniform int       filterSize = 5;
uniform float     sigmaDepth = 10.0f;


float compute_weight2D(vec2 r, float two_sigma2)
{
  return exp(-dot(r, r) / two_sigma2);
}

float compute_weight1D(float r, float two_sigma2)
{
  return exp(-r * r / two_sigma2);
}

float filterDepth(float pixelDepth, float particleRadius)
{
    vec2 blurRadius  = vec2(1.0 / float(viewportWidth), 1.0 / float(viewportHeight));

	float finalDepth;

    float sigma      = filterSize / 3.0f;
    float two_sigma2 = 2.0f * sigma * sigma;

    float threshold       = particleRadius * sigmaDepth;
    float sigmaDepth      = threshold / 3.0f;
    float two_sigmaDepth2 = 2.0f * sigmaDepth * sigmaDepth;

    vec4 f_tex = texcoord.xyxy;
    vec2 r     = vec2(0, 0);
    vec4 sum4  = vec4(pixelDepth, 0, 0, 0);
    vec4 wsum4 = vec4(1, 0, 0, 0);
    vec4 sampleDepth;
    vec4 w4_r;
    vec4 w4_depth;
    vec4 rDepth;

    for(int x = 1; x <= filterSize; ++x)
    {
      r.x     += blurRadius.x;
      f_tex.x += blurRadius.x;
      f_tex.z -= blurRadius.x;
      vec4 f_tex1 = f_tex.xyxy;
      vec4 f_tex2 = f_tex.zwzw;

      for(int y = 1; y <= filterSize; ++y)
      {
        r.y += blurRadius.y;

        f_tex1.y += blurRadius.y;
        f_tex1.w -= blurRadius.y;
        f_tex2.y += blurRadius.y;
        f_tex2.w -= blurRadius.y;

        sampleDepth.x = texture(uFluidTex, f_tex1.xy).r;
        sampleDepth.y = texture(uFluidTex, f_tex1.zw).r;
        sampleDepth.z = texture(uFluidTex, f_tex2.xy).r;
        sampleDepth.w = texture(uFluidTex, f_tex2.zw).r;

        rDepth     = sampleDepth - vec4(pixelDepth);
        w4_r       = vec4(compute_weight2D(blurRadius * r, two_sigma2));
        w4_depth.x = compute_weight1D(rDepth.x, two_sigmaDepth2);
        w4_depth.y = compute_weight1D(rDepth.y, two_sigmaDepth2);
        w4_depth.z = compute_weight1D(rDepth.z, two_sigmaDepth2);
        w4_depth.w = compute_weight1D(rDepth.w, two_sigmaDepth2);

        sum4  += sampleDepth * w4_r * w4_depth;
        wsum4 += w4_r * w4_depth;
      }
    }

    vec2 filterVal;
    filterVal.x = dot(sum4, vec4(1, 1, 1, 1));
    filterVal.y = dot(wsum4, vec4(1, 1, 1, 1));
    finalDepth = filterVal.x / filterVal.y;
    return finalDepth;
}

float filterThickness(float fthick)
{
    vec2  blurRadius = vec2(1.0 / float(viewportWidth), 1.0 / float(viewportHeight));
    float sigma      = float(filterSize) / 3.0;
    float two_sigma2 = 2.0 * sigma * sigma;

    vec4 f_tex = texcoord.xyxy;
    vec2 r     = vec2(0, 0);
    vec4 sum4  = vec4(fthick, 0, 0, 0);
    vec4 wsum4 = vec4(1, 0, 0, 0);
    vec4 sampleThick;
    vec4 w4_r;

    for(int x = 1; x <= filterSize; ++x)
    {
        r.x     += blurRadius.x;
        f_tex.x += blurRadius.x;
        f_tex.z -= blurRadius.x;
        vec4 f_tex1 = f_tex.xyxy;
        vec4 f_tex2 = f_tex.zwzw;

        for(int y = 1; y <= filterSize; ++y)
        {
            r.y += blurRadius.y;
            w4_r = vec4(compute_weight2D(blurRadius * r, two_sigma2));

            f_tex1.y += blurRadius.y;
            f_tex1.w -= blurRadius.y;
            f_tex2.y += blurRadius.y;
            f_tex2.w -= blurRadius.y;

            sampleThick.x = texture(uFluidTex, f_tex1.xy).a;
            sampleThick.y = texture(uFluidTex, f_tex1.zw).a;
            sampleThick.z = texture(uFluidTex, f_tex2.xy).a;
            sampleThick.w = texture(uFluidTex, f_tex2.zw).a;
            sum4         += sampleThick * w4_r;
            wsum4        += w4_r;
        }
    }
    vec2 filteredThickness;
    filteredThickness.x = dot(sum4, vec4(1, 1, 1, 1));
    filteredThickness.y = dot(wsum4, vec4(1, 1, 1, 1));
    return filteredThickness.x / filteredThickness.y;
}

void main(void)
{
	vec4 pixel = texture(uFluidTex, texcoord);
        
	fragOut = pixel;

	if(pixel.b <= 0.0)
	{
		return;
	}

    fragOut.r = filterDepth(pixel.r, pixel.b);
    fragOut.a = filterThickness(pixel.a);
}