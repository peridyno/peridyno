// shader for Post Process
#version 440

#extension GL_GOOGLE_include_directive: enable

layout(location = 0) out vec4  fragColor;
layout(location = 0) in vec2 texCoord;

//layout(location = 0) uniform sampler2D screenTexture;

//layout(location = 1) uniform vec3 colorBalance; // (redBalance, greenBalance, blueBalance)

//layout(location = 2) uniform float contrast;

//layout(location = 3) uniform float saturation;

vec3 AdjustSaturation(vec3 color, float sat)
{
    float gray = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(gray), color, sat);
}

void main()
{
    //vec3 color = texture(screenTexture, texCoord).rgb;

   // color.r = clamp(color.r + colorBalance.r, 0.0, 1.0);
   // color.g = clamp(color.g + colorBalance.g, 0.0, 1.0);
   // color.b = clamp(color.b + colorBalance.b, 0.0, 1.0);

   // color = (color - 0.5) * contrast + 0.5;
   // color = clamp(color, 0.0, 1.0);

  //  color = AdjustSaturation(color, saturation);
  //  color = clamp(color, 0.0, 1.0);

    //fragColor = vec4(color, 1.0);
    fragColor = vec4(0.0,1.0,0.0, 1.0);
}