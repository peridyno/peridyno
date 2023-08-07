vec3 ReinhardTonemap(vec3 v)
{
	return v / (1.0f + v);
}

vec3 GammaCorrect(vec3 v)
{
	float gamma = 2.2;
	return pow(v, vec3(1.0 / gamma));
}
