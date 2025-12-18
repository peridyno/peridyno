#pragma once
#include "Field.h";

namespace dyno 
{
	struct PBRMaterial
	{
		int tintColor = 0;
		int useAOTex = 0;
		int useRoughnessTex = 0;
		int useMetallicTex = 0;

		int useEmissiveTex = 0;
		int useAlphaTex = 0;
		int tempInt1 = 0;
		int tempInt2 = 0;

		glm::vec3 color;
		float roughness = 0.5;
		float metallic = 0.0;
		float alpha = 1.0;
		float ao = 0.0;
		
		float EmissiveIntensity = 0;
		float tempFloat1 = 0;
		float tempFloat2 = 0;
		float tempFloat3 = 0;

	};
}

