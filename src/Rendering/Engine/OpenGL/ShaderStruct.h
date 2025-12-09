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

		glm::vec3 color;
		float roughness = 0.5;
		float metallic = 0.0;
		float alpha = 1.0;
		float ao = 0.0;
	};
}

