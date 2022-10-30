#pragma once
#include <string>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace px
{
	// SSBO cloth grid particle declaration
	struct Particle {
		glm::vec4 pos;
		glm::vec4 vel;
		glm::vec4 uv;
		glm::vec4 normal;
		float pinned;
		glm::vec3 _pad0;
	};



	struct computeUBO {
		float deltaT = 0.0f;
		float particleMass = 0.1f;
		float springStiffness = 2000.0f;
		float damping = 0.25f;
		float restDistH;
		float restDistV;
		float restDistD;
		float sphereRadius = 1.0f;
		glm::vec4 spherePos = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
		glm::vec4 gravity = glm::vec4(0.0f, 9.8f, 0.0f, 0.0f);
		glm::ivec2 particleCount;
	};

	struct ClothRes {
		glm::uvec2 gridsize = glm::uvec2(60, 60);
		glm::vec2 size = glm::vec2(5.0f);
	};
}