#pragma once

#include <glm/glm.hpp>

namespace dyno
{
	
	struct RenderParams
	{

		// camera matrices
		glm::mat4	view;
		glm::mat4	proj;

		// viewport
		struct Viewport
		{
			unsigned int x = 0;
			unsigned int y = 0;
			unsigned int w;
			unsigned int h;
		} viewport;

		// illumination settings
		struct Light
		{
			// ambient light
			glm::vec3	ambientColor = glm::vec3(0.05f);
			float		ambientScale = 1.f;
			// directional light
			glm::vec3	mainLightColor = glm::vec3(1.f);
			float		mainLightScale = 5.f;
			glm::vec3	mainLightDirection = glm::vec3(0.4f, 0.6f, 0.8f);
			float		_padding = 0.f;
			glm::mat4	mainLightVP = glm::mat4(1);
		} light;


		glm::vec3	bgColor0 = glm::vec3(0.2f);
		glm::vec3	bgColor1 = glm::vec3(0.8f);

		// some render options...
		bool  showGround = true;
		float groudScale = 3.f;

		bool showAxisHelper = true;
		bool showSceneBounds = true;
	};
}