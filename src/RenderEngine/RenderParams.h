#pragma once

#include <glm/glm.hpp>

namespace dyno
{
	
	struct RenderParams
	{
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
		};

		struct Camera
		{
			glm::vec3 eye = glm::vec3(0, 0, 2);
			glm::vec3 target = glm::vec3(0);
			glm::vec3 up = glm::vec3(0, 1, 0);

			float aspect = 1.f;
			float y_fov = glm::radians(45.f);
			float z_min = 0.01f;
			float z_max = 20.f;
		};

		struct Viewport
		{
			unsigned int x = 0;
			unsigned int y = 0;
			unsigned int w;
			unsigned int h;
		};

		Light		light;
		Camera		camera;
		Viewport	viewport;

		glm::vec3	bgColor = glm::vec3(0.2f);

		// some render options...
		bool showGround = true;
		float groudScale = 3.f;

		bool showAxisHelper = true;
		bool showSceneBounds = true;
	};
}