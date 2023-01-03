/*

	Rendering interface

*/

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <glm/glm.hpp>

namespace dyno
{
	class SceneGraph;
	class Node;

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

		// framebuffer size
		unsigned int width  = 1;
		unsigned int height = 1;

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
			float		mainLightShadow = 1.f;	// 0 - disable shadow; otherwise enable shadow

			// camera light
			glm::vec3	cameraLightColor = glm::vec3(0.1f);
			float		cameraLightScale = 1.f;
		} light;

		 
		// Backcolor gray scale
		glm::vec3	bgColor0 = glm::vec3(0.2f);
		glm::vec3	bgColor1 = glm::vec3(0.8f);

		// some render options...
		bool  showGround = true;
		float planeScale = 3.f;
		float rulerScale = 1.f;

		bool showAxisHelper = true;
		bool showSceneBounds = false;
	};

	struct SelectionItem {
		Node* node = 0;
		int   instance = -1;
	};

	// RenderEngine interface
	class RenderEngine
	{
	public:
		virtual void initialize() = 0;
		virtual void terminate() = 0;

		virtual void draw(SceneGraph* scene, const RenderParams& rparams) = 0;
		virtual std::vector<SelectionItem> select(int x, int y, int w, int h) = 0;

		virtual std::string name() const = 0;
	};
};

