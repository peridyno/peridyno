/*

	Rendering interface

*/

#pragma once

#include <memory>
#include <string>
#include <glm/glm.hpp>

namespace dyno
{
	class SceneGraph;
	class Camera;

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

		int viewPortflag = -1;
	};

	// RenderEngine interface
	class RenderEngine
	{
	public:
		virtual void initialize(int width, int height) = 0;
		virtual void draw(SceneGraph* scene) = 0;
		virtual void resize(int w, int h) = 0;

		// TODO: re-organize
		RenderParams* renderParams() { return &m_rparams; }
		std::shared_ptr<Camera> camera() { return mCamera; }
		void setCamera(std::shared_ptr<Camera> cam) { mCamera = cam; }

		virtual std::string name() = 0;
	protected:
		RenderParams m_rparams;

		// current active camera
		std::shared_ptr<Camera> mCamera;

	};
};

