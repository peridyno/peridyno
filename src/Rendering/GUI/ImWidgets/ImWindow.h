#pragma once

#include <vector>
#include <memory>

namespace dyno
{
	struct Picture;
	class RenderEngine;
	class SceneGraph;

	class ImWindow
	{
	public:
		void initialize(float scale);
		void draw(RenderEngine* engine, SceneGraph* scene);

		bool cameraLocked();

	private:
		bool mDisenableCamera = false;
		std::vector<std::shared_ptr<Picture>> mPics;
	};
}
