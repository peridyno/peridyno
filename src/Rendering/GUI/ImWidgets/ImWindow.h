#pragma once

#include <vector>
#include <memory>

namespace dyno
{
	class AppBase;
	class SceneGraph;

	class ImWindow
	{
	public:
		void initialize(float scale);
		void draw(AppBase* app);

	public:
		bool cameraLocked();

	private:
		bool mDisenableCamera = false;
	};
}
