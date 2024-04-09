#pragma once

#include <vector>
#include <memory>

#include "Module/MouseInputModule.h"

namespace dyno
{
	class RenderWindow;
	class SceneGraph;

	class ImWindow
	{
	public:
		void initialize(float scale);
		void draw(RenderWindow* app);

		void mousePressEvent(const PMouseEvent& event);
		void mouseReleaseEvent(const PMouseEvent& event);
		void mouseMoveEvent(const PMouseEvent& event);

	public:
		bool cameraLocked();

	private:
		void drawSelectedRegion();
		void drawNodeManipulator(std::shared_ptr<Node> node, glm::mat4 view, glm::mat4 proj);
		void drawViewManipulator(Camera* camera);

	private:
		bool mDisenableCamera = false;
		bool mViewManipulator = true;


		int mEditMode = 0;	// 0 - translate, 1 - scale, 2 - rotate

		// cursor status
		int mRegX = -1;
		int mRegY = -1;
		int mCurX = -1;
		int mCurY = -1;
		PButtonType mButtonType = BT_UNKOWN;
		PActionType	mButtonAction = AT_UNKOWN;
		PModifierBits mButtonMode = MB_NO_MODIFIER;
	};
}
