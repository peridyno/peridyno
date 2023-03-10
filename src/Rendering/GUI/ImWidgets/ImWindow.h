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

	private:
		bool mDisenableCamera = false;

		int mRegX = -1;
		int mRegY = -1;

		int mCurX = -1;
		int mCurY = -1;

		bool mDrawingBox = false;

		int  mEditMode = 0;	// 0 - translate, 1 - scale, 2 - rotate

		PButtonType mButtonType = BT_UNKOWN;
		PActionType	mButtonAction = AT_UNKOWN;
		PModifierBits mButtonMode = MB_NO_MODIFIER;
	};
}
