#pragma once

#include <vector>
#include <memory>

#include "Module/MouseInputModule.h"

namespace dyno
{
	class AppBase;
	class SceneGraph;

	class ImWindow
	{
	public:
		void initialize(float scale);
		void draw(AppBase* app);

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

		PButtonType mButtonType = BT_UNKOWN;
		PActionType	mButtonAction = AT_UNKOWN;
		PModifierBits mButtonMode = MB_NO_MODIFIER;
	};
}
