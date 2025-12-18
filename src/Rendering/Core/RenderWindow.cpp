#include "RenderWindow.h"

#include "OrbitCamera.h"
#include "TrackballCamera.h"

#include <iostream>
#include <sstream>

dyno::RenderWindow::RenderWindow()
{
	// create a default camera
	mCamera = std::make_shared<OrbitCamera>();
	mCamera->setWidth(64);
	mCamera->setHeight(64);
	mCamera->registerPoint(0, 0);
	mCamera->rotateToPoint(-32, 12);
}

void dyno::RenderWindow::setWindowSize(int w, int h)
{
	mCamera->setWidth(w);
	mCamera->setHeight(h);
}

void dyno::RenderWindow::toggleImGUI()
{
	mShowImWindow = !mShowImWindow;
}

void dyno::RenderWindow::saveScreen(unsigned int frame)
{
	if (frame % mSaveScreenInterval == 0)
	{
		std::stringstream adaptor;
		adaptor << frame / mSaveScreenInterval;
		std::string index_str;
		adaptor >> index_str;
		std::string file_name = mScreenRecordingPath + std::string("screen_capture_") + index_str + std::string(".bmp");

		this->onSaveScreen(file_name);
	}
}

void dyno::RenderWindow::setMainLightDirection(glm::vec3 dir)
{
	mRenderParams.light.mainLightDirection = -dir;
}


const dyno::Selection& dyno::RenderWindow::select(int x, int y, int w, int h)
{
	selectedObject = mRenderEngine->select(x, y, w, h);

	if (selectedObject.items.size() > 0)
		onSelected(selectedObject);

	return selectedObject;
}

void dyno::RenderWindow::select(std::shared_ptr<Node> node, int instance, int primitive)
{
	selectedObject = dyno::Selection();
	selectedObject.items.push_back({ node, instance, primitive});
}

std::shared_ptr<dyno::Node> dyno::RenderWindow::getCurrentSelectedNode()
{
	if(selectedObject.items.empty())
		return 0;

	return selectedObject.items[0].node;
}

void dyno::RenderWindow::onSelected(const dyno::Selection& s)
{
	printf("Select: (%d, %d), %d x %d, %u items\n", s.x, s.y, s.w, s.h, s.items.size());
}
