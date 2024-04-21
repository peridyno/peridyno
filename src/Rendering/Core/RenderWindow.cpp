#include "RenderWindow.h"

#include "OrbitCamera.h"
#include "TrackballCamera.h"

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
