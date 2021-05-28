#pragma once

#include <glm/glm.hpp>

class AxisRenderer;
class BBoxRenderer;
class GroundRenderer;
class RenderHelper
{
public:
	RenderHelper();
	~RenderHelper();

	void initialize();

	void drawGround(float scale = 3.f);
	void drawAxis(float lineWidth = 2.f);
	void drawBBox(glm::vec3 pmin, glm::vec3 pmax, int type = 0);

private:

	AxisRenderer*	mAxisRenderer;
	BBoxRenderer*	mBBoxRenderer;
	GroundRenderer* mGroundRenderer;
};