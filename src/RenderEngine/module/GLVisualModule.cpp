#include "GLVisualModule.h"
#include "../RenderEngine.h"

#include "Framework/Node.h"

using namespace dyno;

GLVisualModule::GLVisualModule(): mShadowMode(ShadowMode::ALL)
{
	this->setName("GLVisualModule");
}

void GLVisualModule::display()
{
	if (!this->isGLInitialized)
	{		
		isGLInitialized = initializeGL();

		if (!this->isGLInitialized)
			return;
	}
}

void GLVisualModule::updateRenderingContext()
{
	if (!this->isGLInitialized)
	{		
		isGLInitialized = initializeGL(); 

		if (!this->isGLInitialized)
			return;
	}

	this->updateGL();
}


void GLVisualModule::setColor(const glm::vec3& color)
{
	mBaseColor = color;
}

void GLVisualModule::setMetallic(float m)
{
	this->mMetallic = m;
}

void GLVisualModule::setRoughness(float r)
{
	this->mRoughness = r;
}

void GLVisualModule::setAlpha(float alpha)
{
	mAlpha = alpha;
}

void GLVisualModule::setColorMapMode(ColorMapMode mode)
{
	mColorMode = mode;
}

void GLVisualModule::setColorMapRange(float vmin, float vmax)
{
	mColorMin = vmin;
	mColorMax = vmax;
}

void GLVisualModule::setShadowMode(ShadowMode mode)
{
	mShadowMode = mode;
}

bool GLVisualModule::isTransparent() const
{
	return mAlpha < 1.f;
}
