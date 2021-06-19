#include "GLVisualModule.h"

using namespace dyno;

GLVisualModule::GLVisualModule()
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

bool GLVisualModule::isTransparent() const
{
	return mAlpha < 1.f;
}
