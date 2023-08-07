#include "GLVisualModule.h"

#include "glad/glad.h"

namespace dyno
{
	GLVisualModule::GLVisualModule()
		: VisualModule()
	{
		this->setName("GLVisualModule");

		this->varMetallic()->setRange(0, 1);
		this->varRoughness()->setRange(0, 1);
		this->varAlpha()->setRange(0, 1);

		this->varBaseColor()->setObjectName("Color");
	}

	GLVisualModule::~GLVisualModule()
	{
		if (isGLInitialized) {
			printf("Warning: %s not released!\n", getName().c_str());
		}
	}

	void GLVisualModule::updateImpl()
	{
		printf("Warning: %s::updateImpl is not implemented!\n", getName().c_str());
	}

	void GLVisualModule::preprocess()
	{
		updateMutex.lock();
	}

	void GLVisualModule::postprocess()
	{
		updateMutex.unlock();
		this->changed = clock::now();
	}

	void GLVisualModule::setColor(const Color& color)
	{
		this->varBaseColor()->setValue(color);
	}

	void GLVisualModule::setMetallic(float m)
	{
		this->varMetallic()->setValue(m);
	}

	void GLVisualModule::setRoughness(float r)
	{
		this->varRoughness()->setValue(r);
	}

	void GLVisualModule::setAlpha(float alpha)
	{
		this->varAlpha()->setValue(alpha);
	}

	bool GLVisualModule::isTransparent() const
	{
		// we need to copy the alpha since it doesn't provide const interface...
		auto alpha = this->var_Alpha;
		return alpha.getValue() < 1.f;
	}

	void GLVisualModule::draw(const RenderParams& rparams)
	{
		if (!this->isVisible())
			return;

		if (!isGLInitialized)
			isGLInitialized = initializeGL();

		// if failed to initialize...
		if (!isGLInitialized)
			throw std::runtime_error("Cannot initialize " + getName());

		// check update
		if (changed > updated) {
			updateMutex.lock();
			updateGL();
			updated = clock::now();
			updateMutex.unlock();
		}

		// draw
		this->paintGL(rparams);
	}

	void GLVisualModule::release()
	{
		if (isGLInitialized)
		{
			this->releaseGL();
			isGLInitialized = false;
		}
	}

}


