#include "GLVisualModule.h"

#include "glad/glad.h"

namespace dyno
{
	GLVisualModule::GLVisualModule()
	{
		this->setName("GLVisualModule");

		this->varMetallic()->setRange(0, 1);
		this->varRoughness()->setRange(0, 1);
		this->varAlpha()->setRange(0, 1);
	}

	GLVisualModule::~GLVisualModule()
	{
		if (isGLInitialized) {
			printf("Warning: %s not released!\n", getName().c_str());
		}
	}

	void GLVisualModule::updateGraphicsContext()
	{
		//printf("UpdateGraphicsContext\n");
		//if (!this->isGLInitialized)
		//{
		//	if (!gladLoadGL()) {
		//		printf("Failed to load OpenGL context!\n");
		//		exit(-1);
		//	}

		//	isGLInitialized = initializeGL();

		//	if (!this->isGLInitialized)
		//		return;
		//}

		//this->updateGL();

		this->changed = clock::now();
	}

	void GLVisualModule::setColor(const Vec3f& color)
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

	void GLVisualModule::draw(GLRenderPass pass)
	{
		if (!this->validateInputs() || !this->isVisible()) {
			return;
		}

		if (!this->isGLInitialized) {
			this->isGLInitialized = this->initializeGL();
		}

		if (this->isGLInitialized) {
			// check update
			if (changed > updated) {
				this->updateGL();
				updated = clock::now();
			}

			this->paintGL(pass);
		}

	}

}


