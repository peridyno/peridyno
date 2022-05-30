#include "GLPointVisualModule.h"
#include "GLRenderEngine.h"

#include <Utility.h>
#include <RenderTools.h>

// opengl
#include <glad/glad.h>
// cuda
#include <cuda_gl_interop.h>

namespace dyno
{
	IMPLEMENT_CLASS(GLPointVisualModule)

	GLPointVisualModule::GLPointVisualModule()
	{
		mPointSize = 0.001f;
		mNumPoints = 1;
		this->setName("point_renderer");
		this->inColor()->tagOptional(true);
	}

	GLPointVisualModule::~GLPointVisualModule()
	{
		mColorBuffer.clear();
	}

	void GLPointVisualModule::setPointSize(float size)
	{
		mPointSize = size;
	}

	float GLPointVisualModule::getPointSize() const
	{
		return mPointSize;
	}

	void GLPointVisualModule::setColorMapMode(ColorMapMode mode)
	{
		mColorMode = mode;
	}

	void GLPointVisualModule::setColorMapRange(float vmin, float vmax)
	{
		mColorMin = vmin;
		mColorMax = vmax;
	}

	bool GLPointVisualModule::initializeGL()
	{
		mPosition.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mColor.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		mVertexArray.create();
		mVertexArray.bindVertexBuffer(&mPosition, 0, 3, GL_FLOAT, 0, 0, 0);
		mVertexArray.bindVertexBuffer(&mColor, 1, 3, GL_FLOAT, 0, 0, 0);

		mShaderProgram = gl::CreateShaderProgram("point.vert", "point.frag");

		gl::glCheckError();

		return true;
	}


	void GLPointVisualModule::updateGL()
	{
		auto pPointSet = this->inPointSet()->getDataPtr();

		auto& xyz = pPointSet->getPoints();
		mNumPoints = xyz.size();

		if (mColorBuffer.size() != mNumPoints) {
			mColorBuffer.resize(mNumPoints);
		}

		if (mColorMode == ColorMapMode::PER_OBJECT_SHADER)
		{
			RenderTools::setupColor(mColorBuffer, mBaseColor);
		}
		else
		{
			if (!this->inColor()->isEmpty() && this->inColor()->getDataPtr()->size() == mNumPoints)
			{
				mColorBuffer.assign(this->inColor()->getData());
			}
			else 
			{
				RenderTools::setupColor(mColorBuffer, mBaseColor);
			}
		}

		mPosition.loadCuda(xyz.begin(), mNumPoints * sizeof(float) * 3);
		mColor.loadCuda(mColorBuffer.begin(), mNumPoints * sizeof(float) * 3);
	}

	void GLPointVisualModule::paintGL(RenderPass pass)
	{
		mShaderProgram.use();
		mShaderProgram.setFloat("uPointSize", this->getPointSize());

		unsigned int subroutine;
		if (pass == RenderPass::COLOR)
		{
			mShaderProgram.setVec3("uBaseColor", mBaseColor);
			mShaderProgram.setFloat("uMetallic", mMetallic);
			mShaderProgram.setFloat("uRoughness", mRoughness);
			mShaderProgram.setFloat("uAlpha", mAlpha);	// not implemented!

			mShaderProgram.setInt("uColorMode", mColorMode);
			mShaderProgram.setFloat("uColorMin", mColorMin);
			mShaderProgram.setFloat("uColorMax", mColorMax);

			subroutine = 0;
			glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
		}
		else if (pass == RenderPass::SHADOW)
		{
			subroutine = 1;
			glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
		}
		else
		{
			printf("Unknown render pass!\n");
			return;
		}

		mVertexArray.bind();
		glDrawArrays(GL_POINTS, 0, mNumPoints);
		gl::glCheckError();
	}
}
