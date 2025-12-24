#include "GLPointVisualModule.h"
#include "GLRenderEngine.h"

#include <Utility.h>

// opengl
#include <glad/glad.h>

#include "point.vert.h"
#include "point.frag.h"

#ifdef CUDA_BACKEND
// cuda
#include <cuda_gl_interop.h>
#endif

#ifdef VK_BACKEND

#endif // VK_BACKEND


namespace dyno
{
	IMPLEMENT_CLASS(GLPointVisualModule)

	GLPointVisualModule::GLPointVisualModule()
	{
		mNumPoints = 0;
		this->setName("point_renderer");
		this->inColor()->tagOptional(true);
		this->varPointSize()->setRange(0.001f, 1.0f);
		this->varBaseColor()->setValue(Color::Grey81());
		this->varForceUpdate()->setValue(true);
	}

	GLPointVisualModule::~GLPointVisualModule()
	{
// 		points.clear();
// 		colors.clear();
// 
// 		mPosition.release();
// 		mColor.release();
	}

	void GLPointVisualModule::setColorMapMode(ColorMapMode mode)
	{
		this->varColorMode()->setCurrentKey(mode);
	}

	bool GLPointVisualModule::initializeGL()
	{
		mPosition.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mColor.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		mVertexArray.create();
		mVertexArray.bindVertexBuffer(&mPosition, 0, 3, GL_FLOAT, 0, 0, 0);
		mVertexArray.bindVertexBuffer(&mColor, 1, 3, GL_FLOAT, 0, 0, 0);

		mShaderProgram = Program::createProgramSPIRV(
			POINT_VERT, sizeof(POINT_VERT),
			POINT_FRAG, sizeof(POINT_FRAG));

		// create shader uniform buffer
		mUniformBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);

		glCheckError();

		return true;
	}

	void GLPointVisualModule::releaseGL()
	{
		mShaderProgram->release();
		delete mShaderProgram;

		mPosition.release();
		mColor.release();
		mVertexArray.release();

		mUniformBlock.release();
	}

	void GLPointVisualModule::updateGL()
	{
		mNumPoints = mPosition.count();
		if (mNumPoints == 0) return;
		//printf("GLPoints Visual Module: %d \n", mNumPoints);

		mPosition.updateGL();
		mVertexArray.bindVertexBuffer(&mPosition, 0, 3, GL_FLOAT, 0, 0, 0);

		if (this->varColorMode()->currentKey() == ColorMapMode::PER_VERTEX_SHADER
			&& !this->inColor()->isEmpty())
		{
			mColor.updateGL();
			mVertexArray.bindVertexBuffer(&mColor, 1, 3, GL_FLOAT, 0, 0, 0);
		}
		else
		{
			mVertexArray.bind();
			glDisableVertexAttribArray(1);
			mVertexArray.unbind();
		}
	}

	void GLPointVisualModule::updateImpl()
	{
		// update data
		auto pPointSet = this->inPointSet()->getDataPtr();
		auto points = pPointSet->getPoints();

		mPosition.load(points);

		if (this->varColorMode()->currentKey() == ColorMapMode::PER_VERTEX_SHADER
			&& !this->inColor()->isEmpty())
		{
			auto colors = this->inColor()->getData();
			mColor.load(colors);
		}
	}

	void GLPointVisualModule::paintGL(const RenderParams& rparams)
	{
		if (mNumPoints == 0)
			return;

		// setup uniform buffer
		mUniformBlock.load((void*)&rparams, sizeof(RenderParams));
		mUniformBlock.bindBufferBase(0);

		mShaderProgram->use();
		mShaderProgram->setFloat("uPointSize", this->varPointSize()->getValue());

		if (rparams.mode == GLRenderMode::COLOR)
		{
			mShaderProgram->setFloat("uMetallic", this->varMetallic()->getValue());
			mShaderProgram->setFloat("uRoughness", this->varRoughness()->getValue());
			mShaderProgram->setFloat("uAlpha", this->varAlpha()->getValue());
		}
		else if (rparams.mode == GLRenderMode::SHADOW)
		{

		}
		else if (rparams.mode == GLRenderMode::TRANSPARENCY)
		{
			printf("WARNING: GLPointVisualModule does not support transparency!\n");
			return;
		}
		else
		{
			printf("Unknown render pass!\n");
			return;
		}

		// per-object color color
		auto color = this->varBaseColor()->getValue();
		glVertexAttrib3f(1, color.r, color.g, color.b);

		mVertexArray.bind();
		glDrawArrays(GL_POINTS, 0, mNumPoints);
		glCheckError();
	}
}
