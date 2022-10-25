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
		mNumPoints = 0;
		this->setName("point_renderer");

		this->inColor()->tagOptional(true);
	}

	GLPointVisualModule::~GLPointVisualModule()
	{

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

		mShaderProgram = gl::ShaderFactory::createShaderProgram("point.vert", "point.frag");

		gl::glCheckError();

		return true;
	}

	void GLPointVisualModule::updateGL()
	{
		auto pPointSet = this->inPointSet()->getDataPtr();

		auto& xyz = pPointSet->getPoints();
		mNumPoints = xyz.size();

		mVertexArray.bind();
		if (mColorMode == ColorMapMode::PER_VERTEX_SHADER
			&& !this->inColor()->isEmpty() 
			&& this->inColor()->getDataPtr()->size() == mNumPoints)
		{
			auto color = this->inColor()->getData();
			mColor.loadCuda(color.begin(), mNumPoints * sizeof(float) * 3);
			mVertexArray.bindVertexBuffer(&mColor, 1, 3, GL_FLOAT, 0, 0, 0);
		}
		else
		{
			glDisableVertexAttribArray(1);
		}
		mVertexArray.unbind();


		mPosition.loadCuda(xyz.begin(), mNumPoints * sizeof(float) * 3);
	}

	void GLPointVisualModule::paintGL(GLRenderPass pass)
	{
		if (mNumPoints == 0)
			return;

		mShaderProgram.use();
		mShaderProgram.setFloat("uPointSize", this->varPointSize()->getData());

		unsigned int subroutine;
		if (pass == GLRenderPass::COLOR)
		{
			mShaderProgram.setVec3("uBaseColor", this->varBaseColor()->getData());
			mShaderProgram.setFloat("uMetallic", this->varMetallic()->getData());
			mShaderProgram.setFloat("uRoughness", this->varRoughness()->getData());
			mShaderProgram.setFloat("uAlpha", this->varAlpha()->getData());

			mShaderProgram.setInt("uColorMode", mColorMode);
			mShaderProgram.setFloat("uColorMin", mColorMin);
			mShaderProgram.setFloat("uColorMax", mColorMax);

			subroutine = 0;
			glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
		}
		else if (pass == GLRenderPass::SHADOW)
		{
			subroutine = 1;
			glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
		}
		else
		{
			printf("Unknown render pass!\n");
			return;
		}

		// per-object color color
		auto color = this->varBaseColor()->getData();
		glVertexAttrib3f(1, color[0], color[1], color[2]);

		mVertexArray.bind();
		glDrawArrays(GL_POINTS, 0, mNumPoints);
		gl::glCheckError();
	}
}
