#include "GLSurfaceVisualModule.h"

// opengl
#include <glad/glad.h>
#include "GLRenderEngine.h"
#include "Utility.h"

#include <RenderTools.h>

namespace dyno
{
	IMPLEMENT_CLASS(GLSurfaceVisualModule)

	GLSurfaceVisualModule::GLSurfaceVisualModule()
	{
		this->setName("surface_renderer");

		this->inColor()->tagOptional(true);
	}

	std::string GLSurfaceVisualModule::caption()
	{
		return "Surface Visual Module";
	}

	bool GLSurfaceVisualModule::initializeGL()
	{
		// create vertex buffer and vertex array object
		mVAO.create();
		mIndexBuffer.create(GL_ELEMENT_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mVertexBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mColor.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mNormalBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		mVAO.bindIndexBuffer(&mIndexBuffer);
		mVAO.bindVertexBuffer(&mVertexBuffer, 0, 3, GL_FLOAT, 0, 0, 0);
		mVAO.bindVertexBuffer(&mColor, 1, 3, GL_FLOAT, 0, 0, 0);
		mVAO.bindVertexBuffer(&mNormalBuffer, 2, 3, GL_FLOAT, 0, 0, 0);

		
		// create shader program
		if (!this->varUsePhongShadingModel()->getData())
		{
			mShaderProgram = gl::CreateShaderProgram("surface.vert", "surface.frag", "surface.geom");
		}
		else
		{
			mShaderProgram = gl::CreateShaderProgram("surface.vert", "surface.frag");
		}

		return true;
	}

	void GLSurfaceVisualModule::updateGL()
	{
		auto triSet = this->inTriangleSet()->getDataPtr();

		auto& triangles = triSet->getTriangles();
		auto& vertices = triSet->getPoints();

		auto vSize = vertices.size();

		mDrawCount = triangles.size() * 3;

		if (mColorBuffer.size() != vertices.size()) {
			mColorBuffer.resize(vertices.size());
		}

		if (this->varColorMode()->getValue() == 0)
		{
			RenderTools::setupColor(mColorBuffer, this->varBaseColor()->getData());
		}
		else
		{
			if (!this->inColor()->isEmpty() && this->inColor()->getDataPtr()->size() == vSize)
			{
				mColorBuffer.assign(this->inColor()->getData());
			}
			else
			{
				RenderTools::setupColor(mColorBuffer, this->varBaseColor()->getData());
			}
		}

		mVertexBuffer.loadCuda(vertices.begin(), vertices.size() * sizeof(float) * 3);
		mIndexBuffer.loadCuda(triangles.begin(), triangles.size() * sizeof(unsigned int) * 3);
		mColor.loadCuda(mColorBuffer.begin(), mColorBuffer.size() * sizeof(float) * 3);

		if(this->varUsePhongShadingModel()->getData())
		{
			if (triSet->outVertexNormal()->isEmpty())
				triSet->updateVertexNormal();

			auto& normals = triSet->outVertexNormal()->getData();
			mNormalBuffer.loadCuda(normals.begin(), normals.size() * sizeof(float) * 3);
		}
	}

	void GLSurfaceVisualModule::paintGL(RenderPass mode)
	{
		mShaderProgram.use();

		unsigned int subroutine;
		if (mode == RenderPass::COLOR)
		{
			mShaderProgram.setVec3("uBaseColor", this->varBaseColor()->getData());
			mShaderProgram.setFloat("uMetallic", mMetallic);
			mShaderProgram.setFloat("uRoughness", mRoughness);
			mShaderProgram.setFloat("uAlpha", mAlpha);	// not implemented!

			subroutine = 0;
			glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
		}
		else if (mode == RenderPass::SHADOW)
		{
			subroutine = 1;
			glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
		}
		else
		{
			printf("Unknown render mode!\n");
			return;
		}

		mVAO.bind();
		glDrawElements(GL_TRIANGLES, mDrawCount, GL_UNSIGNED_INT, 0);
		mVAO.unbind();

		gl::glCheckError();
	}
}