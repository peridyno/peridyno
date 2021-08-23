#include "SurfaceRender.h"

// opengl
#include <glad/glad.h>
#include "RenderEngine.h"
#include "Utility.h"

namespace dyno
{
	IMPLEMENT_CLASS(SurfaceRenderer)

	SurfaceRenderer::SurfaceRenderer()
	{
		this->setName("surface_renderer");
	}

	bool SurfaceRenderer::initializeGL()
	{
		// create vertex buffer and vertex array object
		mVAO.create();
		mIndexBuffer.create(GL_ELEMENT_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mVertexBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		mVAO.bindIndexBuffer(&mIndexBuffer);
		mVAO.bindVertexBuffer(&mVertexBuffer, 0, 3, GL_FLOAT, 0, 0, 0);

		// create shader program
		mShaderProgram = gl::CreateShaderProgram("surface.vert", "surface.frag", "surface.geom");

		return true;
	}

	void SurfaceRenderer::updateStarted()
	{
		std::cout << "Update for SurfaceRenderer started!" << std::endl;
	}

	void SurfaceRenderer::updateEnded()
	{
		std::cout << "Update for SurfaceRenderer ended!" << std::endl;
	}

	void SurfaceRenderer::updateGL()
	{
		auto triSet = this->inTriangleSet()->getDataPtr();

		auto triangles = triSet->getTriangles();
		auto vertices = triSet->getPoints();

		mDrawCount = triangles->size() * 3;

		mVertexBuffer.loadCuda(vertices.begin(), vertices.size() * sizeof(float) * 3);
		mIndexBuffer.loadCuda(triangles->begin(), triangles->size() * sizeof(unsigned int) * 3);
	}

	void SurfaceRenderer::paintGL(RenderMode mode)
	{
		mShaderProgram.use();

		unsigned int subroutine;
		if (mode == RenderMode::COLOR)
		{
			mShaderProgram.setVec3("uBaseColor", mBaseColor);
			mShaderProgram.setFloat("uMetallic", mMetallic);
			mShaderProgram.setFloat("uRoughness", mRoughness);
			mShaderProgram.setFloat("uAlpha", mAlpha);	// not implemented!

			subroutine = 0;
			glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
		}
		else if (mode == RenderMode::DEPTH)
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