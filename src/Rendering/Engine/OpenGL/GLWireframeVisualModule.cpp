#include "GLWireframeVisualModule.h"

// opengl
#include <glad/glad.h>
#include "GLRenderEngine.h"
#include "Utility.h"

namespace dyno
{
	IMPLEMENT_CLASS(GLWireframeVisualModule)

	GLWireframeVisualModule::GLWireframeVisualModule()
	{
		this->setName("wireframe_renderer");
	}

	bool GLWireframeVisualModule::initializeGL()
	{
		// create vertex buffer and vertex array object
		mVAO.create();
		mIndexBuffer.create(GL_ELEMENT_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mVertexBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		mVAO.bindIndexBuffer(&mIndexBuffer);
		mVAO.bindVertexBuffer(&mVertexBuffer, 0, 3, GL_FLOAT, 0, 0, 0);

		// create shader program
		mShaderProgram = gl::CreateShaderProgram("line.vert", "line.frag", "line.geom");

		return true;
	}

	void GLWireframeVisualModule::updateGL()
	{
		auto edgeSet = this->inEdgeSet()->getDataPtr();

		auto& edges = edgeSet->getEdges();
		auto& vertices = edgeSet->getPoints();

		mDrawCount = edges.size() * 2;

		mVertexBuffer.loadCuda(vertices.begin(), vertices.size() * sizeof(float) * 3);
		mIndexBuffer.loadCuda(edges.begin(), edges.size() * sizeof(unsigned int) * 2);
	}

	void GLWireframeVisualModule::paintGL(RenderMode mode)
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
		glDrawElements(GL_LINES, mDrawCount, GL_UNSIGNED_INT, 0);
		mVAO.unbind();

		gl::glCheckError();
	}
}