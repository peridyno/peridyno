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


	std::string GLWireframeVisualModule::caption()
	{
		return "Wireframe Visual Module";
	}

	bool GLWireframeVisualModule::initializeGL()
	{
		// create vertex buffer and vertex array object
		mVAO.create();
		mIndexBuffer.create(GL_ELEMENT_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mVertexBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		mVAO.bindIndexBuffer(&mIndexBuffer);
		mVAO.bindVertexBuffer(&mVertexBuffer, 0, 4, GL_FLOAT, 0, 0, 0);

		// create shader program
		mShaderProgram = gl::ShaderFactory::createShaderProgram("line.vert", "surface.frag", "line.geom");

		return true;
	}

	void GLWireframeVisualModule::updateGL()
	{
		auto edgeSet = this->inEdgeSet()->getDataPtr();

		auto& edges = edgeSet->mEdgeIndex;
		auto& vertices = edgeSet->mPoints;

		mDrawCount = edges.size() * 2;

		mVertexBuffer.load(vertices.buffer(), vertices.bufferSize());
		mIndexBuffer.load(edges.buffer(), edges.bufferSize());
	}

	void GLWireframeVisualModule::paintGL(GLRenderPass pass)
	{
		if (mDrawCount == 0)
			return;

		mShaderProgram->use();

		unsigned int subroutine = (unsigned int)pass;

		glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);

		if (pass == GLRenderPass::COLOR)
		{
			mShaderProgram->setVec3("uBaseColor", this->varBaseColor()->getData());
			mShaderProgram->setFloat("uMetallic", this->varMetallic()->getData());
			mShaderProgram->setFloat("uRoughness", this->varRoughness()->getData());
			mShaderProgram->setFloat("uAlpha", this->varAlpha()->getData());
		}
		else if (pass == GLRenderPass::SHADOW)
		{
			// lines should cast shadow?
		}
		else
		{
			printf("Unknown render pass!\n");
			return;
		}

		mVAO.bind();
		glDrawElements(GL_LINES, mDrawCount, GL_UNSIGNED_INT, 0);
		mVAO.unbind();

		gl::glCheckError();
	}
}