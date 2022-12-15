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
		this->varRadius()->setRange(0.001, 0.01);
	}


	std::string GLWireframeVisualModule::caption()
	{
		return "Wireframe Visual Module";
	}

	void GLWireframeVisualModule::setEdgeMode(EEdgeMode mode)
	{
		mEdgeMode = mode;
	}

	bool GLWireframeVisualModule::initializeGL()
	{
		mVAO.create();

		mEdges.create(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
		mPoints.create(GL_ARRAY_BUFFER, GL_STATIC_DRAW);

		mVAO.bindIndexBuffer(&mEdges);
		mVAO.bindVertexBuffer(&mPoints, 0, 3, GL_FLOAT, 0, 0, 0);

		// create shader program
		mShaderProgram = gl::ShaderFactory::createShaderProgram("line.vert", "surface.frag", "line.geom");
		
		return true;
	}


	void GLWireframeVisualModule::updateGL()
	{
		auto edgeSet = this->inEdgeSet()->getDataPtr();

		auto& edges = edgeSet->getEdges();
		auto& vertices = edgeSet->getPoints();

		mNumEdges = edges.size();

		mPoints.loadCuda(vertices.begin(), vertices.size() * sizeof(float) * 3);
		mEdges.loadCuda(edges.begin(), edges.size() * sizeof(unsigned int) * 2);
	}

	void GLWireframeVisualModule::paintGL(GLRenderPass pass)
	{
		if (mNumEdges == 0)
			return;

		mShaderProgram.use();

		unsigned int subroutine = (unsigned int)pass;

		glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);

		if (pass == GLRenderPass::COLOR)
		{
			mShaderProgram.setVec3("uBaseColor", this->varBaseColor()->getData());
			mShaderProgram.setFloat("uMetallic", this->varMetallic()->getData());
			mShaderProgram.setFloat("uRoughness", this->varRoughness()->getData());
			mShaderProgram.setFloat("uAlpha", this->varAlpha()->getData());
		}
		else if (pass == GLRenderPass::SHADOW)
		{
			// cast shadow?
		}
		else
		{
			printf("Unknown render pass!\n");
			return;
		}


		// preserve previous polygon mode
		int mode;
		glGetIntegerv(GL_POLYGON_MODE, &mode);

		if (mEdgeMode == EEdgeMode::LINE)
		{
			// draw as lines
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glLineWidth(this->varLineWidth()->getData());

			mShaderProgram.setInt("uEdgeMode", 0);
		}
		else
		{
			// draw as cylinders
			mShaderProgram.setInt("uEdgeMode", 1);
			mShaderProgram.setFloat("uRadius", this->varRadius()->getData());
		}

		mVAO.bind();		
		glDrawElements(GL_LINES, mNumEdges * 2, GL_UNSIGNED_INT, 0);

		// restore polygon mode
		glPolygonMode(GL_FRONT_AND_BACK, mode);

		gl::glCheckError();
	}


}