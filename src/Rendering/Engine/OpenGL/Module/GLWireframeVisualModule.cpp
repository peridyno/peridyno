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

	GLWireframeVisualModule::~GLWireframeVisualModule()
	{
// 		edges.clear();
// 		vertices.clear();

// 		mVertexBuffer.release();
// 		mIndexBuffer.release();
	}


	std::string GLWireframeVisualModule::caption()
	{
		return "Wireframe Visual Module";
	}

	bool GLWireframeVisualModule::initializeGL()
	{
		mVAO.create();

		mIndexBuffer.create(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
		mVertexBuffer.create(GL_ARRAY_BUFFER, GL_STATIC_DRAW);

		uint vecSize = sizeof(Vec3f) / sizeof(float);

		mVAO.bindIndexBuffer(&mIndexBuffer);
		mVAO.bindVertexBuffer(&mVertexBuffer, 0, vecSize, GL_FLOAT, 0, 0, 0);

		// create shader program
		mShaderProgram = gl::ShaderFactory::createShaderProgram("line.vert", "surface.frag", "line.geom");
		
		return true;
	}

	void GLWireframeVisualModule::destroyGL()
	{
		if (isGLInitialized)
		{
			mShaderProgram->release();
			delete mShaderProgram;

			mVAO.release();
			mVertexBuffer.release();
			mIndexBuffer.release();

			isGLInitialized = false;
		}
	}


	void GLWireframeVisualModule::updateGL()
	{
		updateMutex.lock();

		mNumEdges = edges.size();

#ifdef CUDA_BACKEND
		mVertexBuffer.loadCuda(vertices.begin(), vertices.size() * sizeof(float) * 3);
		mIndexBuffer.loadCuda(edges.begin(), edges.size() * sizeof(unsigned int) * 2);
#endif

#ifdef  VK_BACKEND
		mVertexBuffer.load(vertices.buffer(), vertices.bufferSize());
		mIndexBuffer.load(edges.buffer(), edges.bufferSize());
#endif // DEBUG

		updateMutex.unlock();
	}

	void GLWireframeVisualModule::updateGraphicsContext()
	{
		updateMutex.lock();

		// copy data
		auto edgeSet = this->inEdgeSet()->getDataPtr();
		edges.assign(edgeSet->getEdges());
		vertices.assign(edgeSet->getPoints());

		GLVisualModule::updateGraphicsContext();
		updateMutex.unlock();
	}

	void GLWireframeVisualModule::paintGL(GLRenderPass pass)
	{
		if (mNumEdges == 0)
			return;

		mShaderProgram->use();

		unsigned int subroutine = (unsigned int)pass;

		glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);

		if (pass == GLRenderPass::COLOR)
		{
			Color c = this->varBaseColor()->getData();
			mShaderProgram->setVec3("uBaseColor", Vec3f(c.r, c.g, c.b));
			mShaderProgram->setFloat("uMetallic", this->varMetallic()->getData());
			mShaderProgram->setFloat("uRoughness", this->varRoughness()->getData());
			mShaderProgram->setFloat("uAlpha", this->varAlpha()->getData());
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

		if (this->varRenderMode()->getDataPtr()->currentKey() == EEdgeMode::LINE)
		{
			// draw as lines
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glLineWidth(this->varLineWidth()->getData());

			mShaderProgram->setInt("uEdgeMode", 0);
		}
		else
		{
			// draw as cylinders
			mShaderProgram->setInt("uEdgeMode", 1);
			mShaderProgram->setFloat("uRadius", this->varRadius()->getData());
		}

		mVAO.bind();		
		glDrawElements(GL_LINES, mNumEdges * 2, GL_UNSIGNED_INT, 0);

		// restore polygon mode
		glPolygonMode(GL_FRONT_AND_BACK, mode);

		gl::glCheckError();
	}
}