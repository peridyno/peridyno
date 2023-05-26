#include "GLSurfaceVisualModule.h"
#include "Utility.h"

#include <glad/glad.h>

namespace dyno
{
	IMPLEMENT_CLASS(GLSurfaceVisualModule)

	GLSurfaceVisualModule::GLSurfaceVisualModule()
	{
		this->setName("surface_renderer");
		this->inColor()->tagOptional(true);
	}

	GLSurfaceVisualModule::~GLSurfaceVisualModule()
	{
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
		mColorBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mNormalBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		mVAO.bindIndexBuffer(&mIndexBuffer);

		uint vecSize = sizeof(Vec3f) / sizeof(float);

		mVAO.bindVertexBuffer(&mVertexBuffer, 0, vecSize, GL_FLOAT, 0, 0, 0);
		mVAO.bindVertexBuffer(&mColorBuffer, 1, vecSize, GL_FLOAT, 0, 0, 0);
		mVAO.bindVertexBuffer(&mNormalBuffer, 2, vecSize, GL_FLOAT, 0, 0, 0);

#ifdef CUDA_BACKEND
		// create transform buffer for instances, we should bind it to VAO later if necessary
		mInstanceBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
#endif

		// create shader program
		mShaderProgram = gl::ShaderFactory::createShaderProgram("surface.vert", "surface.frag", "surface.geom");

		return true;
	}

	void GLSurfaceVisualModule::destroyGL()
	{
		if (isGLInitialized)
		{
			mShaderProgram->release();
			delete mShaderProgram;
			mShaderProgram = 0;

			mVAO.release();
			mIndexBuffer.release();
			mVertexBuffer.release();
			mNormalBuffer.release();
			mColorBuffer.release();
			
#ifdef CUDA_BACKEND
			mInstanceBuffer.release();
#endif

			isGLInitialized = false;
		}
	}

	void GLSurfaceVisualModule::updateGL()
	{
		// acquire data update lock
		updateMutex.lock();

		uint vecSize = sizeof(Vec3f) / sizeof(float);

#ifdef CUDA_BACKEND
		mDrawCount = triangles.size() * 3;
		// index
		mIndexBuffer.loadCuda(triangles.begin(), triangles.size() * sizeof(unsigned int) * 3);
		// position
		mVertexBuffer.loadCuda(vertices.begin(), vertices.size() * sizeof(float) * 3);
#endif

#ifdef  VK_BACKEND
		mVertexBuffer.load(vertices.buffer(), vertices.bufferSize());
		mIndexBuffer.load(triangles.buffer(), triangles.bufferSize());

		mDrawCount = triangles.size() * 3;
#endif // DEBUG


		mVAO.bind();
		// vertex or object color
		if (this->varColorMode()->getValue() == EColorMode::CM_Vertex &&
			!colors.isEmpty() && colors.size() == vertices.size())
		{
			mColorBuffer.load(colors.buffer(), colors.bufferSize());
			mVAO.bindVertexBuffer(&mColorBuffer, 1, vecSize, GL_FLOAT, 0, 0, 0);
		}
		else
		{
			glDisableVertexAttribArray(1);
		}

		gl::glCheckError();

		// normal
		if(this->varUseVertexNormal()->getValue())
		{
#ifdef CUDA_BACKEND
			mNormalBuffer.loadCuda(normals.begin(), normals.size() * sizeof(float) * 3);
#endif

#ifdef  VK_BACKEND
			mNormalBuffer.load(normals.buffer(), normals.bufferSize());
#endif // DEBUG

			mVAO.bindVertexBuffer(&mNormalBuffer, 2, vecSize, GL_FLOAT, 0, 0, 0);
		}
		else
		{
			glDisableVertexAttribArray(2);
		}

		mInstanceCount = 0;

		// unbind instance buffer
		glDisableVertexAttribArray(3);
		glDisableVertexAttribArray(4);
		glDisableVertexAttribArray(5);
		glDisableVertexAttribArray(6);
		glDisableVertexAttribArray(7);

		mVAO.unbind();

		// release data update lock
		updateMutex.unlock();
	}

	void GLSurfaceVisualModule::updateGraphicsContext()
	{
		updateMutex.lock();

		// update data

		auto triSet = this->inTriangleSet()->getDataPtr();

		triangles.assign(triSet->getTriangles());
		vertices.assign(triSet->getPoints());

		if (this->varColorMode()->getValue() == EColorMode::CM_Vertex &&
			!this->inColor()->isEmpty() &&
			this->inColor()->getDataPtr()->size() == vertices.size())
		{
			colors.assign(this->inColor()->getData());
		}

		if (this->varUseVertexNormal()->getData())
		{			
			//TODO: optimize the performance
			triSet->update();
#ifdef  CUDA_BACKEND
			normals.assign(triSet->getVertexNormals());
#endif
		}

		GLVisualModule::updateGraphicsContext();
		updateMutex.unlock();
	}

	void GLSurfaceVisualModule::paintGL(GLRenderPass mode)
	{
		if (mDrawCount == 0)
			return;

		unsigned int subroutine;
		if (mode == GLRenderPass::COLOR) {
			subroutine = 0;
		}
		else if (mode == GLRenderPass::SHADOW) {
			subroutine = 1;
		}
		else if (mode == GLRenderPass::TRANSPARENCY) {
			subroutine = 2;
		}
		else {
			printf("GLSurfaceVisualModule: Unknown render mode!\n");
			return;
		}

		mShaderProgram->use();

		// setup uniforms
		mShaderProgram->setFloat("uMetallic", this->varMetallic()->getData());
		mShaderProgram->setFloat("uRoughness", this->varRoughness()->getData());
		mShaderProgram->setFloat("uAlpha", this->varAlpha()->getData());
		mShaderProgram->setInt("uVertexNormal", this->varUseVertexNormal()->getData());

		// instanced rendering?
		mShaderProgram->setInt("uInstanced", mInstanceCount > 0);

		// color
		auto color = this->varBaseColor()->getData();
		glVertexAttrib3f(1, color.r, color.g, color.b);

		glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);

		mVAO.bind();

		gl::glCheckError();
		if(mInstanceCount > 0)
			glDrawElementsInstanced(GL_TRIANGLES, mDrawCount, GL_UNSIGNED_INT, 0, mInstanceCount);
		else
			glDrawElements(GL_TRIANGLES, mDrawCount, GL_UNSIGNED_INT, 0);

		gl::glCheckError();
		mVAO.unbind();

		gl::glCheckError();
	}
}