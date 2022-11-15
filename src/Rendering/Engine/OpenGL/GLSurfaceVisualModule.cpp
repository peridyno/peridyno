#include "GLSurfaceVisualModule.h"
#include "Utility.h"

#include "gl/VulkanBuffer.h"

#include <glad/glad.h>

namespace dyno
{
	IMPLEMENT_CLASS(GLSurfaceVisualModule)

	GLSurfaceVisualModule::GLSurfaceVisualModule()
	{
		this->setName("surface_renderer");

		// buffers
		mIndexBuffer = new gl::VulkanBuffer;
		mVertexBuffer = new gl::VulkanBuffer;
		mColorBuffer = new gl::VulkanBuffer;
		mNormalBuffer = new gl::VulkanBuffer;
		mInstanceBuffer = new gl::VulkanBuffer;
	}

	GLSurfaceVisualModule::~GLSurfaceVisualModule()
	{
		// TODO: release resource within OpenGL context
		delete mIndexBuffer;
		delete mVertexBuffer;
		delete mColorBuffer;
		delete mNormalBuffer;
		delete mInstanceBuffer;
	}

	std::string GLSurfaceVisualModule::caption()
	{
		return "Surface Visual Module";
	}

	bool GLSurfaceVisualModule::initializeGL()
	{
		// create vertex buffer and vertex array object
		mVAO.create();
		mIndexBuffer->create(GL_ELEMENT_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mVertexBuffer->create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mColorBuffer->create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mNormalBuffer->create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		mVAO.bindIndexBuffer(mIndexBuffer);
		mVAO.bindVertexBuffer(mVertexBuffer, 0, 4, GL_FLOAT, 0, 0, 0);
		mVAO.bindVertexBuffer(mColorBuffer, 1, 3, GL_FLOAT, 0, 0, 0);
		mVAO.bindVertexBuffer(mNormalBuffer, 2, 3, GL_FLOAT, 0, 0, 0);

		// create transform buffer for instances, we should bind it to VAO later if necessary
		mInstanceBuffer->create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		// create shader program
		mShaderProgram = gl::ShaderFactory::createShaderProgram("surface.vert", "surface.frag", "surface.geom");

		return true;
	}

	void GLSurfaceVisualModule::updateGL()
	{
		auto triSet = this->inTriangleSet()->getDataPtr();
		auto& indices = triSet->mIndex;
		auto& vertices = triSet->mPoints;

		mVertexBuffer->load(vertices.bufferHandle(), vertices.bufferSize());
		mIndexBuffer->load(indices.bufferHandle(), indices.bufferSize());

		mDrawCount = indices.size() * 3;

		mVAO.bind();

		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);

		gl::glCheckError();

		// unbind instance buffer
		mInstanceCount = 0;
		glDisableVertexAttribArray(3);
		glDisableVertexAttribArray(4);
		glDisableVertexAttribArray(5);
		glDisableVertexAttribArray(6);
		glDisableVertexAttribArray(7);

		mVAO.unbind();
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

		mShaderProgram.use();

		// setup uniforms
		mShaderProgram.setFloat("uMetallic", this->varMetallic()->getData());
		mShaderProgram.setFloat("uRoughness", this->varRoughness()->getData());
		mShaderProgram.setFloat("uAlpha", this->varAlpha()->getData());
		mShaderProgram.setInt("uVertexNormal", this->varUseVertexNormal()->getData());

		// instanced rendering?
		mShaderProgram.setInt("uInstanced", mInstanceCount > 0);

		// color
		auto color = this->varBaseColor()->getData();
		glVertexAttrib3f(1, color[0], color[1], color[2]);

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