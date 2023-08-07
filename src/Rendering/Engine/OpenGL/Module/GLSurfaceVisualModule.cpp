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
// 		mIndexBuffer.release();
// 		mVertexBuffer.release();
// 		mNormalBuffer.release();
// 		mColorBuffer.release();
// 
// 		triangles.clear();
// 		vertices.clear();
// 		normals.clear();
// 		colors.clear();
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

		// create shader uniform buffer
		mUniformBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);

		return true;
	}

	void GLSurfaceVisualModule::releaseGL()
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
		mUniformBlock.release();
	}

	void GLSurfaceVisualModule::updateGL()
	{
		uint vecSize = sizeof(Vec3f) / sizeof(float);

		mVertexBuffer.mapGL();
		mIndexBuffer.mapGL();
		// need to rebind
		mVAO.bindIndexBuffer(&mIndexBuffer);
		mVAO.bindVertexBuffer(&mVertexBuffer, 0, vecSize, GL_FLOAT, 0, 0, 0);

		mVAO.bind();
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);
		// unbind instance buffer
		glDisableVertexAttribArray(3);
		glDisableVertexAttribArray(4);
		glDisableVertexAttribArray(5);
		glDisableVertexAttribArray(6);
		glDisableVertexAttribArray(7);
		mVAO.unbind();

		// vertex color
		if (this->varColorMode()->getValue() == EColorMode::CM_Vertex) {
			mColorBuffer.mapGL();
			mVAO.bindVertexBuffer(&mColorBuffer, 1, vecSize, GL_FLOAT, 0, 0, 0);
		}
		// vertex normal
		if(this->varUseVertexNormal()->getValue()) {
			mNormalBuffer.mapGL();
			mVAO.bindVertexBuffer(&mNormalBuffer, 2, vecSize, GL_FLOAT, 0, 0, 0);
		}
		gl::glCheckError();
	}

	void GLSurfaceVisualModule::updateImpl()
	{
		// update data

		auto triSet = this->inTriangleSet()->getDataPtr();

#ifdef  CUDA_BACKEND
		auto indices = triSet->getTriangles();
		mDrawCount = indices.size() * 3;
#endif // CUDA_BACKEND

#ifdef VK_BACKEND
		auto indices = triSet->getVulkanIndex();
		mDrawCount = indices.size();
#endif // VK_BACKEND

		if (mDrawCount > 0)
		{
			mIndexBuffer.load(indices);

			auto vertices = triSet->getPoints();
			mVertexBuffer.load(vertices);

			if (this->varColorMode()->getValue() == EColorMode::CM_Vertex &&
				!this->inColor()->isEmpty() &&
				this->inColor()->getDataPtr()->size() == vertices.size())
			{
				auto colors = this->inColor()->getData();
				mColorBuffer.load(colors);
			}

			if (this->varUseVertexNormal()->getData())
			{
				//TODO: optimize the performance
#ifdef CUDA_BACKEND
				triSet->update();
				auto normals = triSet->getVertexNormals();
				mNormalBuffer.load(normals);
#endif
			}

		}
	}

	void GLSurfaceVisualModule::paintGL(const RenderParams& rparams)
	{
		if (mDrawCount == 0)
			return;

		unsigned int subroutine;
		if (rparams.mode == GLRenderMode::COLOR) {
			subroutine = 0;
		}
		else if (rparams.mode == GLRenderMode::SHADOW) {
			subroutine = 1;
		}
		else if (rparams.mode == GLRenderMode::TRANSPARENCY) {
			subroutine = 2;
		}
		else {
			printf("GLSurfaceVisualModule: Unknown render mode!\n");
			return;
		}

		// setup uniform buffer
		mUniformBlock.load((void*)&rparams, sizeof(RenderParams));
		mUniformBlock.bindBufferBase(0);

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