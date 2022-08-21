#include "GLInstanceVisualModule.h"

// opengl
#include <glad/glad.h>
#include "GLRenderEngine.h"
#include "Utility.h"

namespace dyno
{
	IMPLEMENT_CLASS(GLInstanceVisualModule)

		GLInstanceVisualModule::GLInstanceVisualModule()
	{
		this->setName("instance_renderer");
	}

	std::string GLInstanceVisualModule::caption()
	{
		return "Instance Visual Module";
	}

	bool GLInstanceVisualModule::initializeGL()
	{
		// create vertex buffer and vertex array object
		mVAO.create();
		mIndexBuffer.create(GL_ELEMENT_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mVertexBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mInstanceBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		mVAO.bindIndexBuffer(&mIndexBuffer);
		mVAO.bindVertexBuffer(&mVertexBuffer, 0, 3, GL_FLOAT, 0, 0, 0);

		// bind the translation vector
		mVAO.bindVertexBuffer(&mInstanceBuffer, 1, 3, GL_FLOAT, sizeof(Transform3f), 0, 1);

		// bind the scale vector
		mVAO.bindVertexBuffer(&mInstanceBuffer, 2, 3, GL_FLOAT, sizeof(Transform3f), sizeof(Vec3f), 1);

		// bind the rotation matrix
 		mVAO.bindVertexBuffer(&mInstanceBuffer, 3, 3, GL_FLOAT, sizeof(Transform3f), 2 * sizeof(Vec3f), 1);
		mVAO.bindVertexBuffer(&mInstanceBuffer, 4, 3, GL_FLOAT, sizeof(Transform3f), 3 * sizeof(Vec3f), 1);
		mVAO.bindVertexBuffer(&mInstanceBuffer, 5, 3, GL_FLOAT, sizeof(Transform3f), 4 * sizeof(Vec3f), 1);

		// create shader program
		mShaderProgram = gl::CreateShaderProgram("instance.vert", "instance.frag", "instance.geom");

		// initialize data
		auto triSet = this->inTriangleSet()->getDataPtr();

		auto& triangles = triSet->getTriangles();
		auto& vertices = triSet->getPoints();

 		auto& transforms = this->inTransform()->getData();
 		mInstanceCount = transforms.size();

		mVertexCount = vertices.size();
		mIndexCount = triangles.size() * 3;
		//mVertexCount = triangles.size() * 3;

		mVertexBuffer.loadCuda(vertices.begin(), vertices.size() * sizeof(float) * 3);
		mIndexBuffer.loadCuda(triangles.begin(), triangles.size() * sizeof(unsigned int) * 3);

		return true;
	}

	void GLInstanceVisualModule::updateGL()
	{
		auto& transforms = this->inTransform()->getData();
		mInstanceCount = transforms.size();

		mInstanceBuffer.loadCuda(transforms.begin(), transforms.size() * sizeof(Transform3f));
	}

	void GLInstanceVisualModule::paintGL(RenderPass pass)
	{
		mShaderProgram.use();

		unsigned int subroutine;
		if (pass == RenderPass::COLOR)
		{
			mShaderProgram.setVec3("uBaseColor", this->varBaseColor()->getData());
			mShaderProgram.setFloat("uMetallic", mMetallic);
			mShaderProgram.setFloat("uRoughness", mRoughness);
			mShaderProgram.setFloat("uAlpha", mAlpha);	// not implemented!

			subroutine = 0;
			glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
		}
		else if (pass == RenderPass::SHADOW)
		{
			subroutine = 1;
			glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
		}
		else
		{
			printf("Unknown render pass!\n");
			return;
		}

		mVAO.bind();
		glDrawElementsInstanced(GL_TRIANGLES, mIndexCount, GL_UNSIGNED_INT, 0, mInstanceCount);
		mVAO.unbind();

		gl::glCheckError();
	}
}