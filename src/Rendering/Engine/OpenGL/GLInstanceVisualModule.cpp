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
		mColorBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mNormalBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		mVAO.bindIndexBuffer(&mIndexBuffer);
		mVAO.bindVertexBuffer(&mVertexBuffer, 0, 3, GL_FLOAT, 0, 0, 0);
		mVAO.bindVertexBuffer(&mColorBuffer, 1, 3, GL_FLOAT, 0, 0, 0);
		mVAO.bindVertexBuffer(&mNormalBuffer, 2, 3, GL_FLOAT, 0, 0, 0);


		mInstanceBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		// bind the translation vector
		mVAO.bindVertexBuffer(&mInstanceBuffer, 3, 3, GL_FLOAT, sizeof(Transform3f), 0, 1);
		// bind the scale vector
		mVAO.bindVertexBuffer(&mInstanceBuffer, 4, 3, GL_FLOAT, sizeof(Transform3f), sizeof(Vec3f), 1);
		// bind the rotation matrix
 		mVAO.bindVertexBuffer(&mInstanceBuffer, 5, 3, GL_FLOAT, sizeof(Transform3f), 2 * sizeof(Vec3f), 1);
		mVAO.bindVertexBuffer(&mInstanceBuffer, 6, 3, GL_FLOAT, sizeof(Transform3f), 3 * sizeof(Vec3f), 1);
		mVAO.bindVertexBuffer(&mInstanceBuffer, 7, 3, GL_FLOAT, sizeof(Transform3f), 4 * sizeof(Vec3f), 1);

		// create shader program
		mShaderProgram = gl::ShaderFactory::createShaderProgram("instance.vert", "surface.frag", "surface.geom");

		return true;
	}

	void GLInstanceVisualModule::updateGL()
	{
		// TODO: check if geometry need to update?
		GLSurfaceVisualModule::updateGL();

		// update instance transforms
		auto& transforms = this->inTransform()->getData();
		mInstanceCount = transforms.size();
		mInstanceBuffer.loadCuda(transforms.begin(), transforms.size() * sizeof(Transform3f));
	}

	void GLInstanceVisualModule::paintGL(GLRenderPass mode)
	{
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

		// color
		auto color = this->varBaseColor()->getData();
		glVertexAttrib3f(1, color[0], color[1], color[2]);

		glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);

		mVAO.bind();
		glDrawElementsInstanced(GL_TRIANGLES, mDrawCount, GL_UNSIGNED_INT, 0, mInstanceCount);
		mVAO.unbind();
	
		gl::glCheckError();
	}
}