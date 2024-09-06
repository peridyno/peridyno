#include "GLInstanceVisualModule.h"

#include <glad/glad.h>

namespace dyno {

	IMPLEMENT_CLASS(GLInstanceVisualModule)

	GLInstanceVisualModule::GLInstanceVisualModule()
	{
		this->setName("instance_renderer");
		this->inInstanceColor()->tagOptional(true);
		this->inInstanceTransform()->tagOptional(false);
	}

	GLInstanceVisualModule::~GLInstanceVisualModule()
	{
	}

	std::string GLInstanceVisualModule::caption()
	{
		return "Instance Visual Module";
	}

	void GLInstanceVisualModule::updateImpl()
	{
		GLSurfaceVisualModule::updateImpl();

		// update instance data
		mInstanceTransforms.load(this->inInstanceTransform()->getData());

		// instance colors if available
		if (this->inInstanceColor()->getDataPtr())
			mInstanceColors.load(this->inInstanceColor()->getData());
	}

	bool GLInstanceVisualModule::initializeGL()
	{
		if (GLSurfaceVisualModule::initializeGL())
		{
			// create buffer for instances, we should bind it to VAO later if necessary
			mInstanceTransforms.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
			mInstanceColors.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
			return true;
		}

		return false;
	}

	void GLInstanceVisualModule::releaseGL()
	{
		GLSurfaceVisualModule::releaseGL();
		// instance  data
		mInstanceTransforms.release();
		mInstanceColors.release();
	}

	void GLInstanceVisualModule::updateGL()
	{
		mInstanceCount = mInstanceTransforms.count();

		if (mInstanceCount < 1) return;

		GLSurfaceVisualModule::updateGL();

		mInstanceTransforms.updateGL();

		// bind the translation vector
		mVAO.bindVertexBuffer(&mInstanceTransforms, 3, 3, GL_FLOAT, sizeof(Transform3f), 0, 1);
		// bind the scale vector
		mVAO.bindVertexBuffer(&mInstanceTransforms, 4, 3, GL_FLOAT, sizeof(Transform3f), sizeof(Vec3f), 1);
		// bind the rotation matrix
		mVAO.bindVertexBuffer(&mInstanceTransforms, 5, 3, GL_FLOAT, sizeof(Transform3f), 2 * sizeof(Vec3f), 1);
		mVAO.bindVertexBuffer(&mInstanceTransforms, 6, 3, GL_FLOAT, sizeof(Transform3f), 3 * sizeof(Vec3f), 1);
		mVAO.bindVertexBuffer(&mInstanceTransforms, 7, 3, GL_FLOAT, sizeof(Transform3f), 4 * sizeof(Vec3f), 1);
		
		// bind instance colors
		if (mInstanceColors.count() >= mInstanceCount) {
			mInstanceColors.updateGL();
			mVAO.bindVertexBuffer(&mInstanceColors, 8, 3, GL_FLOAT, sizeof(Vec3f), 0, 1);
		}
		else
		{
			mVAO.bind();
			glDisableVertexAttribArray(8);
			auto color = this->varBaseColor()->getData();
			glVertexAttrib3f(8, color.r, color.g, color.b);
			mVAO.unbind();
		}

	}
}