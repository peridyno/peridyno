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

	void GLInstanceVisualModule::updateGraphicsContext()
	{
		GLSurfaceVisualModule::updateGraphicsContext();

		updateMutex.lock();

		// update instance data
		instanceTransforms.assign(this->inInstanceTransform()->getData());
		// instance colors if available
		if(this->inInstanceColor()->getDataPtr())
			instanceColors.assign(this->inInstanceColor()->getData());

		updateMutex.unlock();
	}

	void GLInstanceVisualModule::updateGL()
	{
		GLSurfaceVisualModule::updateGL();

		updateMutex.lock();

		mInstanceCount = instanceTransforms.size();
		mInstanceBuffer.loadCuda(instanceTransforms.begin(), instanceTransforms.size() * sizeof(Transform3f));
					
		if (instanceColors.size() == mInstanceCount) {
			mColorBuffer.loadCuda(instanceColors.begin(), instanceColors.size() * sizeof(Vec3f));
			mVAO.bindVertexBuffer(&mColorBuffer, 1, 3, GL_FLOAT, sizeof(Vec3f), 0, 1);
		}

		// bind the translation vector
		mVAO.bindVertexBuffer(&mInstanceBuffer, 3, 3, GL_FLOAT, sizeof(Transform3f), 0, 1);
		// bind the scale vector
		mVAO.bindVertexBuffer(&mInstanceBuffer, 4, 3, GL_FLOAT, sizeof(Transform3f), sizeof(Vec3f), 1);
		// bind the rotation matrix
		mVAO.bindVertexBuffer(&mInstanceBuffer, 5, 3, GL_FLOAT, sizeof(Transform3f), 2 * sizeof(Vec3f), 1);
		mVAO.bindVertexBuffer(&mInstanceBuffer, 6, 3, GL_FLOAT, sizeof(Transform3f), 3 * sizeof(Vec3f), 1);
		mVAO.bindVertexBuffer(&mInstanceBuffer, 7, 3, GL_FLOAT, sizeof(Transform3f), 4 * sizeof(Vec3f), 1);


		updateMutex.unlock();
	}
}