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

	void GLInstanceVisualModule::updateGL()
	{
		GLSurfaceVisualModule::updateGL();

		// update instance transforms
		auto transforms = this->inInstanceTransform()->getDataPtr();

		if (transforms) {
			mInstanceCount = transforms->size();
			mInstanceBuffer.loadCuda(transforms->begin(), transforms->size() * sizeof(Transform3f));

			// instance colors if available
			auto colors = this->inInstanceColor()->getDataPtr();
			if (colors && colors->size() == mInstanceCount) {
				mColorBuffer.loadCuda(colors->begin(), colors->size() * sizeof(Vec3f));
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
		}
	}
}