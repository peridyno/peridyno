#include "Instance.h"

#include "glad/glad.h"

namespace dyno
{
	ShapeInstance::ShapeInstance()
	{
	}

	ShapeInstance::~ShapeInstance()
	{
	}

	void ShapeInstance::create()
	{
		if (!mInitialized)
		{
			gltransform.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
			mInitialized = true;
		}
	}

	void ShapeInstance::release()
	{
		gltransform.release();
	}

	void ShapeInstance::update()
	{
		gltransform.load(transform);
		instanceCount = transform.size();
	}

	void ShapeInstance::updateGL()
	{
		gltransform.updateGL();
	}

}