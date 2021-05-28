#pragma once

#include "GLVisualModule.h"
#include "GLVertexArray.h"

namespace dyno
{
	class SurfaceRenderer : public GLVisualModule
	{
		DECLARE_CLASS(SurfaceRenderer)
	public:
		SurfaceRenderer();

	protected:
		virtual void paintGL() override;
		virtual void updateGL() override;
		virtual bool initializeGL() override;

	private:

		GLCudaBuffer	mVertexBuffer;
		//GLCudaBuffer	mNormalBuffer;
		GLCudaBuffer 	mIndexBuffer;
		GLVertexArray	mVAO;

		unsigned int	mDrawCount = 0;
	};
};