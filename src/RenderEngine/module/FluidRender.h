#pragma once

#include "GLVisualModule.h"
#include "GLVertexArray.h"

namespace dyno
{
	class FluidRenderer : public GLVisualModule
	{
		DECLARE_CLASS(FluidRenderer)
	public:
		FluidRenderer();

		void setPointSize(float size);
		float getPointSize() const;

	protected:
		virtual void paintGL() override;
		virtual void updateGL() override;
		virtual bool initializeGL() override;

	private:
		unsigned int	mNumPoints;
		float			mPointSize;

		GLCudaBuffer	mPointBuffer;
		GLVertexArray	mVertexArray;

	};
};