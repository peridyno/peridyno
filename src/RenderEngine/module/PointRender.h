#pragma once

#include "GLVisualModule.h"
#include "GLVertexArray.h"

namespace dyno
{
	class PointRenderer : public GLVisualModule
	{
		DECLARE_CLASS(PointRenderer)
	public:
		PointRenderer();

		void setPointSize(float size);
		float getPointSize() const;

		bool isTransparent() const
		{
			return false;
		}

	protected:
		virtual void paintGL() override;
		virtual void updateGL() override;
		virtual bool initializeGL() override;

	private:
		unsigned int	mNumPoints;

		GLCudaBuffer	mPosition;
		GLCudaBuffer	mVelocity;
		GLCudaBuffer	mForce;

		float			mPointSize;
		GLVertexArray	mVertexArray;

	};
};
