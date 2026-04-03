#pragma once
#include "Topology/PointSet.h"
#include "GLVisualModule.h"
#include "GraphicsObject/GPUBuffer.h"
#include "GraphicsObject/VertexArray.h"
#include "GraphicsObject/Shader.h"

#include <stb/stb_image.h>
namespace dyno
{
	class GLDigitNumVisualModule : public GLVisualModule
	{
		DECLARE_CLASS(GLDigitNumVisualModule)
	public:
		GLDigitNumVisualModule();

		DEF_VAR(Real, DigitScale, 0.005f, "Scale of rendered digit numbers");
		DEF_VAR(Vec2f, DigitOffset, Vec2f(1.0f), "Offset: Num-Vertex");

		DEF_INSTANCE_IN(PointSet<DataType3f>, PointSet, "Point set data");

	protected:
		virtual void updateImpl() override;
		virtual void paintGL(const RenderParams& rparams) override;
		virtual void updateGL() override;
		virtual bool initializeGL() override;
		virtual void releaseGL() override;

	private:
		bool loadDigitTextures();

		VertexArray mVertexArray;

		XBuffer<Vec3f> mPosition;

		unsigned int mNumPoints;
		Program* mShaderProgram = nullptr;

		unsigned int mDigitTexture; // Texture array for digits
		Buffer mUniformBlock;
	};
};