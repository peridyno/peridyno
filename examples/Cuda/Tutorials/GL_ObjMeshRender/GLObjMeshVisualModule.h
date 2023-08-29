#pragma once

#include <GLSurfaceVisualModule.h>
#include <gl/GPUTexture.h>

using namespace dyno;

class ObjMeshNode;
class GLObjMeshVisualModule : public GLVisualModule
{
	DECLARE_CLASS(GLObjMeshVisualModule)

public:
	void setColorTexture(const std::string& file);

protected:
	virtual void updateImpl() override;

protected:
	virtual void paintGL(const RenderParams& rparams) override;
	virtual void updateGL() override;
	virtual bool initializeGL() override;
	virtual void releaseGL() override;


public:
	DEF_ARRAY_IN(Vec3f, Position,	DeviceType::GPU, "");
	DEF_ARRAY_IN(Vec3f, Normal,		DeviceType::GPU, "");
	DEF_ARRAY_IN(Vec2f, TexCoord,	DeviceType::GPU, "");
	DEF_ARRAY_IN(Vec3i, Index,		DeviceType::GPU, "");

	DEF_ARRAY2D_IN(Vec4f, TexColor, DeviceType::GPU, "");


private:
	gl::XBuffer<dyno::Vec3f> mPositions;
	gl::XBuffer<dyno::Vec3f> mNormals;
	gl::XBuffer<dyno::Vec2f> mTexCoords;
	gl::XBuffer<dyno::Vec3i> mInices;
	int			mDrawCount = 0;

	gl::Program		mProgram;
	gl::VertexArray	mVAO;

	gl::XTexture2D<dyno::Vec4f> mTexColor;

	gl::Buffer		mUniformBlock;
};