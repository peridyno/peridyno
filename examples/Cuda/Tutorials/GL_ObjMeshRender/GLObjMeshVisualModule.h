#pragma once

#include <GLSurfaceVisualModule.h>
#include <gl/Texture.h>

using namespace dyno;

class ObjMeshNode;
class GLObjMeshVisualModule : public GLVisualModule
{
	DECLARE_CLASS(GLObjMeshVisualModule)

public:
	void setColorTexture(const std::string& file);

protected:
	virtual void updateGraphicsContext() override;

protected:
	virtual void paintGL(GLRenderPass mode) override;
	virtual void updateGL() override;
	virtual bool initializeGL() override;
	virtual void destroyGL() override;


public:
	DEF_ARRAY_IN(Vec3f, Position,	DeviceType::GPU, "");
	DEF_ARRAY_IN(Vec3f, Normal,		DeviceType::GPU, "");
	DEF_ARRAY_IN(Vec2f, TexCoord,	DeviceType::GPU, "");
	DEF_ARRAY_IN(Vec3i, Index,		DeviceType::GPU, "");

	DEF_ARRAY2D_IN(Vec4f, TexColor, DeviceType::GPU, "");

private:

private:
	gl::XBuffer mPositions;
	gl::XBuffer mNormals;
	gl::XBuffer mTexCoords;
	gl::XBuffer mInices;
	int			mDrawCount = 0;

	gl::Program		mProgram;
	gl::VertexArray	mVAO;

	CArray2D<Vec4f> mTexData;
	gl::Texture2D	mTexColor;
};