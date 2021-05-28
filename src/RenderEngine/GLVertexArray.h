#pragma once

#include "GLBuffer.h"
#include <glm/vec3.hpp>

class GLVertexArray : public GLObject
{
public:
	virtual void create();
	virtual void release();

	virtual void bind();
	virtual void unbind();

	virtual void bindIndexBuffer(GLBuffer* buffer);
	virtual void bindVertexBuffer(GLBuffer* buffer, 
		int index, int size, int type, int stride, int offset, int divisor);
};

class GLMesh : public GLVertexArray
{
public:
	virtual void create();
	virtual void release();

	virtual void draw(int instance = 0);

public:
	static GLMesh Sphere(float radius = 1.f, int sectors = 16, int stacks = 8);
	static GLMesh AABB(glm::vec3 p0, glm::vec3 p1);
	static GLMesh ScreenQuad();
	static GLMesh Plane(float scale);

private:
	GLBuffer	mVertexBuffer;
	GLBuffer	mIndexBuffer;
	int			mDrawCount;
};