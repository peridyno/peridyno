#pragma once
#include "GLObject.h"

class GLBuffer : public GLObject
{
public:
	virtual void create(int target, int usage);
	virtual void release();

	void bind();
	void unbind();

	virtual void allocate(int size);
	virtual void load(void* data, int size, int offset = 0);

	// for uniform buffer
	void bindBufferBase(int idx);

private:
	virtual void create();

protected:
	int target = -1;
	int usage = -1;
	int size = -1;
};

class GLCudaBuffer : public GLBuffer
{
public:	
	void release();

	virtual void allocate(int size);
	//void load(void* data, int size, int offset = 0);

	void  loadCuda(void* devicePtr, int size);
	void* mapCuda();
	void  unmapCuda();

private:
	struct cudaGraphicsResource* resource = 0;
};