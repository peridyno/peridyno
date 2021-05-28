#pragma once

class GLObject
{
protected:
	virtual void create() = 0;
	virtual void release() = 0;

public:
	unsigned int id = 0xFFFFFFFF;
};

// helper functions
unsigned int glCheckError_(const char* file, int line);
#define glCheckError() glCheckError_(__FILE__, __LINE__) 
