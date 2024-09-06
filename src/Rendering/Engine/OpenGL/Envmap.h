#pragma once

#include "RenderParams.h"
#include "GraphicsObject/Texture.h"
#include "GraphicsObject/Mesh.h"
#include "GraphicsObject/Framebuffer.h"
#include "GraphicsObject/Shader.h"

#include <vector>

namespace dyno
{
class Envmap
{
public:
	Envmap();
	~Envmap();

	void initialize();
	void release();
	void load(const char* path);
	void draw(const RenderParams& rparams);

	void bindIBL();

	void setScale(float scale);

private:
	void update();
	void genLUT();

public:
	const char* path = 0;
		
private:
	bool requireUpdate = false;
		
	struct {
		int width;
		int height;
		int component;
		std::vector<float> data;
		Texture2D tex;
	} image;

	TextureCube irradianceCube;
	TextureCube prefilteredCube;
	Texture2D	brdfLut;

	const int irradianceSize = 16;
	const int prefilteredSize = 128;
	const int brdfLutSize = 128;

	// draw envmap
	Program* prog;

	// cube mesh
	Mesh*	cube = 0;

	Framebuffer fb;

	struct {
		float scale = 1.f;
	} params;
	Buffer	uboParams;
};

};