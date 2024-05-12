#include "Envmap.h"
#include "GraphicsObject/Shader.h"

#include <glad/glad.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <glm/gtc/matrix_transform.hpp>

#include "envmap.vert.h"
#include "envmap.frag.h"
#include "screen.vert.h"
#include "brdf.frag.h"

using namespace dyno;

Envmap::Envmap()
{
	image.tex.format = GL_RGB;
	image.tex.internalFormat = GL_RGB32F;
}

Envmap::~Envmap()
{

}

void Envmap::initialize()
{
	image.tex.format = GL_RGB;
	image.tex.internalFormat = GL_RGB32F;
	image.tex.type = GL_FLOAT;
	image.tex.maxFilter = GL_LINEAR;
	image.tex.minFilter = GL_LINEAR_MIPMAP_LINEAR; // produce seam artifacts...
	image.tex.create();
	// wrapping
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	// initialize cubemaps
	irradianceCube.create();
	irradianceCube.bind();
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	for (unsigned int i = 0; i < 6; ++i) {
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, irradianceSize, irradianceSize, 0,
			GL_RGB, GL_FLOAT, nullptr);
	}
	prefilteredCube.create();
	prefilteredCube.bind();
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

	for (unsigned int i = 0; i < 6; ++i) {
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, prefilteredSize, prefilteredSize, 0,
			GL_RGB, GL_FLOAT, nullptr);
	}
	glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
	glCheckError();


	brdfLut.format = GL_RG;
	brdfLut.internalFormat = GL_RG16F;
	brdfLut.type = GL_FLOAT;
	brdfLut.create();
	brdfLut.bind();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	brdfLut.resize(brdfLutSize, brdfLutSize);

	glCheckError();

	cube = Mesh::Cube();

	// create programs
	prog = Program::createProgramSPIRV(
		ENVMAP_VERT, sizeof(ENVMAP_VERT),
		ENVMAP_FRAG, sizeof(ENVMAP_FRAG));

	fb.create();

	genLUT();
}

void Envmap::release()
{
	fb.release();

	image.tex.release();

	irradianceCube.release();
	prefilteredCube.release();
	brdfLut.release();

	cube->release();
	delete cube;

	prog->release();
	delete prog;

}

void Envmap::load(const char* path)
{
	this->path = path;
	this->requireUpdate = true;

	if (this->path) {
		float* data = stbi_loadf(path, &image.width, &image.height, &image.component, 3);

		if (data) {
			// copy
			int size = image.width * image.height * image.component;
			image.data.resize(size);
			memcpy(image.data.data(), data, size * sizeof(float));
			stbi_image_free(data);
			return;
		}
	}

	// failed to load envmap...
	this->path = 0;
}

void Envmap::draw(const RenderParams& rparams)
{
	if (this->path) {
		if (requireUpdate) {
			// update envmap
			this->update();
			this->requireUpdate = false;
		}

		glm::mat4 mvp =
			rparams.transforms.proj *
			glm::mat4(glm::mat3(rparams.transforms.view));

		prog->use();
		prog->setInt("mode", 0);

		// set transform
		GLuint mvpLoc = glGetUniformLocation(prog->id, "mvp");
		glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, &mvp[0][0]);

		// bind texture
		image.tex.bind(GL_TEXTURE1);
		cube->draw();

		glClear(GL_DEPTH_BUFFER_BIT);
	}
}

void dyno::Envmap::bindIBL()
{
	//glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	irradianceCube.bind(GL_TEXTURE7);
	prefilteredCube.bind(GL_TEXTURE8);
	brdfLut.bind(GL_TEXTURE9);
	glCheckError();
}

void Envmap::update()
{
	image.tex.load(image.width, image.height, image.data.data());
	image.tex.genMipmap();
	glCheckError();

	// preserve current framebuffer
	GLint fbo;
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);
	GLint vp[4];
	glGetIntegerv(GL_VIEWPORT, vp);

	const glm::mat4 proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
	//glm::mat4 proj = glm::perspective(90.0f, 1.0f, 0.1f, 10.0f);
	const glm::mat4 views[] =
	{
		proj * glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
		proj * glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
		proj * glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
		proj * glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
		proj * glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
		proj * glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
	};

	fb.bind();
	//const GLuint attachments = GL_COLOR_ATTACHMENT0;
	//fb.drawBuffers(1, &attachments);
	image.tex.bind(GL_TEXTURE1);
	const GLuint mvpLoc = glGetUniformLocation(prog->id, "mvp");

	// irradiance
	prog->use();
	prog->setInt("mode", 1);
	glViewport(0, 0, irradianceSize, irradianceSize);
	for (int i = 0; i < 6; i++) {
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, irradianceCube.id, 0);
		glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, &views[i][0][0]);
		cube->draw();
	}

	prog->setInt("mode", 2);
	// prefiltered envmap
	const int maxMipLevels = 5;
	for (int mip = 0; mip < maxMipLevels; mip++) {
		float roughness = float(mip) / float(maxMipLevels - 1);
		int   size = prefilteredSize >> mip;
		prog->setFloat("roughness", roughness);
		glViewport(0, 0, size, size);
		for (int i = 0; i < 6; i++) {
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
				GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, prefilteredCube.id, mip);
			glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, &views[i][0][0]);
			cube->draw();
		}
	}

	// recover
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glViewport(vp[0], vp[1], vp[2], vp[3]);
	glCheckError();
}

void Envmap::genLUT()
{
	auto brdfProgram = Program::createProgramSPIRV(
		SCREEN_VERT, sizeof(SCREEN_VERT),
		BRDF_FRAG, sizeof(BRDF_FRAG));

	auto quad = Mesh::ScreenQuad();
	brdfProgram->use();

	fb.bind();
	fb.setTexture(GL_COLOR_ATTACHMENT0, &brdfLut);

	glViewport(0, 0, brdfLutSize, brdfLutSize);
	quad->draw();
	glCheckError();

	quad->release();
	delete quad;
	brdfProgram->release();
	delete brdfProgram;
}
