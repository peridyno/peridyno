#include "RenderHelper.h"
#include "Utility.h"
#include "GLShader.h"
#include "GLVertexArray.h"

#include <glad/glad.h>

#include <vector>

class GroundRenderer
{
public:
	GroundRenderer()
	{
		mGroundProgram = CreateShaderProgram("plane.vert", "plane.frag");
		mGroundMesh = GLMesh::Plane(1.f);

		// create ruler texture
		const int k = 50;
		const int w = k * 10 + 1;
		const int h = k * 10 + 1;
		std::vector<char> img(w * h, 0);
		for (int j = 0; j < h; ++j)
		{
			for (int i = 0; i < w; ++i)
			{
				if (j == 0 || i == 0 || j == (h - 1) || i == (w - 1))
					img[j * w + i] = 192;
				else if ((j % k) == 0 || (i % k) == 0)
					img[j * w + i] = 128;
			}
		}

		glGenTextures(1, &mGroundTex);
		glBindTexture(GL_TEXTURE_2D, mGroundTex);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, img.data());
		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, 16);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);

		glCheckError();
	}

	void draw(float scale = 3.f)
	{
		glEnable(GL_BLEND);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, mGroundTex);
		mGroundProgram.use();

		mGroundProgram.setFloat("uScale", scale);

		glEnable(GL_CULL_FACE);
		mGroundMesh.draw();
		glDisable(GL_CULL_FACE);

		glBindTexture(GL_TEXTURE_2D, 0);
		glDisable(GL_BLEND);
		glCheckError();
	}

private:
	GLMesh			mGroundMesh;
	unsigned int 	mGroundTex;
	GLShaderProgram mGroundProgram;
};

class AxisRenderer
{
public:
	AxisRenderer()
	{
		mAxisProgram = CreateShaderProgram("axis.vert", "axis.frag");

		mAxisVBO.create(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
		float vertices[] = {
			// +x			// color...
			0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
			1.f, 0.f, 0.f, 1.f, 0.f, 0.f,
			// +y
			0.f, 0.f, 0.f, 0.f, 1.f, 0.f,
			0.f, 1.f, 0.f, 0.f, 1.f, 0.f,
			// +z
			0.f, 0.f, 0.f, 0.f, 0.f, 1.f,
			0.f, 0.f, 1.f, 0.f, 0.f, 1.f,
		};
		mAxisVBO.load(vertices, sizeof(vertices) * 4);

		mAxisVAO.create();
		mAxisVAO.bindVertexBuffer(&mAxisVBO, 0, 3, GL_FLOAT, sizeof(vertices) / 6, 0, 0);
		mAxisVAO.bindVertexBuffer(&mAxisVBO, 1, 3, GL_FLOAT, sizeof(vertices) / 6, sizeof(float) * 3, 0);
	}

	void draw(float lineWidth = 2.f)
	{
		mAxisProgram.use();
		mAxisVAO.bind();

		glLineWidth(lineWidth);
		glDisable(GL_DEPTH_TEST);
		glDrawArrays(GL_LINES, 0, 6);
		glEnable(GL_DEPTH_TEST);

		mAxisVAO.unbind();
		glCheckError();
	}

private:
	GLVertexArray	mAxisVAO;
	GLBuffer		mAxisVBO;
	GLShaderProgram mAxisProgram;
};

class BBoxRenderer
{
public:
	BBoxRenderer()
	{
		mBBoxProgram = CreateShaderProgram("bbox.vert", "bbox.frag");
		mCubeVBO.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mCubeVBO.load(0, 8 * 3 * sizeof(float));
		mCubeVAO.create();
		mCubeVAO.bindVertexBuffer(&mCubeVBO, 0, 3, GL_FLOAT, 0, 0, 0);
	}

	void draw(glm::vec3 p0, glm::vec3 p1, int type)
	{
		float vertices[]{
			p0.x, p0.y, p0.z,
			p0.x, p0.y, p1.z,
			p1.x, p0.y, p1.z,
			p1.x, p0.y, p0.z,

			p0.x, p1.y, p0.z,
			p0.x, p1.y, p1.z,
			p1.x, p1.y, p1.z,
			p1.x, p1.y, p0.z,
		};

		mCubeVBO.load(vertices, sizeof(vertices));
		mCubeVAO.bind();
		mBBoxProgram.use();
		mBBoxProgram.setVec4("uColor", glm::vec4(0.75));

		if (true)
		{
			const unsigned int indices[]
			{
				0, 1, 1, 2, 2, 3, 3, 0,
				4, 5, 5, 6, 6, 7, 7, 4,
				0, 4, 1, 5, 2, 6, 3, 7,
			};

			glEnable(GL_DEPTH_TEST);
			glEnable(GL_BLEND);
			glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, indices);
			glDisable(GL_BLEND);
		}
		else
		{
			const unsigned int indices[]
			{
				0, 1, 2, 2, 3, 0, // bottom
				7, 6, 5, 5, 4, 7, // top
				4, 5, 1, 1, 0, 4, // left
				3, 2, 6, 6, 7, 3, // right
				0, 3, 7, 7, 4, 0, // front
				5, 6, 2, 2, 1, 5, // back
			};
			glEnable(GL_CULL_FACE);
			glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, indices);
			glDisable(GL_CULL_FACE);
		}

		mCubeVAO.unbind();
		glCheckError();
	}

private:
	GLVertexArray	mCubeVAO;
	GLBuffer		mCubeVBO;
	GLShaderProgram mBBoxProgram;
};

class BackgroundRenderer
{
public:
	BackgroundRenderer()
	{
		// create a quad object
		mScreenQuad = GLMesh::ScreenQuad();
		mBackgroundProgram = CreateShaderProgram("screen.vert", "background.frag");
	}

	void draw(glm::vec3 color0, glm::vec3 color1)
	{
		// render background
		mBackgroundProgram.use();
		mBackgroundProgram.setVec3("uColor0", color0);
		mBackgroundProgram.setVec3("uColor1", color1);
		mScreenQuad.draw();
	}

private:
	// background
	GLShaderProgram mBackgroundProgram;
	GLMesh			mScreenQuad;
};



RenderHelper::RenderHelper()
{
}

RenderHelper::~RenderHelper()
{
	if (mAxisRenderer) delete mAxisRenderer;
	if (mBBoxRenderer) delete mBBoxRenderer;
	if (mGroundRenderer) delete mGroundRenderer;
	if (mBackgroundRenderer) delete mBackgroundRenderer;
}

void RenderHelper::initialize()
{
	mAxisRenderer		= new AxisRenderer();
	mBBoxRenderer		= new BBoxRenderer();
	mGroundRenderer		= new GroundRenderer();
	mBackgroundRenderer = new BackgroundRenderer();
}

void RenderHelper::drawGround(float scale)
{
	if (mGroundRenderer != NULL)
		mGroundRenderer->draw(scale);
}

void RenderHelper::drawAxis(float lineWidth)
{	
	if(mAxisRenderer != NULL)
		mAxisRenderer->draw(lineWidth);
}

void RenderHelper::drawBBox(glm::vec3 p0, glm::vec3 p1, int type)
{
	if (mBBoxRenderer != NULL)
		mBBoxRenderer->draw(p0, p1, type);
}

void RenderHelper::drawBackground(glm::vec3 color0, glm::vec3 color1)
{
	if (mBackgroundRenderer != NULL)
		mBackgroundRenderer->draw(color0, color1);
}