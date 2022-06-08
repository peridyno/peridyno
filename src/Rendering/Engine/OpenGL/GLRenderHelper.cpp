#include "GLRenderHelper.h"

#include "gl/Program.h"
#include "gl/Mesh.h"
#include "gl/Texture.h"

#include <glad/glad.h>
#include <vector>

namespace dyno
{
	class GroundRenderer
	{
	public:
		GroundRenderer()
		{
			mProgram = gl::CreateShaderProgram("plane.vert", "plane.frag");
			mPlane = gl::Mesh::Plane(1.f);

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

			mRulerTex.wrapS = GL_REPEAT;
			mRulerTex.wrapT = GL_REPEAT;
			mRulerTex.minFilter = GL_LINEAR_MIPMAP_LINEAR;
			mRulerTex.maxFilter = GL_LINEAR;
			mRulerTex.format = GL_RED;
			mRulerTex.internalFormat = GL_RED;
			mRulerTex.type = GL_UNSIGNED_BYTE;

			mRulerTex.create();
			mRulerTex.load(w, h, img.data());
			mRulerTex.genMipmap();
			// set anisotropy
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, 16);
		}


		void draw(float planeScale, float rulerScale)
		{
			glEnable(GL_BLEND);
			glEnable(GL_CULL_FACE);

			mRulerTex.bind(GL_TEXTURE1);
			
			mProgram.use();
			mProgram.setFloat("uPlaneScale", planeScale);
			mProgram.setFloat("uRulerScale", rulerScale);

			mPlane.draw();
			
			glDisable(GL_CULL_FACE);
			glDisable(GL_BLEND);

			// clear depth to get avoid object cross ground
			glClear(GL_DEPTH_BUFFER_BIT);

			gl::glCheckError();
		}

	private:
		gl::Mesh			mPlane;
		gl::Texture2D 		mRulerTex;
		gl::Program			mProgram;
	};

	class AxisRenderer
	{
	public:
		AxisRenderer()
		{
			mProgram = gl::CreateShaderProgram("axis.vert", "axis.frag");

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
			mProgram.use();
			mAxisVAO.bind();

			glDisable(GL_DEPTH_TEST);
			glDrawArrays(GL_LINES, 0, 6);
			glEnable(GL_DEPTH_TEST);

			mAxisVAO.unbind();
		}

	private:
		gl::VertexArray	mAxisVAO;
		gl::Buffer		mAxisVBO;
		gl::Program		mProgram;
	};

	class BBoxRenderer
	{
	public:
		BBoxRenderer()
		{
			mProgram = gl::CreateShaderProgram("bbox.vert", "bbox.frag");
			mCubeVBO.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
			mCubeVBO.load(0, 8 * 3 * sizeof(float));
			mCubeVAO.create();
			mCubeVAO.bindVertexBuffer(&mCubeVBO, 0, 3, GL_FLOAT, 0, 0, 0);
		}

		void draw(Vec3f p0, Vec3f p1, int type)
		{
			float vertices[]{
				p0[0], p0[1], p0[2],
				p0[0], p0[1], p1[2],
				p1[0], p0[1], p1[2],
				p1[0], p0[1], p0[2],

				p0[0], p1[1], p0[2],
				p0[0], p1[1], p1[2],
				p1[0], p1[1], p1[2],
				p1[0], p1[1], p0[2],
			};

			mCubeVBO.load(vertices, sizeof(vertices));
			mCubeVAO.bind();
			mProgram.use();
			mProgram.setVec4("uColor", Vec4f(0.75));

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
			gl::glCheckError();
		}

	private:
		gl::VertexArray	mCubeVAO;
		gl::Buffer		mCubeVBO;
		gl::Program		mProgram;
	};

	class BackgroundRenderer
	{
	public:
		BackgroundRenderer()
		{
			// create a quad object
			mScreenQuad = gl::Mesh::ScreenQuad();
			mBackgroundProgram = gl::CreateShaderProgram("screen.vert", "background.frag");
		}

		void draw(Vec3f color0, Vec3f color1)
		{
			// render background
			mBackgroundProgram.use();
			mBackgroundProgram.setVec3("uColor0", color0);
			mBackgroundProgram.setVec3("uColor1", color1);
			mScreenQuad.draw();
		}

	private:
		// background
		gl::Program		mBackgroundProgram;
		gl::Mesh		mScreenQuad;
	};



	GLRenderHelper::GLRenderHelper()
	{
	}

	GLRenderHelper::~GLRenderHelper()
	{
		if (mAxisRenderer) delete mAxisRenderer;
		if (mBBoxRenderer) delete mBBoxRenderer;
		if (mGroundRenderer) delete mGroundRenderer;
		if (mBackgroundRenderer) delete mBackgroundRenderer;
	}

	void GLRenderHelper::initialize()
	{
		mAxisRenderer = new AxisRenderer();
		mBBoxRenderer = new BBoxRenderer();
		mGroundRenderer = new GroundRenderer();
		mBackgroundRenderer = new BackgroundRenderer();
	}

	void GLRenderHelper::drawGround(float planeScale, float rulerScale)
	{
		if (mGroundRenderer != NULL)
			mGroundRenderer->draw(planeScale, rulerScale);
	}

	void GLRenderHelper::drawAxis(float lineWidth)
	{
		if (mAxisRenderer != NULL)
			mAxisRenderer->draw(lineWidth);
	}

	void GLRenderHelper::drawBBox(Vec3f p0, Vec3f p1, int type)
	{
		if (mBBoxRenderer != NULL)
			mBBoxRenderer->draw(p0, p1, type);
	}

	void GLRenderHelper::drawBackground(Vec3f color0, Vec3f color1)
	{
		if (mBackgroundRenderer != NULL)
			mBackgroundRenderer->draw(color0, color1);
	}
}
