#include "ShadowMap.h"
#include "GLVisualModule.h"

#include <SceneGraph.h>
#include <Action.h>

#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <array>

#include "screen.vert.h"
#include "blur.frag.h"

namespace dyno 
{

	ShadowMap::ShadowMap(int w, int h): width(w), height(h)
	{
		const glm::vec4 border = glm::vec4(1);

		mShadowTex.format = GL_RG;
		mShadowTex.internalFormat = GL_RG32F;
		mShadowTex.maxFilter = GL_LINEAR;
		mShadowTex.minFilter = GL_LINEAR;
		mShadowTex.create();

		// setup border
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, glm::value_ptr(border));

		mShadowBlur.format = GL_RG;
		mShadowBlur.internalFormat = GL_RG32F;
		mShadowBlur.maxFilter = GL_LINEAR;
		mShadowBlur.minFilter = GL_LINEAR;
		mShadowBlur.create();

		// setup border
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, glm::value_ptr(border));

		mShadowDepth.internalFormat = GL_DEPTH_COMPONENT32;
		mShadowDepth.format = GL_DEPTH_COMPONENT;
		mShadowDepth.create();

		mShadowTex.resize(width, height);
		mShadowBlur.resize(width, height);
		mShadowDepth.resize(width, height);

		mFramebuffer.create();

		mFramebuffer.bind();
		mFramebuffer.setTexture2D(GL_DEPTH_ATTACHMENT, mShadowDepth.id);
		mFramebuffer.checkStatus();

		mFramebuffer.unbind();

		// uniform buffers
		mShadowUniform.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);

		// for blur depth textures
		mQuad = gl::Mesh::ScreenQuad();
		mBlurProgram = gl::Program::createProgramSPIRV(
			SCREEN_VERT, sizeof(SCREEN_VERT),
			BLUR_FRAG, sizeof(BLUR_FRAG));
	}

	ShadowMap::~ShadowMap()
	{
		mFramebuffer.release();
		mShadowTex.release();
		mShadowDepth.release();
		mShadowBlur.release();

		mShadowUniform.release();

		mQuad->release();
		delete mQuad;

		mBlurProgram->release();
		delete mBlurProgram;
	}


	// extract frustum corners from camera projection matrix
	std::array<glm::vec4, 8> getFrustumCorners(const glm::mat4& proj)
	{
		const glm::vec4 p[8] = {
		   glm::vec4(-1.0f, -1.0f, -1.0f, 1.0f),
		   glm::vec4(-1.0f, -1.0f, 1.0f, 1.0f),

		   glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f),
		   glm::vec4(-1.0f, 1.0f, 1.0f, 1.0f),

		   glm::vec4(1.0f, -1.0f, -1.0f, 1.0f),
		   glm::vec4(1.0f, -1.0f, 1.0f, 1.0f),

		   glm::vec4(1.0f, 1.0f, -1.0f, 1.0f),
		   glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
		};
		
		const glm::mat4 invProj = glm::inverse(proj);

		std::array<glm::vec4, 8> corners;
		for (int i = 0; i < 8; i++)
		{
			// camera space corners
			corners[i] = invProj * p[i];
			corners[i] /= corners[i].w;
		}

		return corners;
	}

	glm::mat4 getLightViewMatrix(glm::vec3 lightDir)
	{
		glm::vec3 lightUp = glm::vec3(0, 1, 0);
		if (glm::length(glm::cross(lightUp, lightDir)) == 0.f)
		{
			lightUp = glm::vec3(0, 0, 1);
		}
		glm::mat4 lightView = glm::lookAt(glm::vec3(0), -lightDir, lightUp);
		return lightView;
	}

	glm::mat4 getLightProjMatrix(glm::mat4 lightView,
		Vec3f lowerBound,
		Vec3f upperBound,
		glm::mat4 cameraView,
		glm::mat4 cameraProj)
	{
		glm::vec4 p[8] = {
			lightView * glm::vec4{lowerBound[0], lowerBound[1], lowerBound[2], 1},
			lightView * glm::vec4{lowerBound[0], lowerBound[1], upperBound[2], 1},
			lightView * glm::vec4{lowerBound[0], upperBound[1], lowerBound[2], 1},
			lightView * glm::vec4{lowerBound[0], upperBound[1], upperBound[2], 1},
			lightView * glm::vec4{upperBound[0], lowerBound[1], lowerBound[2], 1},
			lightView * glm::vec4{upperBound[0], lowerBound[1], upperBound[2], 1},
			lightView * glm::vec4{upperBound[0], upperBound[1], lowerBound[2], 1},
			lightView * glm::vec4{upperBound[0], upperBound[1], upperBound[2], 1},
		};
			   
		glm::vec4 bmin = p[0];
		glm::vec4 bmax = p[0];
		for (int i = 1; i < 8; i++)
		{
			bmin = glm::min(bmin, p[i]);
			bmax = glm::max(bmax, p[i]);
		}

		// frustrum clamp
		std::array<glm::vec4, 8> corners = getFrustumCorners(cameraProj);
		glm::mat4 tm = lightView * glm::inverse(cameraView);

		glm::vec4 fbmin = tm * corners[0];
		glm::vec4 fbmax = tm * corners[0];
		for (int i = 1; i < 8; i++)
		{
			glm::vec4 c = tm * corners[i];
			fbmin = glm::min(fbmin, c);
			fbmax = glm::max(fbmax, c);
		}

		bmin.x = glm::max(bmin.x, fbmin.x);
		bmin.y = glm::max(bmin.y, fbmin.y);
		bmax.x = glm::min(bmax.x, fbmax.x);
		bmax.y = glm::min(bmax.y, fbmax.y);

		float cx = (bmin.x + bmax.x) * 0.5;
		float cy = (bmin.y + bmax.y) * 0.5;
		float d = glm::max(bmax.y - bmin.y, bmax.x - bmin.x) * 0.5f;
		
		glm::mat4 lightProj = glm::ortho(cx - d, cx + d, cy - d, cy + d, -bmax.z, -bmin.z);
		return lightProj;
	}

	void ShadowMap::update(dyno::SceneGraph* scene, const dyno::RenderParams& rparams)
	{
		// initialization
		mFramebuffer.bind();
		mFramebuffer.setTexture2D(GL_COLOR_ATTACHMENT0, mShadowTex.id);
		mFramebuffer.clearDepth(1.0);
		mFramebuffer.clearColor(1.0, 1.0, 1.0, 1.0);

		if (rparams.light.mainLightShadow > 0.f	&& 
			scene != nullptr && !scene->isEmpty())
		{
			glViewport(0, 0, width, height);

			glm::mat4 lightView = getLightViewMatrix(rparams.light.mainLightDirection);
			glm::mat4 lightProj = getLightProjMatrix(lightView, 
				scene->getLowerBound(), 
				scene->getUpperBound(), 
				rparams.transforms.view, 
				rparams.transforms.proj);

			// draw objects to shadow texture
			class DrawShadow : public Action
			{
			public:
				void process(Node* node) override
				{
					if (!node->isVisible())	return;

					for (auto iter : node->graphicsPipeline()->activeModules()) {
						auto m = dynamic_cast<GLVisualModule*>(iter.get());
						if (m && m->isVisible()) {
							m->draw(params);
						}
					}
				}
				RenderParams params;
			} action;

			action.params.transforms.model = glm::mat4(1);
			action.params.transforms.view = lightView;
			action.params.transforms.proj = lightProj;
			action.params.mode = GLRenderMode::SHADOW;
			action.params.width = this->width;
			action.params.height = this->height;

			scene->traverseForward(&action);

			// blur shadow map		
			glDisable(GL_DEPTH_TEST);
			mBlurProgram->use();
			for (int i = 0; i < blurIters; i++)
			{
				mBlurProgram->setVec2("uScale", { 1.f / width, 0.f / height });
				mShadowTex.bind(GL_TEXTURE5);
				mFramebuffer.setTexture2D(GL_COLOR_ATTACHMENT0, mShadowBlur.id);
				mQuad->draw();

				mBlurProgram->setVec2("uScale", { 0.f / width, 1.f / height });
				mShadowBlur.bind(GL_TEXTURE5);
				mFramebuffer.setTexture2D(GL_COLOR_ATTACHMENT0, mShadowTex.id);
				mQuad->draw();
			}
			glEnable(GL_DEPTH_TEST);

			// update shadow map uniform
			struct {
				glm::mat4	transform;
				float		minValue;
			} shadow;

			shadow.transform = lightProj * lightView * glm::inverse(rparams.transforms.view);
			shadow.minValue = minValue;
			mShadowUniform.load(&shadow, sizeof(shadow));
		}

	}

	void ShadowMap::bind(int shadowUniformLoc, int shadowTexSlot)
	{		
		// bind the shadow texture to the slot
		mShadowUniform.bindBufferBase(shadowUniformLoc);

		if (shadowTexSlot >= GL_TEXTURE0) 
			mShadowTex.bind(shadowTexSlot);
		else
			mShadowTex.bind(GL_TEXTURE0 + shadowTexSlot);

	}

}

