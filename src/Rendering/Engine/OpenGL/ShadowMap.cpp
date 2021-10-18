#include "ShadowMap.h"
#include "GLVisualModule.h"

#include <SceneGraph.h>
#include <Action.h>

#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>

#include <array>

namespace dyno 
{
	// draw depth for shadow map
	class DrawDepth : public Action
	{
	private:
		void process(Node* node) override
		{
			if (!node->isVisible())
				return;

			for (auto iter : node->graphicsPipeline()->activeModules())
			{
				auto m = dynamic_cast<GLVisualModule*>(iter);
				if (m && m->isVisible())
				{
					m->paintGL(dyno::GLVisualModule::DEPTH);
				}
			}
		}
	};


	ShadowMap::ShadowMap(int w, int h)
	{
		width = w;
		height = h;
	}

	ShadowMap::~ShadowMap()
	{

	}

	void ShadowMap::initialize()
	{
		mShadowDepth.internalFormat = GL_DEPTH_COMPONENT32;
		mShadowDepth.format = GL_DEPTH_COMPONENT;
		mShadowDepth.create();

		mShadowDepth.resize(width, height);

		mFramebuffer.create();
		mFramebuffer.bind();
		glDrawBuffer(GL_NONE);
		mFramebuffer.unbind();

		// uniform buffers
		mTransformUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW); 
		mShadowMatrixUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
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

	glm::mat4 getLightView(glm::vec3 lightDir)
	{
		glm::vec3 lightUp = glm::vec3(0, 1, 0);
		if (glm::length(glm::cross(lightUp, lightDir)) == 0.f)
		{
			lightUp = glm::vec3(0, 0, 1);
		}
		glm::mat4 lightView = glm::lookAt(glm::vec3(0), -lightDir, lightUp);
		return lightView;
	}

	glm::mat4 getLightProj(glm::mat4 lightView, dyno::SceneGraph* scene, const dyno::RenderParams& rparams)
	{
		Vec3f bbox[2] = { scene->getLowerBound(), scene->getUpperBound() };

		glm::vec4 p[8] = {
			lightView * glm::vec4{bbox[0][0], bbox[0][1], bbox[0][2], 1},
			lightView * glm::vec4{bbox[0][0], bbox[0][1], bbox[1][2], 1},
			lightView * glm::vec4{bbox[0][0], bbox[1][1], bbox[0][2], 1},
			lightView * glm::vec4{bbox[0][0], bbox[1][1], bbox[1][2], 1},
			lightView * glm::vec4{bbox[1][0], bbox[0][1], bbox[0][2], 1},
			lightView * glm::vec4{bbox[1][0], bbox[0][1], bbox[1][2], 1},
			lightView * glm::vec4{bbox[1][0], bbox[1][1], bbox[0][2], 1},
			lightView * glm::vec4{bbox[1][0], bbox[1][1], bbox[1][2], 1},
		};

		glm::vec4 bmin = p[0];
		glm::vec4 bmax = p[0];
		for (int i = 1; i < 8; i++)
		{
			bmin = glm::min(bmin, p[i]);
			bmax = glm::max(bmax, p[i]);
		}


		glm::mat4 lightProj = glm::ortho(-1, 1, -1, 1, -1, 1);
		return lightProj;
	}

	void ShadowMap::update(dyno::SceneGraph* scene, const dyno::RenderParams & rparams)
	{
		glm::mat4 lightView = getLightView(rparams.light.mainLightDirection);
		glm::mat4 lightProj = getLightProj(lightView, scene, rparams);

		// update light transform infomation
		struct
		{
			// MVP
			glm::mat4 model;
			glm::mat4 view;
			glm::mat4 proj;
			int width;
			int height;
		} lightMVP;

		lightMVP.width  = width;
		lightMVP.height = height;
		lightMVP.model  = glm::mat4(1);		
		lightMVP.view   = lightView;
		lightMVP.proj   = lightProj;
			   
		mTransformUBO.load(&lightMVP, sizeof(lightMVP));
		mTransformUBO.bindBufferBase(0);

		// draw depth
		mFramebuffer.bind();
		mFramebuffer.setTexture2D(GL_DEPTH_ATTACHMENT, mShadowDepth.id);			
		//glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, mShadowDepth.id, 0, i);

		mFramebuffer.checkStatus();
		mFramebuffer.clearDepth(1.0);

		glViewport(0, 0, width, height);
		gl::glCheckError();
		// shadow pass
		if ((scene != 0) && (scene->getRootNode() != 0))
		{
			scene->getRootNode()->traverseTopDown<DrawDepth>();
		}

		// bind the shadow texture to the slot
		mShadowDepth.bind(GL_TEXTURE5);
	
		// shadow map uniform
		struct {
			glm::mat4 transform;
			float bias0;
			float bias1;
			float radius;
			float clamp;
		} shadow;
		shadow.transform = lightProj * lightView * glm::inverse(rparams.view);
		shadow.bias0 = bias0;
		shadow.bias1 = bias1;
		shadow.radius = radius; 
		shadow.clamp = clamp;

		mShadowMatrixUBO.load(&shadow, sizeof(shadow));
		mShadowMatrixUBO.bindBufferBase(2);
	}
}

