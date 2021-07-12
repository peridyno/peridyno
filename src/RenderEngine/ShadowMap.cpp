#include "ShadowMap.h"
#include <glad/glad.h>
#include "Framework/SceneGraph.h"
#include "Action/Action.h"

#include <glm/gtc/matrix_transform.hpp>

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
		mFramebuffer.setTexture2D(GL_DEPTH_ATTACHMENT, mShadowDepth.id);
		mFramebuffer.bind();
		glDrawBuffer(GL_NONE);

		mFramebuffer.checkStatus();
		mFramebuffer.unbind();

		// uniform buffers
		mTransformUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW); 
		mShadowMatrixUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
	}

	void ShadowMap::update(dyno::SceneGraph* scene, const dyno::RenderParams & rparams)
	{
		// update light transform infomation
		struct
		{
			// MVP
			glm::mat4 model;
			glm::mat4 view;
			glm::mat4 projection;
			int width;
			int height;
		} lightMVP;

		lightMVP.model = glm::mat4(1);		
		lightMVP.width = this->width;
		lightMVP.height = this->height;

		glm::vec3 center = glm::vec3(0.f);
		float	  radius = 1.f;
		glm::vec3 lightUp = glm::vec3(0, 1, 0);

		if (scene)
		{
			// get bounding box of the scene
			auto p0 = scene->getLowerBound();
			auto p1 = scene->getUpperBound();
			glm::vec3 pmin = { p0[0], p0[1], p0[2] };
			glm::vec3 pmax = { p1[0], p1[1], p1[2] };
			center = (pmin + pmax) * 0.5f;
			radius = glm::distance(pmin, pmax) * 0.5f;
		}

		if (glm::length(glm::cross(lightUp, rparams.light.mainLightDirection)) == 0.f)
		{
			lightUp = glm::vec3(0, 0, 1);
		}

		lightMVP.projection = glm::ortho(-radius, radius, -radius, radius, -radius, radius);
		lightMVP.view = glm::lookAt(center, center - rparams.light.mainLightDirection, lightUp);

		mTransformUBO.load(&lightMVP, sizeof(lightMVP));
		mTransformUBO.bindBufferBase(0);

		// draw depth
		mFramebuffer.bind();
		mFramebuffer.clearDepth(1.0);
		glViewport(0, 0, width, height);

		// shadow pass
		if ((scene != 0) && (scene->getRootNode() != 0))
		{
			scene->getRootNode()->traverseTopDown<DrawDepth>();
		}

		// bind the shadow texture to the slot
		mShadowDepth.bind(GL_TEXTURE5);

		// set the shadowmap lookup matrix uniform buffer
		struct
		{
			glm::mat4 transform;
			glm::vec4 wtf;
		} shadowMat;

		shadowMat.wtf = glm::vec4(1, 0, 0, 1);
		// transform uniform block for lookup shadow maps
		shadowMat.transform = lightMVP.projection * lightMVP.view * glm::inverse(rparams.view);
		mShadowMatrixUBO.load(&shadowMat, sizeof(shadowMat));
		mShadowMatrixUBO.bindBufferBase(2);

	}
}

