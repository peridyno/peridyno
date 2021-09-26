#include "ShadowMap.h"
#include "GLVisualModule.h"

#include <SceneGraph.h>
#include <Action.h>

#include <glad/glad.h>
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
		//mShadowDepth.minFilter = GL_NEAREST_MIPMAP_NEAREST;
		mShadowDepth.create();

		mShadowDepth.resize(width, height, 4);
		//mShadowDepth.genMipmap(); // need to create mipmap texture

		mFramebuffer.create();
		mFramebuffer.bind();
		glDrawBuffer(GL_NONE);
		mFramebuffer.unbind();

		// uniform buffers
		mTransformUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW); 
		mShadowMatrixUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
	}

	// extract frustum corners from camera projection matrix
	void getFrustumCorners(const glm::mat4& proj, glm::vec4 corners[8])
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

		for (int i = 0; i < 8; i++)
		{
			// camera space corners
			corners[i] = invProj * p[i];
			corners[i] /= corners[i].w;
		}
	}

	// slice frustum corners and get the points of slice plane
	void getSplitCorners(const glm::vec4 frustumCorners[8], float split, glm::vec4 p[4])
	{
		for (int i = 0; i < 4; i++)
		{
			p[i] = glm::mix(frustumCorners[i * 2], frustumCorners[i * 2 + 1], split);
		}
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

	struct ShadowMapSplit
	{
		glm::mat4 projection;
		float	  minDepth;
		float     maxDepth;
	};

	std::vector<ShadowMapSplit> getShadowMapSplit(dyno::SceneGraph* scene,
		const dyno::RenderParams & rparams)
	{
		const int NUM_SPLIT = 4;

		std::vector<ShadowMapSplit> split(NUM_SPLIT);

		glm::vec4 corners[8];
		getFrustumCorners(rparams.proj, corners);

		glm::vec4 splitCorners[NUM_SPLIT + 1][4];
		float     splitDepth[NUM_SPLIT + 1];
		
		for (int i = 0; i <= NUM_SPLIT; i++)
		{
			getSplitCorners(corners, float(i) / NUM_SPLIT, splitCorners[i]);
			splitDepth[i] = splitCorners[i][0].z;
		}

		glm::vec4 bbox[NUM_SPLIT][2];
		
		// get bounding box in light space
		const glm::mat4 lightView = getLightView(rparams.light.mainLightDirection);
		const glm::mat4 invView = glm::inverse(rparams.view);

		for (int i = 0; i < NUM_SPLIT; i++)
		{
			glm::vec4& b0 = bbox[i][0];
			glm::vec4& b1 = bbox[i][1];

			b0 = glm::vec4(FLT_MAX);
			b1 = glm::vec4(-FLT_MAX);

			for (int j = 0; j < 4; j++)
			{
				for (int k = 0; k < 2; k++)
				{
					glm::vec4 p = splitCorners[i + k][j];
					p = lightView * invView * p;
					b0 = glm::min(b0, p);
					b1 = glm::max(b1, p);
				}
			}
		}

		float zMin = -1.f;
		float zMax =  1.f;

		if (scene)
		{
			// get bounding box of the scene
			auto p0 = scene->getLowerBound();
			auto p1 = scene->getUpperBound();
			glm::vec3 pmin = { p0[0], p0[1], p0[2] };
			glm::vec3 pmax = { p1[0], p1[1], p1[2] };

			float r = glm::distance(pmin, pmax) * 0.5f;

			glm::vec3 center = (pmin + pmax) * 0.5f;
			center = glm::vec3(lightView * glm::vec4(center, 1));
			zMin = - center.z - r;
			zMax = - center.z + r;
		}
		

		for (int i = 0; i < NUM_SPLIT; i++)
		{
			split[i].projection = glm::ortho(bbox[i][0].x, bbox[i][1].x, 
				bbox[i][0].y, bbox[i][1].y, 
				zMin, zMax);

			split[i].minDepth = splitDepth[i];
			split[i].maxDepth = splitDepth[i + 1];
		}

		return split;
	}

	void ShadowMap::update(dyno::SceneGraph* scene, const dyno::RenderParams & rparams)
	{
		std::vector<ShadowMapSplit> split = getShadowMapSplit(scene, rparams);

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

		lightMVP.width = width;
		lightMVP.height = height;
		lightMVP.model = glm::mat4(1);		
		lightMVP.view = getLightView(rparams.light.mainLightDirection);
		
		mFramebuffer.bind();

		for (int i = 0; i < 4; i++)
		{
			//int w = width >> i;
			//int h = height >> i;
			lightMVP.projection = split[i].projection;

			mTransformUBO.load(&lightMVP, sizeof(lightMVP));
			mTransformUBO.bindBufferBase(0);

			// draw depth
			//mFramebuffer.setTexture2D(GL_DEPTH_ATTACHMENT, mShadowDepth.id, i);
			
			glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, mShadowDepth.id, 0, i);

			mFramebuffer.checkStatus();
			mFramebuffer.clearDepth(1.0);
			glViewport(0, 0, width, height);
			gl::glCheckError();
			// shadow pass
			if ((scene != 0) && (scene->getRootNode() != 0))
			{
				scene->getRootNode()->traverseTopDown<DrawDepth>();
			}
		}

		// bind the shadow texture to the slot
		mShadowDepth.bind(GL_TEXTURE5);

		// set the cascaded shadowmap matrix and depth
		struct
		{
			glm::mat4	transform[4];
			float		minDepth[4];
			float		maxDepth[4];
		} data;

		for (int i = 0; i < 4; i++)
		{
			data.transform[i] = split[i].projection * lightMVP.view * glm::inverse(rparams.view);
			data.minDepth[i] = fabs(split[i].minDepth);
			data.maxDepth[i] = fabs(split[i].maxDepth);
		}
		mShadowMatrixUBO.load(&data, sizeof(data));
		mShadowMatrixUBO.bindBufferBase(2);

	}
}

