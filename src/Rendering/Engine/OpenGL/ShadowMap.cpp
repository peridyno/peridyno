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

	ShadowMap::ShadowMap(int size)
	{
		this->setSize(size);
	}

	ShadowMap::~ShadowMap()
	{

	}

	void ShadowMap::initialize()
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

		mShadowTex.resize(size, size);
		mShadowBlur.resize(size, size);
		mShadowDepth.resize(size, size);
		sizeUpdated = false;

		mFramebuffer.create();

		mFramebuffer.bind();
		mFramebuffer.setTexture(GL_DEPTH_ATTACHMENT, &mShadowDepth);
		mFramebuffer.checkStatus();

		mFramebuffer.unbind();

		// uniform buffers
		mShadowUniform.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);

		// for blur depth textures
		mQuad = Mesh::ScreenQuad();
		mBlurProgram = Program::createProgramSPIRV(
			SCREEN_VERT, sizeof(SCREEN_VERT),
			BLUR_FRAG, sizeof(BLUR_FRAG));
	}

	void ShadowMap::release()
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

	glm::vec4 computePlane(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c)
	{
		glm::vec3 normal = glm::normalize(glm::cross(b - a, c - a));
		float d = -glm::dot(normal, a);
		return glm::vec4(normal, d);
	}


	std::array<glm::vec4, 6> getFrustumPlanes(const std::array<glm::vec4, 8>& corners)
	{
		std::array<glm::vec4, 6> planes;

		planes[0] = computePlane(glm::vec3(corners[1]), glm::vec3(corners[3]), glm::vec3(corners[5]));
		planes[1] = computePlane(glm::vec3(corners[0]), glm::vec3(corners[4]), glm::vec3(corners[2]));
		planes[2] = computePlane(glm::vec3(corners[0]), glm::vec3(corners[1]), glm::vec3(corners[2]));
		planes[3] = computePlane(glm::vec3(corners[4]), glm::vec3(corners[6]), glm::vec3(corners[5]));
		planes[4] = computePlane(glm::vec3(corners[2]), glm::vec3(corners[3]), glm::vec3(corners[6]));
		planes[5] = computePlane(glm::vec3(corners[0]), glm::vec3(corners[4]), glm::vec3(corners[1]));

		return planes;
	}

	bool isBoxInFrustum(const Vec3f& up, const Vec3f& low, const std::array<glm::vec4, 6>& planes)
	{
		for (const auto& plane : planes)
		{
			const glm::vec3 normal(plane.x, plane.y, plane.z);
			float w = plane.w;

			glm::vec3 negativeVertex;
			negativeVertex.x = (normal.x >= 0) ? low.x : up.x;
			negativeVertex.y = (normal.y >= 0) ? low.y : up.y;
			negativeVertex.z = (normal.z >= 0) ? low.z : up.z;

			float dist = glm::dot(normal, negativeVertex) + w;

			if (dist > 0)
				return false; 
		}
		return true;
	}


	void ShadowMap::getMergedBoundingBoxInFrustum(const glm::mat4& proj, const std::vector<Vec3f>& up, const std::vector<Vec3f>& low, Vec3f& resultUp, Vec3f& resultLow)
	{
		auto frustumCorners = getFrustumCorners(proj);
		auto planes = getFrustumPlanes(frustumCorners);

		bool hasBoxInFrustum = false;

		for (size_t i = 0; i < up.size(); i++)
		{
			Vec3f UpperBound = up[i];
			Vec3f LowerBound = low[i];

			if (isBoxInFrustum(UpperBound, LowerBound, planes))
			{
				if (!hasBoxInFrustum)
				{
					resultUp = up[i];
					resultLow = low[i];
					hasBoxInFrustum = true;
				}
				else
				{
					resultLow.x = glm::min(resultLow.x, low[i].x);
					resultLow.y = glm::min(resultLow.y, low[i].y);
					resultLow.z = glm::min(resultLow.z, low[i].z);

					resultUp.x = glm::max(resultUp.x, up[i].x);
					resultUp.y = glm::max(resultUp.y, up[i].y);
					resultUp.z = glm::max(resultUp.z, up[i].z);
				}
			}
		}
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

	void GetBoundingBoxOfAllNodes(dyno::SceneGraph* scene, std::vector<Vec3f>& uppers, std::vector<Vec3f>& lowers, bool clampToSceneBounds)
	{
		if (!scene->isEmpty())
		{
			Vec3f sceneLower = scene->getLowerBound();
			Vec3f sceneUpper = scene->getUpperBound();

			for (SceneGraph::Iterator itor = scene->begin(); itor != scene->end(); itor++) {
				auto node = itor.get();
				if (node->isVisible())
				{
					if (clampToSceneBounds)
					{
						uppers.push_back(Vec3f(
							node->boundingBox().upper.x > sceneUpper.x ? sceneUpper.x : node->boundingBox().upper.x,
							node->boundingBox().upper.y > sceneUpper.y ? sceneUpper.y : node->boundingBox().upper.y,
							node->boundingBox().upper.z > sceneUpper.z ? sceneUpper.z : node->boundingBox().upper.z
						));
						lowers.push_back(Vec3f(
							node->boundingBox().lower.x < sceneLower.x ? sceneLower.x : node->boundingBox().lower.x,
							node->boundingBox().lower.y < sceneLower.y ? sceneLower.y : node->boundingBox().lower.y,
							node->boundingBox().lower.z < sceneLower.z ? sceneLower.z : node->boundingBox().lower.z
						));
					}
					else
					{
						uppers.push_back(node->boundingBox().upper);
						lowers.push_back(node->boundingBox().lower);
					}
				}
			}

		}
	}

	void ShadowMap::update(dyno::SceneGraph* scene, const dyno::RenderParams& rparams)
	{
		if (sizeUpdated)
		{
			mShadowTex.resize(size, size);
			mShadowBlur.resize(size, size);
			mShadowDepth.resize(size, size);
			sizeUpdated = false;
		}

		// initialization
		mFramebuffer.bind();
		mFramebuffer.setTexture(GL_COLOR_ATTACHMENT0, &mShadowTex);
		mFramebuffer.clearDepth(1.0);
		mFramebuffer.clearColor(1.0, 1.0, 1.0, 1.0);

		Vec3f resultUp, resultLow;
		if (useSceneBounds) 
		{
			resultUp = scene->getUpperBound();
			resultLow = scene->getLowerBound();
		}
		else 
		{
			std::vector<Vec3f> uppers;
			std::vector<Vec3f> lowers;
			GetBoundingBoxOfAllNodes(scene, uppers, lowers, clampToSceneBounds);

			getMergedBoundingBoxInFrustum(rparams.transforms.proj, uppers, lowers, resultUp, resultLow);
		}

		if (rparams.light.mainLightShadow > 0.f	&& 
			scene != nullptr && !scene->isEmpty())
		{
			glViewport(0, 0, size, size);

			glm::mat4 lightView = getLightViewMatrix(rparams.light.mainLightDirection);
			glm::mat4 lightProj = getLightProjMatrix(lightView, 
				resultLow,
				resultUp,
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
			action.params.width = this->size;
			action.params.height = this->size;

			scene->traverseForward(&action);

			// blur shadow map		
			glDisable(GL_DEPTH_TEST);
			mBlurProgram->use();
			for (int i = 0; i < blurIters; i++)
			{
				mBlurProgram->setVec2("uScale", { 1.f / size, 0.f / size });
				mShadowTex.bind(GL_TEXTURE5);
				mFramebuffer.setTexture(GL_COLOR_ATTACHMENT0, &mShadowBlur);
				mQuad->draw();

				mBlurProgram->setVec2("uScale", { 0.f / size, 1.f / size });
				mShadowBlur.bind(GL_TEXTURE5);
				mFramebuffer.setTexture(GL_COLOR_ATTACHMENT0, &mShadowTex);
				mQuad->draw();
			}
			glEnable(GL_DEPTH_TEST);

			// update shadow map uniform
			struct {
				glm::mat4	transform;
				float		minValue;
				//int		range;
				//float    lightRadius;
			} shadow;

			shadow.transform = lightProj * lightView * glm::inverse(rparams.transforms.view);
			shadow.minValue = minValue;
			//shadow.lightRadius = mLightRadius;
			//shadow.range = this->blurIters;

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

	int ShadowMap::getSize() const
	{
		return this->size;
	}

	void ShadowMap::setSize(int size)
	{
		if (this->size == size)
			return;

		this->size = size;
		this->sizeUpdated = true;
	}

	int ShadowMap::getNumBlurIterations() const
	{
		return this->blurIters;
	}

	void ShadowMap::setNumBlurIterations(int iter)
	{
		this->blurIters = iter;
	}

	void ShadowMap::setLightRadius(float radius)
	{
		this->mLightRadius = radius;
	}

	float ShadowMap::getLightRadius()
	{	
		return this->mLightRadius;
	}

}

