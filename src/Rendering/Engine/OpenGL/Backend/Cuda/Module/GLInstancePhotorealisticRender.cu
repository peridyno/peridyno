#include "GLInstancePhotorealisticRender.h"
#include "Utility.h"

#include <glad/glad.h>

#include "surface.vert.h"
#include "surface.frag.h"
#include "surface.geom.h"

namespace dyno
{
	IMPLEMENT_CLASS(GLInstancePhotorealisticRender)

	GLInstancePhotorealisticRender::GLInstancePhotorealisticRender()
		: GLPhotorealisticRender()
	{
		
	}

	GLInstancePhotorealisticRender::~GLInstancePhotorealisticRender()
	{
	
	}

	std::string GLInstancePhotorealisticRender::caption()
	{
		return "Photorealistic Instance Render";
	}

	bool GLInstancePhotorealisticRender::initializeGL()
	{
		return GLPhotorealisticRender::initializeGL();
	}

	void GLInstancePhotorealisticRender::releaseGL()
	{
		GLPhotorealisticRender::releaseGL();

	}

	void GLInstancePhotorealisticRender::updateGL()
	{
		GLPhotorealisticRender::updateGL();

		auto& instances = this->inInstances()->constData();

		for (int i = 0; i < instances.size(); i++)
		{
			instances[i]->create();
			instances[i]->updateGL();
		}	
	}


	void GLInstancePhotorealisticRender::updateImpl()
	{
		GLPhotorealisticRender::updateImpl();

		auto& instances = this->inInstances()->constData();

		for (int i = 0; i < instances.size(); i++)
			instances[i]->update();
	}

	void GLInstancePhotorealisticRender::paintGL(const RenderParams& rparams)
	{	
		struct {
			glm::vec3 color;
			float metallic;
			float roughness;
			float alpha;
		} pbr;

		mShaderProgram->use();

		// setup uniforms
		if (mNormal.count() > 0
			&& mTangent.count() > 0
			&& mBitangent.count() > 0
			&& mNormal.count() == mTangent.count()
			&& mNormal.count() == mBitangent.count())
		{
			mShaderProgram->setInt("uVertexNormal", 1);
			mNormal.bindBufferBase(9);
			mTangent.bindBufferBase(12);
			mBitangent.bindBufferBase(13);
		}
		else
			mShaderProgram->setInt("uVertexNormal", 0);

		mShaderProgram->setInt("uInstanced", 1);

		mRenderParamsUBlock.load((void*)&rparams, sizeof(RenderParams));
		mRenderParamsUBlock.bindBufferBase(0);

		mPosition.bindBufferBase(8);
		mTexCoord.bindBufferBase(10);


		auto& instances = this->inInstances()->constData();
		auto& shapes = this->inShapes()->constData();
		for (int i = 0; i < shapes.size(); i++)
		{
			auto shape = shapes[i];
			auto mtl = shape->material;

			// material 
			{
				pbr.color = { mtl->diffuse.x, mtl->diffuse.y, mtl->diffuse.z };
				pbr.metallic = this->varMetallic()->getValue();
				pbr.roughness = this->varRoughness()->getValue();
				pbr.alpha = this->varAlpha()->getValue();
				mPBRMaterialUBlock.load((void*)&pbr, sizeof(pbr));
				mPBRMaterialUBlock.bindBufferBase(1);
			}

			// bind textures 
			{
				// reset 
				glActiveTexture(GL_TEXTURE10);		// color
				glBindTexture(GL_TEXTURE_2D, 0);
				glActiveTexture(GL_TEXTURE11);		// bump map
				glBindTexture(GL_TEXTURE_2D, 0);

				if (mtl->mColorTexture.isValid()) {
					mShaderProgram->setInt("uColorMode", 2);
					mtl->mColorTexture.bind(GL_TEXTURE10);
				}
				else {
					mShaderProgram->setInt("uColorMode", 0);
				}

				if (mtl->mBumpTexture.isValid()) {
					mtl->mBumpTexture.bind(GL_TEXTURE11);
					mShaderProgram->setFloat("uBumpScale", mtl->bumpScale);
				}
			}

			int numTriangles = shape->glVertexIndex.count();

			mVAO.bind();

			// setup VAO binding...
			{
				// vertex index
				shape->glVertexIndex.bind();
				glEnableVertexAttribArray(0);
				glVertexAttribIPointer(0, 1, GL_INT, sizeof(int), (void*)0);

				if (shape->glNormalIndex.count() == numTriangles) {
					shape->glNormalIndex.bind();
					glEnableVertexAttribArray(1);
					glVertexAttribIPointer(1, 1, GL_INT, sizeof(int), (void*)0);
				}
				else
				{
					glDisableVertexAttribArray(1);
					glVertexAttribI4i(1, -1, -1, -1, -1);
				}

				if (shape->glTexCoordIndex.count() == numTriangles) {
					shape->glTexCoordIndex.bind();
					glEnableVertexAttribArray(2);
					glVertexAttribIPointer(2, 1, GL_INT, sizeof(int), (void*)0);
				}
				else
				{
					glDisableVertexAttribArray(2);
					glVertexAttribI4i(2, -1, -1, -1, -1);
				}

			}

			if (!this->inInstances()->isEmpty())
			{
				instances[i]->gltransform.updateGL();
				// bind the translation vector
				mVAO.bindVertexBuffer(&instances[i]->gltransform, 3, 3, GL_FLOAT, sizeof(Transform3f), 0, 1);
				// bind the scale vector
				mVAO.bindVertexBuffer(&instances[i]->gltransform, 4, 3, GL_FLOAT, sizeof(Transform3f), sizeof(Vec3f), 1);
				// bind the rotation matrix
				mVAO.bindVertexBuffer(&instances[i]->gltransform, 5, 3, GL_FLOAT, sizeof(Transform3f), 2 * sizeof(Vec3f), 1);
				mVAO.bindVertexBuffer(&instances[i]->gltransform, 6, 3, GL_FLOAT, sizeof(Transform3f), 3 * sizeof(Vec3f), 1);
				mVAO.bindVertexBuffer(&instances[i]->gltransform, 7, 3, GL_FLOAT, sizeof(Transform3f), 4 * sizeof(Vec3f), 1);
				mVAO.bind();
				glDrawArraysInstanced(GL_TRIANGLES, 0, numTriangles * 3, instances[i]->instanceCount);
			}
			else
				glDrawArrays(GL_TRIANGLES, 0, numTriangles * 3);

			mVAO.unbind();

		}
	}


	
	
}