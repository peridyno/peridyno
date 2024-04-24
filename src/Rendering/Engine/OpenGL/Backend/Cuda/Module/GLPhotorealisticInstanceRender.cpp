#include "GLPhotorealisticInstanceRender.h"
#include "Utility.h"

#include <glad/glad.h>

#include "surface.vert.h"
#include "surface.frag.h"
#include "surface.geom.h"

namespace dyno
{
	IMPLEMENT_CLASS(GLPhotorealisticInstanceRender)

	GLPhotorealisticInstanceRender::GLPhotorealisticInstanceRender()
		: GLPhotorealisticRender()
	{
	}

	GLPhotorealisticInstanceRender::~GLPhotorealisticInstanceRender()
	{
	
	}

	std::string GLPhotorealisticInstanceRender::caption()
	{
		return "Photorealistic Instance Render";
	}

	bool GLPhotorealisticInstanceRender::initializeGL()
	{
		mXTransformBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		return GLPhotorealisticRender::initializeGL();
	}

	void GLPhotorealisticInstanceRender::releaseGL()
	{
		mXTransformBuffer.release();
		GLPhotorealisticRender::releaseGL();
	}

	void GLPhotorealisticInstanceRender::updateGL()
	{
		mXTransformBuffer.updateGL();

		GLPhotorealisticRender::updateGL();
	}


	void GLPhotorealisticInstanceRender::updateImpl()
	{
		auto inst = this->inTransform()->constDataPtr();

		if (this->inTransform()->isModified())
		{
			auto texMesh = this->inTextureMesh()->constDataPtr();

			mOffset.assign(inst->index());
			mLists.assign(inst->lists());

			mXTransformBuffer.load(inst->elements());
		}

		if (this->inTextureMesh()->isModified())
			GLPhotorealisticRender::updateImpl();
	}

	void GLPhotorealisticInstanceRender::paintGL(const RenderParams& rparams)
	{	
		struct {
			glm::vec3 color;
			float metallic;
			float roughness;
			float alpha;
		} pbr;

		auto& vertices = mTextureMesh.vertices();
		auto& normals = mTextureMesh.normals();
		auto& texCoords = mTextureMesh.texCoords();

		mShaderProgram->use();

		// setup uniforms
		if (normals.count() > 0
			&& mTangent.count() > 0
			&& mBitangent.count() > 0
			&& normals.count() == mTangent.count()
			&& normals.count() == mBitangent.count())
		{
			mShaderProgram->setInt("uVertexNormal", 1);
			normals.bindBufferBase(9);
			mTangent.bindBufferBase(12);
			mBitangent.bindBufferBase(13);
		}
		else
			mShaderProgram->setInt("uVertexNormal", 0);

		mShaderProgram->setInt("uInstanced", 1);

		//Reset the model transform
		RenderParams rp = rparams;
		rp.transforms.model = glm::mat4{ 1.0 };
		mRenderParamsUBlock.load((void*)&rp, sizeof(RenderParams));
		mRenderParamsUBlock.bindBufferBase(0);

		vertices.bindBufferBase(8);
		texCoords.bindBufferBase(10);

		auto& shapes = mTextureMesh.shapes();
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

				if (mtl->texColor.isValid()) {
					mShaderProgram->setInt("uColorMode", 2);
					mtl->texColor.bind(GL_TEXTURE10);
				}
				else {
					mShaderProgram->setInt("uColorMode", 0);
				}

				if (mtl->texBump.isValid()) {
					mtl->texBump.bind(GL_TEXTURE11);
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

			uint offset_i = sizeof(Transform3f) * mOffset[i];
			mVAO.bindVertexBuffer(&mXTransformBuffer, 3, 3, GL_FLOAT, sizeof(Transform3f), offset_i + 0, 1);
			// bind the scale vector
			mVAO.bindVertexBuffer(&mXTransformBuffer, 4, 3, GL_FLOAT, sizeof(Transform3f), offset_i + sizeof(Vec3f), 1);
			// bind the rotation matrix
			mVAO.bindVertexBuffer(&mXTransformBuffer, 5, 3, GL_FLOAT, sizeof(Transform3f), offset_i + 2 * sizeof(Vec3f), 1);
			mVAO.bindVertexBuffer(&mXTransformBuffer, 6, 3, GL_FLOAT, sizeof(Transform3f), offset_i + 3 * sizeof(Vec3f), 1);
			mVAO.bindVertexBuffer(&mXTransformBuffer, 7, 3, GL_FLOAT, sizeof(Transform3f), offset_i + 4 * sizeof(Vec3f), 1);
			mVAO.bind();
			glDrawArraysInstanced(GL_TRIANGLES, 0, numTriangles * 3, mLists[i].size());

			mVAO.unbind();

		}
	}
}