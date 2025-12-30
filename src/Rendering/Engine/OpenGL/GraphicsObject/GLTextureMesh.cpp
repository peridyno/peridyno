#include "GLTextureMesh.h"

#include "glad/glad.h"

namespace dyno
{
	/**
	 * GLMaterial
	 */
	GLMaterial::GLMaterial()
	{
	}

	GLMaterial::~GLMaterial()
	{
		release();
	}

	void GLMaterial::create()
	{
	}

	void GLMaterial::release()
	{
		texColor.release();
		texBump.release();
		texORM.release();
		texAlpha.release();
		texEmissiveColor.release();
	}

	void GLMaterial::updateGL()
	{
		texColor.updateGL();
		texBump.updateGL();
		texORM.updateGL();
		texAlpha.updateGL();
		texEmissiveColor.updateGL();

	}

	/**
	 * GLShape
	 */
	GLShape::GLShape()
	{
	}

	GLShape::~GLShape()
	{
		release();
	}

	void GLShape::create()
	{
		if (!mInitialized)
		{
			glVertexIndex.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
			glNormalIndex.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
			glTexCoordIndex.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

			mInitialized = true;
		}
	}

	void GLShape::release()
	{
		glVertexIndex.release();
		glNormalIndex.release();
		glTexCoordIndex.release();
	}

	void GLShape::updateGL()
	{
		if (!mInitialized)
			create();

		glVertexIndex.updateGL();
		glNormalIndex.updateGL();
		glTexCoordIndex.updateGL();
		if (this->material != NULL)
			this->material->updateGL();

	}


	GLTextureMesh::GLTextureMesh()
	{
	}

	GLTextureMesh::~GLTextureMesh()
	{
	}

	void GLTextureMesh::create()
	{
		if (!mInitialized)
		{
			mVertices.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
			mNormal.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
			mTexCoord.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);

			mInitialized = true;
		}
	}

	void GLTextureMesh::release()
	{
		mVertices.release();
		mNormal.release();
		mTexCoord.release();

		for (auto s : mShapes) {
			s->release();
		}

		mShapes.clear();
	}

#ifdef CUDA_BACKEND
	void GLTextureMesh::load(const std::shared_ptr<TextureMesh> mesh)
	{
		if (mesh == nullptr)
			return;

		mVertices.load(mesh->meshDataPtr()->vertices());
		mNormal.load(mesh->meshDataPtr()->normals());
		mTexCoord.load(mesh->meshDataPtr()->texCoords());

		uint shapeNum = mesh->shapes().size();

		if (mShapes.size() != shapeNum)
		{
			mShapes.resize(shapeNum);
			for (uint i = 0; i < shapeNum; i++)
			{
				mShapes[i] = std::make_shared<GLShape>();
			}
		}
	
		for (uint i = 0; i < shapeNum; i++)
		{
			mShapes[i]->glVertexIndex.load(mesh->shapes()[i]->vertexIndex);
			mShapes[i]->glNormalIndex.load(mesh->shapes()[i]->normalIndex);
			mShapes[i]->glTexCoordIndex.load(mesh->shapes()[i]->texCoordIndex);
	
			Vec3f S =  mesh->shapes()[i]->boundingTransform.scale();
			Mat3f R = mesh->shapes()[i]->boundingTransform.rotation();
			Vec3f T = mesh->shapes()[i]->boundingTransform.translation();

			Mat3f RS = R * Mat3f(
				S[0], 0, 0,
				0, S[1], 0,
				0, 0, S[2]);

			glm::mat4 tm = glm::mat4{
				RS(0, 0), RS(1, 0), RS(2, 0), 0,
				RS(0, 1), RS(1, 1), RS(2, 1), 0,
				RS(0, 2), RS(1, 2), RS(2, 2), 0,
				T[0],	  T[1],	    T[2],	  1 };

			mShapes[i]->transform = tm;

			//Setup the material for each shape
			if (mesh->shapes()[i]->material != NULL) 
			{
				std::shared_ptr<GLMaterial> currentShapeMtl = std::make_shared<GLMaterial>();
				currentShapeMtl->baseColor = Vec3f(mesh->shapes()[i]->material->baseColor.r, mesh->shapes()[i]->material->baseColor.g, mesh->shapes()[i]->material->baseColor.b);
				currentShapeMtl->roughness = mesh->shapes()[i]->material->roughness;
				currentShapeMtl->metallic = mesh->shapes()[i]->material->metallic;
				currentShapeMtl->bumpScale = mesh->shapes()[i]->material->bumpScale;
				currentShapeMtl->alpha = mesh->shapes()[i]->material->alpha;
				currentShapeMtl->texColor.load(mesh->shapes()[i]->material->texColor);
				currentShapeMtl->texBump.load(mesh->shapes()[i]->material->texBump);
				currentShapeMtl->texORM.load(mesh->shapes()[i]->material->texORM);
				currentShapeMtl->texAlpha.load(mesh->shapes()[i]->material->texAlpha);
				currentShapeMtl->texEmissiveColor.load(mesh->shapes()[i]->material->texEmissive);
				currentShapeMtl->emissiveIntensity = mesh->shapes()[i]->material->emissiveIntensity;

				if(mShapes[i]->material)
					mShapes[i]->material->release();
				mShapes[i]->material = currentShapeMtl;
			}
			else 
			{
				mShapes[i]->material = NULL;
			}		
		}
	}
#endif

	void GLTextureMesh::updateGL()
	{
		if (!mInitialized)
			create();

		mVertices.updateGL();
		mNormal.updateGL();
		mTexCoord.updateGL();

		for (uint i = 0; i < mShapes.size(); i++)
		{
			mShapes[i]->updateGL();
		}
	}

}