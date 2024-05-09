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
	}

	void GLMaterial::updateGL()
	{
		texColor.updateGL();
		texBump.updateGL();
	}

	/**
	 * GLShape
	 */
	GLShape::GLShape()
	{
	}

	GLShape::~GLShape()
	{
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

		for (auto m : mMaterials) {
			m->release();
		}

		for (auto s : mShapes) {
			s->release();
		}

		mMaterials.clear();
		mShapes.clear();
	}

	void GLTextureMesh::load(const std::shared_ptr<TextureMesh> mesh)
	{
		if (mesh == nullptr)
			return;

		mVertices.load(mesh->vertices());
		mNormal.load(mesh->normals());
		mTexCoord.load(mesh->texCoords());

		uint shapeNum = mesh->shapes().size();

		if (mShapes.size() != shapeNum)
		{
			mShapes.resize(shapeNum);
			for (uint i = 0; i < shapeNum; i++)
			{
				mShapes[i] = std::make_shared<GLShape>();
			}
		}

		uint matNum = mesh->materials().size();
		if (mMaterials.size() != matNum)
		{
			mMaterials.resize(matNum);
			for (uint i = 0; i < matNum; i++)
			{
				mMaterials[i] = std::make_shared<GLMaterial>();
			}
		}

		std::map<std::shared_ptr<Material>, uint> mapper;

		for (uint i = 0; i < matNum; i++)
		{
			mMaterials[i]->baseColor = mesh->materials()[i]->baseColor;
			mMaterials[i]->metallic = mesh->materials()[i]->metallic;
			mMaterials[i]->roughness = mesh->materials()[i]->roughness;
			mMaterials[i]->bumpScale = mesh->materials()[i]->bumpScale;
			mMaterials[i]->texColor.load(mesh->materials()[i]->texColor);
			mMaterials[i]->texBump.load(mesh->materials()[i]->texBump);

			mapper[mesh->materials()[i]] = i;
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
			auto test = mapper[mesh->shapes()[i]->material];
			auto testsm = mesh->shapes()[i]->material;

			if (mesh->shapes()[i]->material != NULL) 
			{
				mShapes[i]->material = mMaterials[mapper[mesh->shapes()[i]->material]];
			}
			else 
			{
				mShapes[i]->material = NULL;
			}
				
		}

		mapper.clear();
	}

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

		for (uint i = 0; i < mMaterials.size(); i++)
		{
			mMaterials[i]->updateGL();
		}
	}

}