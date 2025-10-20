#include "GLPhotorealisticRender.h"
#include "Utility.h"

#include <glad/glad.h>
#include "ShaderStruct.h"
#include "surface.vert.h"
#include "surface.frag.h"
#include "surface.geom.h"

namespace dyno
{
	IMPLEMENT_CLASS(GLPhotorealisticRender)

	GLPhotorealisticRender::GLPhotorealisticRender()
	{
		this->setName("ObjMeshRenderer");

		this->varMaterialIndex()->attach(std::make_shared<FCallBackFunc>(
			[=]() {
				uint idx = this->varMaterialIndex()->getValue();

				if (mTextureMesh.materials().size() > idx)
				{
					auto material = mTextureMesh.materials()[idx];

					this->varAlpha()->setValue(material->alpha);
					this->varMetallic()->setValue(material->metallic);
					this->varRoughness()->setValue(material->roughness);
					this->varBaseColor()->setValue(Color(material->baseColor.x, material->baseColor.y, material->baseColor.z));
				}
			}));

		this->varMetallic()->attach(
			std::make_shared<FCallBackFunc>(
				[=]() {
					uint idx = this->varMaterialIndex()->getValue();

					if (mTextureMesh.materials().size() > idx)
					{
						auto material = mTextureMesh.materials()[idx];

						material->metallic = this->varMetallic()->getValue();
					}
				}));

		this->varRoughness()->attach(
			std::make_shared<FCallBackFunc>(
				[=]() {
					uint idx = this->varMaterialIndex()->getValue();

					if (mTextureMesh.materials().size() > idx)
					{
						auto material = mTextureMesh.materials()[idx];

						material->roughness = this->varRoughness()->getValue();
					}
				}));

		this->varAlpha()->attach(
			std::make_shared<FCallBackFunc>(
				[=]() {
					uint idx = this->varMaterialIndex()->getValue();

					if (mTextureMesh.materials().size() > idx)
					{
						auto material = mTextureMesh.materials()[idx];

						material->roughness = this->varAlpha()->getValue();
					}
				}));

#ifdef CUDA_BACKEND
		mTangentSpaceConstructor = std::make_shared<ConstructTangentSpace>();
		this->inTextureMesh()->connect(mTangentSpaceConstructor->inTextureMesh());
#endif
	}

	GLPhotorealisticRender::~GLPhotorealisticRender()
	{
	}

	std::string GLPhotorealisticRender::caption()
	{
		return "Photorealistic Render";
	}

	bool GLPhotorealisticRender::initializeGL()
	{
		// create vertex buffer and vertex array object
		mVAO.create();		
		// create shader program
		mShaderProgram = Program::createProgramSPIRV(
			SURFACE_VERT, sizeof(SURFACE_VERT),
			SURFACE_FRAG, sizeof(SURFACE_FRAG),
			SURFACE_GEOM, sizeof(SURFACE_GEOM));
		// create shader uniform buffer
		mRenderParamsUBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		mPBRMaterialUBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);

#ifdef CUDA_BACKEND
		mTangent.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
		mBitangent.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
#endif

		mShapeTransform.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		return true;
	}

	void GLPhotorealisticRender::releaseGL()
	{
		mShaderProgram->release();
		delete mShaderProgram;

		mTangent.release();
		mBitangent.release();

		mRenderParamsUBlock.release();
		mPBRMaterialUBlock.release();

		mVAO.release();

		mShapeTransform.release();

		mTextureMesh.release();
	}

	void GLPhotorealisticRender::updateGL()
	{
		mTangent.updateGL();
		mBitangent.updateGL();

		mShapeTransform.updateGL();

		mTextureMesh.updateGL();

		glCheckError();
	}

	void GLPhotorealisticRender::updateImpl()
	{
		if (this->inTextureMesh()->isModified()) {
			mTextureMesh.load(this->inTextureMesh()->constDataPtr());
			this->varMaterialIndex()->setRange(0, mTextureMesh.materials().size() - 1);
		}

#ifdef CUDA_BACKEND
		auto texMesh = this->inTextureMesh()->constDataPtr();
		if (texMesh->materials().size() > 0)
		{
			mTangentSpaceConstructor->update();

			if (!mTangentSpaceConstructor->outTangent()->isEmpty())
			{
				mTangent.load(mTangentSpaceConstructor->outTangent()->constData());
				mBitangent.load(mTangentSpaceConstructor->outBitangent()->constData());
			}
		}
#endif
	}

	void GLPhotorealisticRender::paintGL(const RenderParams& rparams)
	{

		mShaderProgram->use();

		auto& vertices = mTextureMesh.vertices();
		auto& normals = mTextureMesh.normals();
		auto& texCoords = mTextureMesh.texCoords();

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
		
		mShaderProgram->setInt("uInstanced", 0);

		vertices.bindBufferBase(8);
		texCoords.bindBufferBase(10);

		auto& shapes = mTextureMesh.shapes();
		for (int i = 0; i < shapes.size(); i++)
		{
			auto shape = shapes[i];
			if (shape->material != nullptr) 
			{
				auto mtl = shape->material;

				// 
				{
					RenderParams pm_i = rparams;

					pm_i.transforms.model = shape->transform;

					mRenderParamsUBlock.load((void*)&pm_i, sizeof(RenderParams));
					mRenderParamsUBlock.bindBufferBase(0);
				}

				// material 
				{

					PBRMaterial pbr;
					auto color = this->varBaseColor()->getValue();

					pbr.color = { mtl->baseColor.x, mtl->baseColor.y, mtl->baseColor.z };
					pbr.metallic = mtl->metallic;
					pbr.roughness = mtl->roughness;
					pbr.alpha = mtl->alpha;

					if(mtl->texORM.isValid())
						pbr.useAOTex = 1;
					if (mtl->texORM.isValid())
					{
						pbr.useRoughnessTex = 1;
						pbr.useMetallicTex = 1;
					}

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
					glActiveTexture(GL_TEXTURE12);		// bump map
					glBindTexture(GL_TEXTURE_2D, 0);

					if (mtl->texColor.isValid()) {
						mShaderProgram->setInt("uColorMode", 2);
						mtl->texColor.bind(GL_TEXTURE10);
					}
					else {
						mtl->texColor.unbind();
						mShaderProgram->setInt("uColorMode", 0);
					}

					if (mtl->texBump.isValid()) {
						mtl->texBump.bind(GL_TEXTURE11);
						mShaderProgram->setFloat("uBumpScale", mtl->bumpScale);
					}
					if (mtl->texORM.isValid()) 
					{
						mtl->texORM.bind(GL_TEXTURE12);
					}
				}
			}
			else 
			{
				RenderParams pm_i = rparams;
				pm_i.transforms.model = shape->transform;

				// setup uniform buffer
				mRenderParamsUBlock.load((void*)&rparams, sizeof(RenderParams));
				mRenderParamsUBlock.bindBufferBase(0);

				// material 
				{
					PBRMaterial pbr;

					auto color = this->varBaseColor()->getValue();
					pbr.color = { color.r, color.g, color.b };
					pbr.metallic = this->varMetallic()->getValue();
					pbr.roughness = this->varRoughness()->getValue();
					pbr.alpha = this->varAlpha()->getValue();
					mPBRMaterialUBlock.load((void*)&pbr, sizeof(pbr));
					mPBRMaterialUBlock.bindBufferBase(1);
				}

				mShaderProgram->setInt("uColorMode", 0);
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

			glDrawArrays(GL_TRIANGLES, 0, numTriangles * 3);

			glCheckError();
			mVAO.unbind();
		}
	}


}