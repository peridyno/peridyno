#include "GLPhotorealisticRender.h"
#include "Utility.h"

#include <glad/glad.h>

#include "surface.vert.h"
#include "surface.frag.h"
#include "surface.geom.h"

namespace dyno
{
	IMPLEMENT_CLASS(GLPhotorealisticRender)

	GLPhotorealisticRender::GLPhotorealisticRender()
	{
		this->setName("ObjMeshRenderer");

		this->inNormal()->tagOptional(true);
		this->inTexCoord()->tagOptional(true);
		this->inMaterials()->tagOptional(true);
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
		mShaderProgram = gl::Program::createProgramSPIRV(
			SURFACE_VERT, sizeof(SURFACE_VERT),
			SURFACE_FRAG, sizeof(SURFACE_FRAG),
			SURFACE_GEOM, sizeof(SURFACE_GEOM));
		// create shader uniform buffer
		mRenderParamsUBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		mPBRMaterialUBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);

		mPosition.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
		mNormal.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
		mTexCoord.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);

		return true;
	}

	void GLPhotorealisticRender::releaseGL()
	{
		mPosition.release();
		mNormal.release();
		mTexCoord.release();
	}

	void GLPhotorealisticRender::updateGL()
	{
		auto& shapes = this->inShapes()->constData();
		auto& materials = this->inMaterials()->constData();

		for (int i = 0; i < shapes.size(); i++)
		{
			shapes[i]->create();
			shapes[i]->updateGL();
		}

		for (int i = 0; i < materials.size(); i++)
		{
			materials[i]->create();
			materials[i]->updateGL();
		}

		// update shader storage buffer
		mPosition.updateGL();
		mNormal.updateGL();

		// texture coordinates
		if (mTexCoord.count() > 0) {
			mTexCoord.updateGL();
		}

		gl::glCheckError();
	}

	void GLPhotorealisticRender::updateImpl()
	{
		// update data
		auto& vertices = this->inVertex()->constData();
		auto& normals  = this->inNormal()->constData();
		auto& texCoord = this->inTexCoord()->constData();

		mPosition.load(vertices);
		mNormal.load(normals);
		mTexCoord.load(texCoord);

		auto shapes = this->inShapes()->constData();
		auto materials = this->inMaterials()->constData();

		for (int i = 0; i < shapes.size(); i++)
			shapes[i]->update();

		for (int i = 0; i < materials.size(); i++)
			materials[i]->update();
	}

	void GLPhotorealisticRender::paintGL(const RenderParams& rparams)
	{
		struct {
			glm::vec3 color;
			float metallic;
			float roughness;
			float alpha;
		} pbr;

		mShaderProgram->use();

		// setup uniforms
		mShaderProgram->setInt("uVertexNormal", 1);
		mShaderProgram->setInt("uColorMode", 2);
		mShaderProgram->setInt("uInstanced", 0);

		mRenderParamsUBlock.load((void*)&rparams, sizeof(RenderParams));
		mRenderParamsUBlock.bindBufferBase(0);

		mPosition.bindBufferBase(8);
		mNormal.bindBufferBase(9);
		mTexCoord.bindBufferBase(10);

		auto shapes = this->inShapes()->constData();
		for (int i = 0; i < shapes.size(); i++)
		{
			auto shape = shapes[i];
			auto mtl   = shape->material;

			// material 
			{
				auto color = this->varBaseColor()->getValue();
				pbr.color = { color.r, color.g, color.b };
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

				if (mtl->mColorTexture.isValid()) 
					mtl->mColorTexture.bind(GL_TEXTURE10);
				if (mtl->mBumpTexture.isValid())  
					mtl->mBumpTexture.bind(GL_TEXTURE11);
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

			gl::glCheckError();
			mVAO.unbind();
		}
	}
}