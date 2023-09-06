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
		this->setName("surface_renderer");

		this->inColor()->tagOptional(true);

		this->inNormal()->tagOptional(true);
		this->inTexCoord()->tagOptional(true);
		this->inShapes()->tagOptional(true);
		this->inMaterials()->tagOptional(true);
	}

	GLPhotorealisticRender::~GLPhotorealisticRender()
	{
// 		mIndexBuffer.release();
// 		mVertexBuffer.release();
// 		mNormalBuffer.release();
// 		mColorBuffer.release();
// 
// 		triangles.clear();
// 		vertices.clear();
// 		normals.clear();
// 		colors.clear();
	}

	std::string GLPhotorealisticRender::caption()
	{
		return "Photorealistic Render";
	}

	bool GLPhotorealisticRender::initializeGL()
	{
		// create vertex buffer and vertex array object
		mVAO.create();

		mVertexPosition.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
		mVertexColor.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
		mNormal.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
		mTexCoord.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);

		// create shader program
		mShaderProgram = gl::Program::createProgramSPIRV(
			SURFACE_VERT, sizeof(SURFACE_VERT),
			SURFACE_FRAG, sizeof(SURFACE_FRAG),
			SURFACE_GEOM, sizeof(SURFACE_GEOM));

		// create shader uniform buffer
		mRenderParamsUBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		mPBRMaterialUBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);

		auto& shapes = this->inShapes()->getData();
		for (int i = 0; i < shapes.size(); i++)
		{
			shapes[i]->create();
		}

		auto& materials = this->inMaterials()->getData();
		for (int i = 0; i < materials.size(); i++)
		{
			materials[i]->create();
		}

		return true;
	}

	void GLPhotorealisticRender::releaseGL()
	{
		mShaderProgram->release();
		delete mShaderProgram;
		mShaderProgram = 0;

		// vertex array object
		mVAO.release();

		// shader storage buffer
		mVertexColor.release();
		mVertexPosition.release();
		mNormal.release();
		mTexCoord.release();

		// release uniform block
		mRenderParamsUBlock.release();
		mPBRMaterialUBlock.release();
	}

	void GLPhotorealisticRender::updateGL()
	{
		auto& shapes = this->inShapes()->getData();
		auto& materials = this->inMaterials()->getData();

		for (int i = 0; i < shapes.size(); i++)
		{
			shapes[i]->updateGL();
		}

		for (int i = 0; i < materials.size(); i++)
		{
			materials[i]->updateGL();
		}

		// update shader storage buffer
		mVertexPosition.updateGL();

		// vertex color
		if (this->varColorMode()->getValue() == EColorMode::CM_Vertex) {
			mVertexColor.updateGL();
		}
		// vertex normal
		if(this->varUseVertexNormal()->getValue()) {
			mNormal.updateGL();
		}

		// texture coordinates
		if (mTexCoord.count() > 0) {
			mTexCoord.updateGL();
		}

		gl::glCheckError();
	}

	void GLPhotorealisticRender::updateImpl()
	{
		// update data
		auto& vertices = this->inVertex()->getData();
		auto& normals = this->inNormal()->getData();
		auto& texCoord = this->inTexCoord()->getData();

		mVertexPosition.load(vertices);
		mNormal.load(normals);
		mTexCoord.load(texCoord);

		auto& shapes = this->inShapes()->getData();
		auto& materials = this->inMaterials()->getData();

		for (int i = 0; i < shapes.size(); i++)
		{
			shapes[i]->update();
		}

		for (int i = 0; i < materials.size(); i++)
		{
			materials[i]->update();
		}
	}

	void GLPhotorealisticRender::paintGL(const RenderParams& rparams)
	{
		auto& shapes = this->inShapes()->getData();
		auto& materials = this->inMaterials()->getData();

		mShaderProgram->use();

		if (rparams.mode == GLRenderMode::COLOR) {
		}
		else if (rparams.mode == GLRenderMode::SHADOW) {
		}
		else if (rparams.mode == GLRenderMode::TRANSPARENCY) {
		}
		else {
			printf("GLSurfaceVisualModule: Unknown render mode!\n");
			return;
		}

		// bind vertex data
		mVertexPosition.bindBufferBase(8);
		mNormal.bindBufferBase(9);
		mTexCoord.bindBufferBase(10);
		mVertexColor.bindBufferBase(11);

		// setup uniforms
		struct {
			glm::vec3 color;
			float metallic;
			float roughness;
			float alpha;
		} pbr;

		auto color = this->varBaseColor()->getValue();
		pbr.color = { color.r, color.g, color.b };
		pbr.metallic = this->varMetallic()->getValue();
		pbr.roughness = this->varRoughness()->getValue();
		pbr.alpha = this->varAlpha()->getValue();

		mShaderProgram->setInt("uVertexNormal", this->varUseVertexNormal()->getValue());
		mShaderProgram->setInt("uColorMode", this->varColorMode()->getValue().currentKey());
		mShaderProgram->setInt("uInstanced", mInstanceCount > 0);

		// setup uniform buffer
		mRenderParamsUBlock.load((void*)&rparams, sizeof(RenderParams));
		mRenderParamsUBlock.bindBufferBase(0);

		mPBRMaterialUBlock.load((void*)&pbr, sizeof(pbr));
		mPBRMaterialUBlock.bindBufferBase(1);


		for (int i = 0; i < 1; i++)
		{
#ifdef CUDA_BACKEND
			// bind texture
			if (shapes[i]->material != nullptr && shapes[i]->material->mColorTexture.isValid()) {
				shapes[i]->material->mColorTexture.bind(GL_TEXTURE10);
			}
			else
#endif
			{
				glActiveTexture(GL_TEXTURE10);
				glBindTexture(GL_TEXTURE_2D, 0);
			}

			mVAO.bind();

			// vertex index
			uint num = shapes[i]->vertexIndex.size();
			{
				shapes[i]->glVertexIndex.bind();
				glEnableVertexAttribArray(0);
				glVertexAttribIPointer(0, 1, GL_INT, sizeof(int), (void*)0);
			}

			// normal
			{
				if (shapes[i]->glNormalIndex.count() == num) {
					shapes[i]->glNormalIndex.bind();
					glEnableVertexAttribArray(1);
					glVertexAttribIPointer(1, 1, GL_INT, sizeof(int), (void*)0);
				}
				else
				{
					glDisableVertexAttribArray(1);
					glVertexAttribI4i(1, -1, -1, -1, -1);
				}
			}

			// texcoord
			{
				if (shapes[i]->glTexCoordIndex.count() == num) {
					shapes[i]->glTexCoordIndex.bind();
					glEnableVertexAttribArray(2);
					glVertexAttribIPointer(2, 1, GL_INT, sizeof(int), (void*)0);
				}
				else
				{
					glDisableVertexAttribArray(2);
					glVertexAttribI4i(2, -1, -1, -1, -1);
				}
			}

			// instance transforms
			glDisableVertexAttribArray(3);
			glDisableVertexAttribArray(4);
			glDisableVertexAttribArray(5);
			glDisableVertexAttribArray(6);
			glDisableVertexAttribArray(7);
			glDisableVertexAttribArray(8);

			mVAO.unbind();

			mVAO.bind();

			if (mInstanceCount > 0)
				glDrawArraysInstanced(GL_TRIANGLES, 0, num * 3, mInstanceCount);
			else
				glDrawArrays(GL_TRIANGLES, 0, num * 3);

			gl::glCheckError();
			mVAO.unbind();
		}
	}
}