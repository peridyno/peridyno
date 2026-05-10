#include "GLCarInstanceRender.h"
#include "Utility.h"

#include <glad/glad.h>

#include "car.geom.h"

#include "car.vert.h"
#include "car.frag.h"

#include "surface.frag.h"
#include "surface.geom.h"
#include "surface.vert.h"

#include "ShaderStruct.h"
#include "CarMaterial.h"

namespace dyno
{
	IMPLEMENT_CLASS(GLCarInstanceRender)

		GLCarInstanceRender::GLCarInstanceRender()
		: GLVisualModule()
	{
		this->setName("ObjMeshRenderer");

		this->inTransform()->tagOptional(true);
		this->inBrakeLight()->tagOptional(true);
		this->inHeadLight()->tagOptional(true);
		this->inTurnSignal()->tagOptional(true);
		this->inRightDirection()->tagOptional(true);

		this->varMaterialShapeIndex()->attach(std::make_shared<FCallBackFunc>(
			[=]() {
				uint idx = this->varMaterialShapeIndex()->getValue();

				if (idx < mTextureMesh.shapes().size())
				{
					auto material = mTextureMesh.shapes()[idx]->material;
					if (material)
					{
						this->varAlpha()->setValue(material->alpha);
						this->varMetallic()->setValue(material->metallic);
						this->varRoughness()->setValue(material->roughness);
						this->varBaseColor()->setValue(Color(material->baseColor.x, material->baseColor.y, material->baseColor.z));
					}
				}

			}));

		this->varMetallic()->attach(
			std::make_shared<FCallBackFunc>(
				[=]() {
					uint idx = this->varMaterialShapeIndex()->getValue();

					if (idx < mTextureMesh.shapes().size())
					{
						auto material = mTextureMesh.shapes()[idx]->material;
						if (material)
							material->metallic = this->varMetallic()->getValue();
					}
				}));

		this->varRoughness()->attach(
			std::make_shared<FCallBackFunc>(
				[=]() {
					uint idx = this->varMaterialShapeIndex()->getValue();

					if (idx < mTextureMesh.shapes().size())
					{
						auto material = mTextureMesh.shapes()[idx]->material;
						if (material)
							material->roughness = this->varRoughness()->getValue();
					}
				}));

		this->varAlpha()->attach(
			std::make_shared<FCallBackFunc>(
				[=]() {
					uint idx = this->varMaterialShapeIndex()->getValue();

					if (idx < mTextureMesh.shapes().size())
					{
						auto material = mTextureMesh.shapes()[idx]->material;
						if (material)
							material->roughness = this->varAlpha()->getValue();
					}
				}));

#ifdef CUDA_BACKEND
		mTangentSpaceConstructor = std::make_shared<ConstructTangentSpace>();
		this->inTextureMesh()->connect(mTangentSpaceConstructor->inTextureMesh());
#endif



	}

	GLCarInstanceRender::~GLCarInstanceRender()
	{
	
	}

	std::string GLCarInstanceRender::caption()
	{
		return "Car Instance Render";
	}

	bool GLCarInstanceRender::initializeGL()
	{
		mXTransformBuffer.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		mInstanceHeadlight.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mInstanceBrakeLight.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
		mInstanceTurnSignal.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		// create vertex buffer and vertex array object
		mVAO.create();
		// create shader program
		mShaderProgram = Program::createProgramSPIRV(
			CAR_VERT, sizeof(CAR_VERT),
			CAR_FRAG, sizeof(CAR_FRAG),
			CAR_GEOM, sizeof(CAR_GEOM));

		//// create shader program
		//mShaderProgram = Program::createProgramSPIRV(
		//	SURFACE_VERT, sizeof(SURFACE_VERT),
		//	SURFACE_FRAG, sizeof(SURFACE_FRAG),
		//	SURFACE_GEOM, sizeof(SURFACE_GEOM));
		// create shader uniform buffer
		mRenderParamsUBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		mPBRMaterialUBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		mLightControlUBlock.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);

#ifdef CUDA_BACKEND
		mTangent.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
		mBitangent.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
#endif

		mShapeTransform.create(GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

		return true;
	}

	void GLCarInstanceRender::releaseGL()
	{
		mXTransformBuffer.release();

		mInstanceHeadlight.release();
		mInstanceBrakeLight.release();
		mInstanceTurnSignal.release();

		mShaderProgram->release();
		delete mShaderProgram;

		mTangent.release();
		mBitangent.release();

		mRenderParamsUBlock.release();
		mPBRMaterialUBlock.release();
		mLightControlUBlock.release();

		mVAO.release();

		mShapeTransform.release();

		mTextureMesh.release();
	}

	void GLCarInstanceRender::updateGL()
	{
		if (mNeedUpdateInstanceTransform)
		{
			mXTransformBuffer.updateGL();
			mNeedUpdateInstanceTransform = false;
		}

		if (mNeedUpdateLight) 
		{
			mInstanceHeadlight.updateGL();
			mInstanceBrakeLight.updateGL();
			mInstanceTurnSignal.updateGL();
			mNeedUpdateLight = false;
		}

		if (mNeedUpdateTextureMesh)
		{
			mTangent.updateGL();
			mBitangent.updateGL();

			mShapeTransform.updateGL();

			mTextureMesh.updateGL();

			mNeedUpdateTextureMesh = false;
		}

		glCheckError();
	}


	void GLCarInstanceRender::updateImpl()
	{
		auto inst = this->inTransform()->constDataPtr();

		if (this->inTransform()->isModified())
		{
			auto texMesh = this->inTextureMesh()->constDataPtr();

			mOffset.assign(inst->index());
			mLists.assign(inst->lists());

			mXTransformBuffer.load(inst->elements());
			mNeedUpdateInstanceTransform = true;
		}

		if (this->inHeadLight()->isModified()|| this->inBrakeLight()->isModified()|| this->inTurnSignal()->isModified())
		{
			auto instHead = this->inHeadLight()->constDataPtr();
			auto instBrake = this->inBrakeLight()->constDataPtr();
			auto instTurn = this->inTurnSignal()->constDataPtr();
			if (instHead) 
			{
				mInstanceHeadlight.load(instHead->elements());
				CArrayList<float> cheadlight;
				cheadlight.assign(this->inHeadLight()->getData());
			}
			if(instBrake)
				mInstanceBrakeLight.load(instBrake->elements());
			if(instTurn)
				mInstanceTurnSignal.load(instTurn->elements());

			mNeedUpdateLight = true;
		}

		//
		if (this->inTextureMesh()->isModified()) {
			mTextureMesh.load(this->inTextureMesh()->constDataPtr());
			this->varMaterialShapeIndex()->setRange(0, mTextureMesh.shapes().size());
			mNeedUpdateTextureMesh = true;
		}

#ifdef CUDA_BACKEND
		auto texMesh = this->inTextureMesh()->constDataPtr();

		if (!texMesh->geometry()->normals().isEmpty() &&
			!texMesh->geometry()->texCoords().isEmpty())
		{
			mTangentSpaceConstructor->update();
		}
		if (!mTangentSpaceConstructor->outTangent()->isEmpty())
		{
			mTangent.load(mTangentSpaceConstructor->outTangent()->constData());
			mBitangent.load(mTangentSpaceConstructor->outBitangent()->constData());
		}

#endif
	}

	void GLCarInstanceRender::paintGL(const RenderParams& rparams)
	{


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
			if (mtl != nullptr)
			{
				// material 
				{

					PBRMaterial pbr;
					auto color = this->varBaseColor()->getValue();

					pbr.color = { mtl->baseColor.x, mtl->baseColor.y, mtl->baseColor.z };
					pbr.metallic = mtl->metallic;
					pbr.roughness = mtl->roughness;
					pbr.alpha = mtl->alpha;
					pbr.EmissiveIntensity = mtl->emissiveIntensity;

					if (mtl->texORM.isValid())
						pbr.useAOTex = 1;
					if (mtl->texORM.isValid())
					{
						pbr.useRoughnessTex = 1;
						pbr.useMetallicTex = 1;
					}
					else
					{
						pbr.useRoughnessTex = 0;
						pbr.useMetallicTex = 0;
					}
					if (mtl->texEmissiveColor.isValid())
						pbr.useEmissiveTex = 1;
					else
						pbr.useEmissiveTex = 0;

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
					glActiveTexture(GL_TEXTURE12);		// orm
					glBindTexture(GL_TEXTURE_2D, 0);

					glActiveTexture(GL_TEXTURE13);		// emissive
					glBindTexture(GL_TEXTURE_2D, 0);

					auto carMtl = std::dynamic_pointer_cast<GLCarMaterial>(mtl);
					if (carMtl) 
					{
						glActiveTexture(GL_TEXTURE14);		// CarlightMask   Channel: "R:HeadLightMask; G:BrakeLightMask; B:TurnLightMask"
						glBindTexture(GL_TEXTURE_2D, 0);

						if (carMtl->texLightMask.isValid())
						{
							carMtl->texLightMask.bind(GL_TEXTURE14);
						}

						{
							struct VehicleLightingControl
							{
								glm::vec3 rightDirection = glm::vec3(0);
								float useLightMask = 0.0f;
							};
							VehicleLightingControl lightControl;
							
							if (!this->inRightDirection()->isEmpty()) 
							{
								auto dir = this->inRightDirection()->getValue().normalize();
								lightControl.rightDirection = glm::vec3(dir.x,dir.y,dir.z);
							}

							if (carMtl->texLightMask.isValid())
							{
								lightControl.useLightMask = 1;
							}

							mLightControlUBlock.load((void*)&lightControl, sizeof(lightControl));
							mLightControlUBlock.bindBufferBase(2);

						}
					}


					if (mtl->texColor.isValid()) {
						mShaderProgram->setInt("uColorMode", 2);
						mtl->texColor.bind(GL_TEXTURE10);
					}
					else {
						mShaderProgram->setInt("uColorMode", 1);
					}

					if (mtl->texBump.isValid()) {
						mtl->texBump.bind(GL_TEXTURE11);
						mShaderProgram->setFloat("uBumpScale", mtl->bumpScale);
					}
					if (mtl->texORM.isValid())
					{
						mtl->texORM.bind(GL_TEXTURE12);
					}
					if (mtl->texEmissiveColor.isValid())
					{
						mtl->texEmissiveColor.bind(GL_TEXTURE13);
					}
				}
			}
			else
			{
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


				mShaderProgram->setInt("uColorMode", 1);
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
			if (mOffset.size() >= shapes.size())
			{

				uint offset_i = sizeof(Transform3f) * mOffset[i];
				mVAO.bindVertexBuffer(&mXTransformBuffer, 3, 3, GL_FLOAT, sizeof(Transform3f), offset_i + 0, 1);
				// bind the scale vector
				mVAO.bindVertexBuffer(&mXTransformBuffer, 4, 3, GL_FLOAT, sizeof(Transform3f), offset_i + sizeof(Vec3f), 1);
				// bind the rotation matrix
				mVAO.bindVertexBuffer(&mXTransformBuffer, 5, 3, GL_FLOAT, sizeof(Transform3f), offset_i + 2 * sizeof(Vec3f), 1);
				mVAO.bindVertexBuffer(&mXTransformBuffer, 6, 3, GL_FLOAT, sizeof(Transform3f), offset_i + 3 * sizeof(Vec3f), 1);
				mVAO.bindVertexBuffer(&mXTransformBuffer, 7, 3, GL_FLOAT, sizeof(Transform3f), offset_i + 4 * sizeof(Vec3f), 1);

				uint offset_Light = sizeof(float) * mOffset[i];

				if (!this->inHeadLight()->isEmpty()) {
					if (this->inHeadLight()->getDataPtr()->index().size() >= shapes.size())
					{
						mVAO.bindVertexBuffer(&mInstanceHeadlight, 8, 1, GL_FLOAT, sizeof(float), offset_Light, 1);
					}
					else {
						mVAO.bind();
						glDisableVertexAttribArray(8);
						glVertexAttrib1f(8, 0.0f);
						mVAO.unbind();
					}
				}
				else {
					mVAO.bind();
					glDisableVertexAttribArray(8);
					glVertexAttrib1f(8, 0.0f);
					mVAO.unbind();
				}
				if (!this->inBrakeLight()->isEmpty()) {
					if (this->inBrakeLight()->getDataPtr()->index().size() >= shapes.size()) {
						mVAO.bindVertexBuffer(&mInstanceBrakeLight, 9, 1, GL_FLOAT, sizeof(float), offset_Light, 1);
					}
					else {
						mVAO.bind();
						glDisableVertexAttribArray(9);
						glVertexAttrib1f(9, 0.0f);
						mVAO.unbind();
					}
				}
				else {
					mVAO.bind();
					glDisableVertexAttribArray(9);
					glVertexAttrib1f(9, 0.0f);
					mVAO.unbind();
				}
				if (!this->inTurnSignal()->isEmpty()) {
					if (this->inTurnSignal()->getDataPtr()->index().size() >= shapes.size()) {
						mVAO.bindVertexBuffer(&mInstanceTurnSignal, 10, 1, GL_FLOAT, sizeof(float), offset_Light, 1);
					}
					else {
						mVAO.bind();
						glDisableVertexAttribArray(10);
						glVertexAttrib1f(10, 0.0f);
						mVAO.unbind();
					}
				}
				else {
					mVAO.bind();
					glDisableVertexAttribArray(10);
					glVertexAttrib1f(10, 0.0f);
					mVAO.unbind();
				}


				mVAO.bind();
				glDrawArraysInstanced(GL_TRIANGLES, 0, numTriangles * 3, mLists[i].size());

			}
			else
			{
				printf("GLPhotorealisticInstanceRender::inTransform Is Error !!!!!!\n");
			}
			mVAO.unbind();

		}

	}
}