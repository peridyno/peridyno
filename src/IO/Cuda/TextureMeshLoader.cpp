#include "TextureMeshLoader.h"

#include "helpers/tinyobj_helper.h"

#include <GLPhotorealisticRender.h>

namespace dyno
{
	IMPLEMENT_CLASS(TextureMeshLoader)

	TextureMeshLoader::TextureMeshLoader()
	{
		this->stateTextureMesh()->setDataPtr(std::make_shared<TextureMesh>());

		auto callbackLoadFile = std::make_shared<FCallBackFunc>(std::bind(&TextureMeshLoader::callbackLoadFile, this));

		this->varFileName()->attach(callbackLoadFile);
		this->varUseInstanceTransform()->attach(callbackLoadFile);

		auto callbackTransform = std::make_shared<FCallBackFunc>(std::bind(&TextureMeshLoader::callbackTransform, this));
		this->varLocation()->attach(callbackTransform);
		this->varRotation()->attach(callbackTransform);
		this->varScale()->attach(callbackTransform);

		auto render = this->graphicsPipeline()->createModule<GLPhotorealisticRender>();
		this->stateTextureMesh()->connect(render->inTextureMesh());
		this->graphicsPipeline()->pushModule(render);

		this->stateTextureMesh()->promoteOuput();
	}

	TextureMeshLoader::~TextureMeshLoader()
	{
		mInitialVertex.clear();
		mInitialNormal.clear();
		mInitialTexCoord.clear();
		initialShapeCenter.clear();
	}

	void TextureMeshLoader::resetStates()
	{

	}

	void TextureMeshLoader::callbackLoadFile()
	{
		auto fullname = this->varFileName()->getValue();
		auto root = fullname.path().parent_path();

		auto texMesh = this->stateTextureMesh()->getDataPtr();
		
		bool success = loadTextureMeshFromObj(texMesh, fullname,this->varUseInstanceTransform()->getValue());
		if (!success)
			return;

		mInitialVertex.assign(texMesh->geometry()->vertices());
		mInitialNormal.assign(texMesh->geometry()->normals());
		mInitialTexCoord.assign(texMesh->geometry()->texCoords());

		auto& shape = this->stateTextureMesh()->getDataPtr()->shapes();
		initialShapeCenter.clear();
		for (size_t i = 0; i < shape.size(); i++)
		{
			initialShapeCenter.pushBack(shape[i]->boundingTransform.translation());
		}

		callbackTransform();
	}

	void TextureMeshLoader::callbackTransform()
	{
		auto& shape = this->stateTextureMesh()->getDataPtr()->shapes();
		auto vertices = this->stateTextureMesh()->getDataPtr()->geometry()->vertices();
		CArray<Vec3f> cv;
		cv.assign(vertices);

		if (varUseInstanceTransform()->getValue())
		{
			for (size_t i = 0; i < shape.size(); i++)
			{
				auto quat = this->computeQuaternion();
				shape[i]->boundingTransform.translation() = quat.rotate(initialShapeCenter[i]) * this->varScale()->getValue() + this->varLocation()->getValue();
				shape[i]->boundingTransform.rotation() = quat.toMatrix3x3();
				shape[i]->boundingTransform.scale() = this->varScale()->getValue();
			}
		}
		else
		{
			for (size_t i = 0; i < shape.size(); i++)
			{
				shape[i]->boundingTransform.translation() = Vec3f(0);
			}
		}
	}

}