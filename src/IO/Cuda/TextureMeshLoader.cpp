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
	}

	void TextureMeshLoader::resetStates()
	{

	}

	void TextureMeshLoader::callbackLoadFile()
	{
		auto fullname = this->varFileName()->getValue();
		auto root = fullname.path().parent_path();

		auto texMesh = this->stateTextureMesh()->getDataPtr();
		
		bool success = loadTextureMeshFromObj(texMesh, fullname);
		if (!success)
			return;

		mInitialVertex.assign(texMesh->vertices());
		mInitialNormal.assign(texMesh->normals());
		mInitialTexCoord.assign(texMesh->texCoords());

		//reset the transform
		this->varLocation()->setValue(Vec3f(0));
		this->varRotation()->setValue(Vec3f(0));
		this->varScale()->setValue(Vec3f(1));
	}

	void TextureMeshLoader::callbackTransform()
	{
#ifdef CUDA_BACKEND
		TriangleSet<DataType3f> ts;
#endif

#ifdef VK_BACKEND
		TriangleSet ps;
#endif
		ts.setPoints(mInitialVertex);
		ts.setNormals(mInitialNormal);

		// apply transform to vertices
		{
			auto t = this->varLocation()->getValue();
			auto q = this->computeQuaternion();
			auto s = this->varScale()->getValue();

#ifdef CUDA_BACKEND
			ts.scale(s);
			ts.rotate(q);
			ts.translate(t);
#endif
		}

		auto texMesh = this->stateTextureMesh()->getDataPtr();

		texMesh->vertices().assign(ts.getPoints());
		texMesh->normals().assign(ts.getVertexNormals());
		texMesh->texCoords().assign(mInitialTexCoord);

		ts.clear();
	}

}