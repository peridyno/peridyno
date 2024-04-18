#include "TextureMeshToTriangleSet.h"

namespace dyno
{
	IMPLEMENT_TCLASS(TextureMeshToTriangleSet, TDataType)

	template<typename TDataType>
	TextureMeshToTriangleSet<TDataType>::TextureMeshToTriangleSet()
		: TopologyMapping()
	{

	}

	template<typename TDataType>
	TextureMeshToTriangleSet<TDataType>::~TextureMeshToTriangleSet()
	{

	}

	template<typename TDataType>
	bool TextureMeshToTriangleSet<TDataType>::apply()
	{
		if (this->outTriangleSet()->isEmpty()) {
			this->outTriangleSet()->allocate();
		}

		auto mesh = this->inTextureMesh()->constDataPtr();

		uint indexNum = 0;
		for (uint i = 0; i < mesh->shapes().size(); i++) {
			indexNum += mesh->shapes()[i]->vertexIndex.size();
		}

		auto ts = this->outTriangleSet()->getDataPtr();

		auto& vertices = ts->getPoints();
		auto& indices = ts->getTriangles();

		vertices.assign(mesh->vertices());

		if (indices.size() != indexNum)
		{
			indices.resize(indexNum);
		}

		uint offset = 0;
		for (uint i = 0; i < mesh->shapes().size(); i++) {
			uint num = mesh->shapes()[i]->vertexIndex.size();
			indices.assign(mesh->shapes()[i]->vertexIndex, num, offset, 0);

			offset += num;
		}

		ts->update();

		return true;
	}

	DEFINE_CLASS(TextureMeshToTriangleSet);


	IMPLEMENT_TCLASS(TextureMeshToTriangleSetNode, TDataType);

	template<typename TDataType>
	TextureMeshToTriangleSetNode<TDataType>::TextureMeshToTriangleSetNode()
		: Node()
	{
		mTM2TS = std::make_shared<TextureMeshToTriangleSet<TDataType>>();

		this->inTextureMesh()->connect(mTM2TS->inTextureMesh());
		mTM2TS->outTriangleSet()->connect(this->outTriangleSet());
	}

	template<typename TDataType>
	void TextureMeshToTriangleSetNode<TDataType>::resetStates()
	{
		mTM2TS->update();
	}

	template<typename TDataType>
	void TextureMeshToTriangleSetNode<TDataType>::updateStates()
	{
		mTM2TS->update();
	}

	DEFINE_CLASS(TextureMeshToTriangleSetNode);
}