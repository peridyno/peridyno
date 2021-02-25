#include "TexturedShape.h"

#include "Surface_Mesh_IO/ObjFileLoader.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(TexturedShape, TDataType)

	template<typename TDataType>
	TexturedShape<TDataType>::TexturedShape(std::string name)
		: Node(name)
	{
		this->setTopologyModule(std::make_shared<TriangleSet<TDataType>>());
	}

	template<typename TDataType>
	TexturedShape<TDataType>::~TexturedShape()
	{

	}

	template<typename TDataType>
	void TexturedShape<TDataType>::loadFile(std::string filename)
	{
		ObjFileLoader objLoader(filename);

		auto topo = TypeInfo::cast<TriangleSet<TDataType>>(this->getTopologyModule());

		topo->setPoints(objLoader.getVertexList());
		topo->setTriangles(objLoader.getFaceList());

		if (objLoader.hasTexCoords())
		{
			this->currentTextureCoord()->setValue(objLoader.getTexCoordList());
			this->currentTextureIndex()->setValue(objLoader.getTexIndexList());
		}
	}


	template<typename TDataType>
	void TexturedShape<TDataType>::scale(Real s)
	{
		auto topo = TypeInfo::cast<TriangleSet<TDataType>>(this->getTopologyModule());

		topo->scale(s);
	}
}