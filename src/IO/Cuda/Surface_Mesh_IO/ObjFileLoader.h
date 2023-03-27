#pragma once
#include <string>
#include <vector>
#include "Vector.h"

namespace dyno {

	typedef Vector<float, 2> TexCoord;
	typedef VectorND<int, 3>	Face;
	typedef VectorND<int, 3>	TexIndex;

	class ObjFileLoader
	{
	public:
		ObjFileLoader(std::string filename);
		~ObjFileLoader() {}

		bool load(const std::string &filename);

		bool save(const std::string &filename);

		std::vector<Vec3f>& getVertexList();
		std::vector<Face>& getFaceList();

		std::vector<TexCoord>& getTexCoordList()
		{
			return texCoords;
		}

		std::vector<TexIndex>& getTexIndexList()
		{
			return texList;
		}

		bool hasTexCoords() {
			return texCoords.size() != 0;
		}

	private:
		std::vector<Vec3f> vertList;
		std::vector<Face> faceList;

		std::vector<TexCoord> texCoords;
		std::vector<TexIndex> texList;
	};

} //end of namespace dyno
