#include "CapsuleModel.h"

#include "Primitive/Primitive3D.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"

#include "Mapping/Extract.h"

struct Index2D
{
	Index2D() : x(), y() {}
	Index2D(int x, int y) : x(x), y(y) {}
	int x, y;
};

bool operator<(const Index2D& lhs, const Index2D& rhs)
{
	return lhs.x != rhs.x ? lhs.x < rhs.x : lhs.y < rhs.y;
}

namespace dyno
{
	template<typename TDataType>
	CapsuleModel<TDataType>::CapsuleModel()
		: BasicShape<TDataType>()
	{
		this->varRadius()->setRange(0.001f, 100.0f);

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		this->statePolygonSet()->setDataPtr(std::make_shared<PolygonSet<TDataType>>());
		
		this->varLatitude()->setRange(2, 10000);
		this->varLongitude()->setRange(3, 10000);
		this->varHeight()->setRange(0, 100);
		this->varHeightSegment()->setRange(1, 10000);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&CapsuleModel<TDataType>::varChanged, this));


		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);
		this->varRadius()->attach(callback);
		this->varLatitude()->attach(callback);
		this->varLongitude()->attach(callback);

		auto tsRender = std::make_shared<GLSurfaceVisualModule>();
		tsRender->setAlpha(0.5f);
		tsRender->setColor(Color(0.8f, 0.52f, 0.25f));
		tsRender->setVisible(true);
		this->stateTriangleSet()->connect(tsRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(tsRender);

		auto exES = std::make_shared<ExtractEdgeSetFromPolygonSet<TDataType>>();
		this->statePolygonSet()->connect(exES->inPolygonSet());
		this->graphicsPipeline()->pushModule(exES);

		auto esRender = std::make_shared<GLWireframeVisualModule>();
		esRender->varBaseColor()->setValue(Color(0, 0, 0));
		exES->outEdgeSet()->connect(esRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(esRender);

		this->stateTriangleSet()->promoteOuput();
		this->statePolygonSet()->promoteOuput();

	}

	template<typename TDataType>
	void CapsuleModel<TDataType>::resetStates()
	{
		varChanged();
	}

	template<typename TDataType>
	NBoundingBox CapsuleModel<TDataType>::boundingBox()
	{
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		auto radius = this->varRadius()->getData();

		Coord length(radius);
		length[0] *= scale[0];
		length[1] *= scale[1];
		length[2] *= scale[2];

		Quat<Real> q = computeQuaternion();

		q.normalize();

		TOrientedBox3D<Real> box;
		box.center = center;
		box.u = q.rotate(Coord(1, 0, 0));
		box.v = q.rotate(Coord(0, 1, 0));
		box.w = q.rotate(Coord(0, 0, 1));
		box.extent = length;

		auto AABB = box.aabb();
		auto& v0 = AABB.v0;
		auto& v1 = AABB.v1;
		

		NBoundingBox ret;
		ret.lower = Vec3f(v0.x, v0.y, v0.z);
		ret.upper = Vec3f(v1.x, v1.y, v1.z);

		return ret;
	}




	template<typename TDataType>
	void CapsuleModel<TDataType>::varChanged()
	{
		auto center = this->varLocation()->getValue();
		auto rot = this->varRotation()->getValue();
		auto scale = this->varScale()->getValue();

		auto radius = this->varRadius()->getValue();
		auto latitude = this->varLatitude()->getValue();
		auto longitude = this->varLongitude()->getValue();
		Real halfHeight = this->varHeight()->getValue() / 2;
		auto row = this->varHeightSegment()->getValue();

		Quat<Real> q = computeQuaternion();

		//Setup a capsule primitive
		TCapsule3D<Real> capsulePrim = TCapsule3D<Real>(center, q, radius, halfHeight);

		this->outCapsule()->setValue(capsulePrim);

		auto polySet = this->statePolygonSet()->getDataPtr();

		std::vector<Coord> vertices;
		std::map<Index2D, uint> index_ID;
		std::map<Index2D, Coord> index_Coord;

		{
			Real deltaTheta = M_PI / latitude;
			Real deltaPhi = 2 * M_PI / longitude;

			//Setup vertices
			Real theta = 0;

			vertices.push_back(Coord(0, radius + halfHeight, 0));

			for (uint i = 0; i < (latitude) / 2; i++)
			{
				theta += deltaTheta;
				
				Real tempheight = i <= (latitude) / 2 ? halfHeight : - halfHeight;			

				Real phi = 0;
				for (uint j = 0; j < longitude; j++)
				{
					phi += deltaPhi;

					Real y = radius * std::cos(theta) + tempheight;
					Real x = (std::sin(theta) * radius) * std::sin(phi);
					Real z = (std::sin(theta) * radius) * std::cos(phi);

					vertices.push_back(Coord(x, y, z));

					if (i == latitude / 2 - 1)
					{
						Index2D RC = Index2D(row,j);
						index_ID[RC] = vertices.size() - 1;

						index_Coord[RC] = Coord(x, y, z);
					}
				}

			}

			uint halfVtNum = vertices.size();
			for (size_t i = 0; i < halfVtNum; i++)
			{
				vertices.push_back(Coord(vertices[i][0], - vertices[i][1], vertices[i][2]));
			}

			for (auto list : index_ID) 
			{
				auto f = list.first;
				auto s = list.second;
				Index2D RC = Index2D(0, f.y);
				index_ID[RC] = s + halfVtNum;
			}

			theta = 0;

			for (int i = 0; i <= row ; i++)
			{
				Real tempy = halfHeight * 2 / row * i - halfHeight;
				theta += deltaTheta;

				
				if (i != 0 && i != row )
				{
					Real phi = deltaPhi * longitude;

					for (int j = 0; j < longitude; j++) {
						Index2D RC = Index2D(i, j);

						Coord v = index_Coord.find(Index2D(row,j))->second;
						v = Coord(v[0], tempy,v[2]);

						vertices.push_back(v);
				
						index_ID[RC] = vertices.size() - 1;
					}
				}		
			}


			//Setup polygon indices
			uint numOfPolygon = 0;

			if (latitude % 2) 
				numOfPolygon = latitude * longitude + row * longitude - longitude;
			else
				numOfPolygon = latitude * longitude + row * longitude;


			CArray<uint> counter(numOfPolygon);


			uint incre = 0;
			
			//up
			for (uint j = 0; j < longitude; j++)
			{
				counter[incre] = 3;
				incre++;
			}

			for (uint i = 0; i < std::floor((latitude - 2)/2); i++)
			{
				for (uint j = 0; j < longitude; j++)
				{
					counter[incre] = 4;
					incre++;
				}
			}
			//down
			for (uint j = 0; j < longitude; j++)
			{
				counter[incre] = 3;
				incre++;
			}

			for (uint i = 0; i < std::floor((latitude - 2) / 2); i++)
			{
				for (uint j = 0; j < longitude; j++)
				{
					counter[incre] = 4;
					incre++;
				}
			}

			//side
			for (uint i = 0; i < row; i++)
			{
				for (uint j = 0; j < longitude; j++)
				{
					counter[incre] = 4;
					incre++;
				}
			}


			
			CArrayList<uint> polygonIndices;
			polygonIndices.resize(counter);

			incre = 0;
			uint offset = 1;


			for (uint j = 0; j < longitude; j++)
			{
				auto& index = polygonIndices[incre];
				index.insert(0);
				index.insert(offset + j);
				index.insert(offset + (j + 1) % longitude);

				incre++;
			}

			for (uint i = 0; i < (latitude - 2)/2; i++)
			{
				for (uint j = 0; j < longitude; j++)
				{
					auto& index = polygonIndices[incre];
					index.insert(offset + j);
					index.insert(offset + j + longitude);
					index.insert(offset + (j + 1) % longitude + longitude);
					index.insert(offset + (j + 1) % longitude);

					incre++;
				}
				offset += longitude;
			}

			for (size_t i = 0; i < incre; i++)
			{
				auto& source = polygonIndices[i];
				auto& index = polygonIndices[i + incre];

				for (size_t j = 0; j < source.size(); j++) 
				{	
					uint N = source.size() - 1;
					index.insert(source[N - j] + halfVtNum);

				}

			}

			//side
			incre = incre * 2;
			for (size_t i = 0; i < row ; i++)
			{
				for (size_t j = 0; j < longitude; j++)
				{
					auto& index = polygonIndices[incre];
					index.insert(index_ID.find(Index2D(i, (j + 1) % longitude))->second);
					index.insert(index_ID.find(Index2D(i + 1, (j + 1) % longitude))->second);
					index.insert(index_ID.find(Index2D(i + 1, j))->second);
					index.insert(index_ID.find(Index2D(i, j))->second);

					incre++;
				}
			}


			//Apply transformation
			Quat<Real> q = computeQuaternion();

			auto RV = [&](const Coord& v)->Coord {
				return center + q.rotate(v - center);
			};

			int numpt = vertices.size();

			for (int i = 0; i < numpt; i++) {
				vertices[i] = RV(vertices[i] * scale + RV(center));
			}

			polySet->setPoints(vertices);
			polySet->setPolygons(polygonIndices);
			polySet->update();

			polygonIndices.clear();
		}


		auto& ts = this->stateTriangleSet()->getData();
		polySet->turnIntoTriangleSet(ts);


		vertices.clear();
	}


	DEFINE_CLASS(CapsuleModel);
}