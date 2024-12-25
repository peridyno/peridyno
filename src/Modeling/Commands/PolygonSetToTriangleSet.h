#pragma once
#include "Node.h"
#include "Module/TopologyMapping.h"

#include "Topology/PolygonSet.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class PolygonSetToTriangleSetModule : public TopologyMapping
	{
		DECLARE_TCLASS(PolygonSetToTriangleSetModule, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename Transform<Real, 3> Transform;

		PolygonSetToTriangleSetModule()
		{
			this->outTriangleSet()->setDataPtr(std::make_shared<TriangleSet<DataType3f>>());
			if (this->outPolygon2Triangles()->isEmpty())
			{
				this->outPolygon2Triangles()->allocate();
			}
		};
		~PolygonSetToTriangleSetModule() {} ;

		std::string caption() override { return "PolygonSetToTriangleSetModule"; }

		DEF_INSTANCE_IN(PolygonSet<TDataType>, PolygonSet, "");
 
 		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");

		DEF_ARRAYLIST_OUT(uint, Polygon2Triangles, DeviceType::GPU, "");

	public:

		void convert(std::shared_ptr<PolygonSet<TDataType>> polygonset, std::shared_ptr<TriangleSet<TDataType>> triset, DArrayList<uint>& poly2tri);

	protected:
		bool apply() override
		{
			auto& poly2tri = this->outPolygon2Triangles()->getData();
			convert(this->inPolygonSet()->constDataPtr(), this->outTriangleSet()->getDataPtr(), poly2tri);

			return true;
		};
	};
	IMPLEMENT_TCLASS(PolygonSetToTriangleSetModule, TDataType)



	template<typename TDataType>
	class PolygonSetToTriangleSetNode : public Node
	{
		DECLARE_TCLASS(PolygonSetToTriangleSetNode, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PolygonSetToTriangleSetNode();

		std::string caption() override { return "PolygonSetToTriangleSet"; }

		DEF_INSTANCE_IN(PolygonSet<TDataType>, PolygonSet, "");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		DEF_ARRAYLIST_STATE(uint,Polygon2Triangles,DeviceType::GPU,"");



	protected:
		void resetStates() override 
		{
			if (this->statePolygon2Triangles()->isEmpty())
			{
				this->statePolygon2Triangles()->allocate();
			}
			auto& poly2triId = this->statePolygon2Triangles()->getData();

			mPolygonSetToTriangleSetMoudle->convert(this->inPolygonSet()->constDataPtr(),this->stateTriangleSet()->getDataPtr(), poly2triId);
		};
		void updateStates() override {};

	private:
		std::shared_ptr<PolygonSetToTriangleSetModule<DataType3f>> mPolygonSetToTriangleSetMoudle;
	};

	IMPLEMENT_TCLASS(PolygonSetToTriangleSetNode, TDataType);
}