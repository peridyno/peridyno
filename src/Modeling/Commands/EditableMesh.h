#pragma once
#include "Node.h"
#include "Module/TopologyMapping.h"

#include "Topology/PolygonSet.h"
#include "Topology/TriangleSet.h"

#include "PolygonSetToTriangleSet.h"

namespace dyno
{


	template<typename TDataType>
	class EditableMesh : public PolygonSetToTriangleSetNode<TDataType>
	{
		DECLARE_TCLASS(EditableMesh, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		EditableMesh() {};

		std::string caption() override { return "EditableMesh"; }

		DEF_INSTANCE_IN(PolygonSet<TDataType>, PolygonSet, "");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		DEF_ARRAYLIST_STATE(Vec3f,VertexNormal, DeviceType::GPU,"");
		
		DEF_ARRAYLIST_STATE(Vec3f,TriangleNormal, DeviceType::GPU, "");
		
		DEF_ARRAY_STATE(Vec3f, PolygonNormal, DeviceType::GPU, "");



	protected:
		void resetStates() override ;


	private:



	};

	IMPLEMENT_TCLASS(EditableMesh, TDataType);
}