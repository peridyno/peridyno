/*
This Module is designed to output surface mesh file of TetrahedronSet;
the output file format: obj
*/

#pragma once
#include "Module/OutputModule.h"
#include "Module/TopologyModule.h"

#include "Topology/TetrahedronSet.h"

#include <string>
#include <memory>

namespace dyno
{

	template <typename TDataType> class TriangleSet;
	template <typename TDataType> class TetrahedronSet;

	template<typename TDataType>
	class TetraMeshWriter : public OutputModule
	{
		DECLARE_TCLASS(TetraMeshWriter, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;
		typedef typename TopologyModule::Tetrahedron Tetrahedron;
		typedef typename TopologyModule::Tri2Tet Tri2Tet;

		TetraMeshWriter();
		~TetraMeshWriter() override;

		DEF_INSTANCE_IN(TetrahedronSet<TDataType>, TetrahedronSet, "");

	protected:
		void output()override;
	};
}