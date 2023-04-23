#pragma once
#include "Node/ParametricModel.h"

#include "Topology/TriangleSet.h"
#include "Field.h"
#include "FilePath.h"

namespace dyno
{
	template <typename TDataType> class TriangleSet;
	/*!
	*	\class	StaticTriangularMesh
	*	\brief	A node containing a TriangleSet object
	*
	*	This class is typically used as a representation for a static boundary mesh or base class to other classes
	*
	*/
	template<typename TDataType>
	class StaticTriangularMesh : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(StaticTriangularMesh, TDataType)
	public:

		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename ::dyno::Quat<Real> TQuat;

		StaticTriangularMesh();

		//void update() override;

	public:
		DEF_VAR(FilePath, FileName, "", "");
		//DEF_VAR(std::string, InputPath, "", "");

		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");
		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet_IN, "");
		DEF_VAR(bool, Sequence, false, "Import Sequence");

		DEF_VAR(Coord, Velocity, Coord(0), "");
		DEF_VAR(Coord, Center, Coord(0), "");
		DEF_VAR(Coord, AngularVelocity, Coord(0), "");

		DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");

		DEF_VAR(bool, ConvertInput, false, "ConvertInput");
		//DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

	protected:
		void resetStates() override;
		void updateStates() override;

	private:
		

		Quat<Real> rotQuat = Quat<Real>();
		Matrix rotMat;

		DArray<Coord> initPos;

		Coord center;
		Coord centerInit;
		Real PI = 3.1415926535;
	};
}