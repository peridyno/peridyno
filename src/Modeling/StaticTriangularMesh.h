#pragma once
#include "Node.h"
#include "Topology/TriangleSet.h"

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
	class StaticTriangularMesh : public Node
	{
		DECLARE_TCLASS(StaticTriangularMesh, TDataType)
	public:

		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename Quat<Real> TQuat;

		StaticTriangularMesh();

		//void update() override;

	public:
		DEF_VAR(Vec3f, Location, 0, "Node location");
		DEF_VAR(Vec3f, Rotation, 0, "Node rotation");
		DEF_VAR(Vec3f, Scale, Vec3f(1.0f), "Node scale");

		DEF_VAR(FilePath, FileName, "", "");

		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");


		
		DEF_VAR(Coord, Velocity, Coord(0), "");
		DEF_VAR(Coord, Center, Coord(0), "");
		DEF_VAR(Coord, AngularVelocity, Coord(0), "");

		DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");
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
		

	};
}