#pragma once
#include "Node.h"
#include "Topology/TriangleSet.h"
#include "Field.h"
#include "Field/FilePath.h"
#include "GLPointVisualModule.h"

namespace dyno
{
	template <typename TDataType> class TriangleSet;
	/*!
	*	\class	ObjLoader
	*	\brief	A node containing a TriangleSet object
	*
	*	This class is typically used as a representation for a static boundary mesh or base class to other classes
	*
	*/
	template<typename TDataType>
	class ObjPoint : public Node
	{
		DECLARE_TCLASS(ObjPoint, TDataType)
	public:

		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename ::dyno::Quat<Real> TQuat;

		ObjPoint();
		std::string getNodeType() override { return "IO"; }
		//void update() override;

	public:
		DEF_VAR(Vec3f, Location, 0, "Node location");
		DEF_VAR(Vec3f, Rotation, 0, "Node rotation");
		DEF_VAR(Vec3f, Scale, Vec3f(1.0f), "Node scale");

		DEF_VAR(FilePath, FileName, "", "");
		//DEF_VAR(std::string, InputPath, "", "");
		DEF_VAR(Real, Radius, 1, "Point Radius");
		DEF_INSTANCE_OUT(PointSet<TDataType>, PointSet, "");

		DEF_VAR(bool, Sequence, false, "Import Sequence");

		DEF_VAR(Coord, Center, Coord(0), "");
		DEF_VAR(Coord, Velocity, Coord(0), "");
		DEF_VAR(Coord, AngularVelocity, Coord(0), "");
		DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");
		//DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

	protected:
		void resetStates() override;
		void updateStates() override;
		void loadObj(PointSet<TDataType>& Pointset,std::string filename);
		void convertData(TriangleSet<TDataType>& Triangleset, std::string filename,PointSet<TDataType>& PointSet);

	private:
		GLPointVisualModule* pointrender;
		Quat<Real> rotQuat = Quat<Real>();
		Matrix rotMat;

		DArray<Coord> initPos;

		Coord center;
		Coord centerInit;
		Real PI = 3.1415926535;
	};
}