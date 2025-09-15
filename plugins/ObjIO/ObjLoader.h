#pragma once
#include "Node.h"
#include "Topology/TriangleSet.h"
#include "Field.h"
#include "Field/FilePath.h"
#include "Node/ParametricModel.h"

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
	class ObjLoader : virtual public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(ObjLoader, TDataType)
	public:

		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename ::dyno::Quat<Real> TQuat;

		ObjLoader();

		std::string getNodeType() override { return "IO"; }

	public:
		DEF_VAR(FilePath, FileName, "", "");

		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");

		DEF_VAR(bool, Sequence, false, "Import Sequence");
		DEF_VAR(Coord, Velocity, Coord(0), "");
		DEF_VAR(Coord, Center, Coord(0), "");
		DEF_VAR(Coord, AngularVelocity, Coord(0), "");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, Topology, "Topology");

	public:

	protected:
		void resetStates() override;
		void updateStates() override;
		void loadObj(TriangleSet<TDataType>& Triangleset,std::string filename);

	private:
		void animationUpdate();

		Quat<Real> rotQuat = Quat<Real>();
		Matrix rotMat;

		DArray<Coord> initPos;

		Coord center;
		Coord centerInit;
		Real PI = 3.1415926535;
	};
}