#pragma once
#include "ModelEditing.h"
#include "Topology/TriangleSet.h"
#include "Field.h"
#include "Field/FilePath.h"

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
	class SplineConstraint : public ModelEditing<TDataType>
	{
		DECLARE_TCLASS(SplineConstraint, TDataType)
	public:

		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename ::dyno::Quat<Real> TQuat;

		SplineConstraint();

		//void update() override;
	public:

		DEF_INSTANCE_IN(PointSet<TDataType>, Spline, "");
		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "");

		//DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");

		DEF_VAR(Real, Velocity, 10, "");
		DEF_VAR(Real, Offest, 0, "");

		DEF_VAR(bool, Accelerate, false, "");

		DEF_VAR(Real, AcceleratedSpeed, 0, "");

		//DEF_VAR(Coord, AngularVelocity, Coord(0), "");

		DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");

	protected:
		void resetStates() override;
		void updateStates() override;
		void updateTransform();
		void SLerp(Quat<Real> a, Quat<Real> b, double t, Quat<Real>& out);
		void getQuatFromVector(Vec3f va,Vec3f vb,Quat<Real> &q);
		void UpdateCurrentVelocity();


	private:


		Quat<Real> rotQuat = Quat<Real>();
		Matrix rotMat;

		DArray<Coord> initPos;

		Coord center;
		Coord centerInit;
		int totalIndex = 0;
		float tempLength = 0;
		int currentIndex = 0;
		Quat<Real> tempQ;

		float CurrentVelocity = 0;

		//Real PI = 3.1415926535;
	};
}