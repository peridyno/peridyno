#pragma once
#include "Node/ParametricModel.h"

#include "Quat.h"

namespace dyno
{
	/*!
	*	\class	RigidBody
	* 
	*	This class implements a simple rigid body.
	*
	*/
	template<typename TDataType>
	class RigidBody : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(RigidBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename dyno::Quat<Real> Quat;

		RigidBody();
		~RigidBody() override;

	public:
		DEF_VAR(Coord, Gravity, Coord(0.0, -9.8, 0.0), "Gravity");

	public:
		/**
		 * @brief Rigid body mass
		 */
		DEF_VAR_STATE(Real, Mass, Real(1), "Mass of the rigid body");

		/**
		 * @brief Rigid body center
		 */
		DEF_VAR_STATE(Coord, Center, Coord(0), "Center of the rigid body");

		/**
		 * @brief Rigid body velocity
		 */
		DEF_VAR_STATE(Coord, Velocity, Coord(0), "Velocity of rigid bodies");

		/**
		 * @brief Rigid body angular velocity
		 */
		DEF_VAR_STATE(Coord, AngularVelocity, Coord(0), "Angular velocity of of the rigid body");

		/**
		 * @brief Particle position
		 */
		DEF_VAR_STATE(Matrix, RotationMatrix, Matrix::identityMatrix(), "Rotation matrix of of the rigid body");

		DEF_VAR_STATE(Matrix, Inertia, Matrix::identityMatrix(), "Inertia matrix");

		DEF_VAR_STATE(Quat, Quaternion, Quat(0, 0, 0), "Quaternion");

		DEF_VAR_STATE(Matrix, InitialInertia, Matrix::identityMatrix(), "Initial inertia matrix");

	protected:
		void updateStates() override;
	};
}
