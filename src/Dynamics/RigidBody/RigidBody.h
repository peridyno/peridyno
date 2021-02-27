#pragma once
#include "Framework/Node.h"
#include "Quat.h"

namespace dyno
{
	template<typename TDataType> class Frame;
	/*!
	*	\class	RigidBody
	*	\brief	Rigid body dynamics.
	*
	*	This class implements a simple rigid body.
	*
	*/
	template<typename TDataType>
	class RigidBody : public Node
	{
		DECLARE_CLASS_1(RigidBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		RigidBody(std::string name = "default");
		virtual ~RigidBody();

		void loadShape(std::string filename);

		void advance(Real dt) override;

		void setMass(Real mass);
		void setCenter(Coord center);
		void setVelocity(Coord vel);

		void updateTopology() override;

		void translate(Coord t);
		void scale(Real t);

		std::shared_ptr<Node> getSurface() { return m_surfaceNode; }

	public:
		bool initialize() override;

	private:
		VarField<Real> m_mass;
		VarField<Coord> m_center;
		VarField<Coord> m_transVelocity;
		VarField<Coord> m_angularVelocity;
		VarField<Coord> m_force;
		VarField<Coord> m_torque;
		VarField<Matrix> m_angularMass;
		VarField<Matrix> m_rotation;


		Quat<Real> m_quaternion;

		std::shared_ptr<Node> m_surfaceNode;
		std::shared_ptr<Node> m_collisionNode;

		std::shared_ptr<Frame<TDataType>> m_frame;
	};

#ifdef PRECISION_FLOAT
	template class RigidBody<DataType3f>;
#else
	template class RigidBody<DataType3d>;
#endif
}