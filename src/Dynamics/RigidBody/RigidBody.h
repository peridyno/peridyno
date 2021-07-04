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
		FVar<Real> m_mass;
		FVar<Coord> m_center;
		FVar<Coord> m_transVelocity;
		FVar<Coord> m_angularVelocity;
		FVar<Coord> m_force;
		FVar<Coord> m_torque;
		FVar<Matrix> m_angularMass;
		FVar<Matrix> m_rotation;


		Quat<Real> m_quaternion;

		std::shared_ptr<Node> m_surfaceNode;
		std::shared_ptr<Node> m_collisionNode;

		std::shared_ptr<Frame<TDataType>> m_frame;
	};
}