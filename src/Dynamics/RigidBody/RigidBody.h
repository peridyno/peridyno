#pragma once
#include "Node.h"

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
		DECLARE_TCLASS(RigidBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		RigidBody(std::string name = "default");
		virtual ~RigidBody();
	};
}