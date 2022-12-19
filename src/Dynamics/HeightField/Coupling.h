#pragma once
#include <vector>
#include <iostream>
#include <string.h>

#include "CapillaryWave.h"
#include "OceanPatch.h"
#include "Ocean.h"
#include "RigidBody/RigidBodySystem.h"
#include "Boat.h"
#include "Topology/TriangleSet.h"
#include "../Core/Algorithm/Reduction.h"
namespace dyno
{
	template<typename TDataType>
	class Coupling : public Node
	{
		DECLARE_TCLASS(Coupling, TDataType)
	public:

		typedef typename TDataType::Matrix Matrix;
		typedef typename TDataType::Coord Coord;

		Coupling(std::string name = "");
		~Coupling();

		void animate(float dt);

		DEF_NODE_PORT(Boat<TDataType>, Boat, "Boat");
		DEF_NODE_PORT(Ocean<TDataType>, Ocean, "Ocean");
		
	protected:
		void resetStates() override;
		void updateStates() override;

	private:
		Vec2f m_origin;

		Vec3f m_prePos;

		std::string m_name;
		float m_heightShift;

		float m_eclipsedTime;

		//	float3* m_oceanCentroid;			//fft displacement at the center of boat

		float* m_forceX;					//forces at sample points
		float* m_forceY;
		float* m_forceZ;

		float* m_torqueX;					//torques at sample points
		float* m_torqueY;
		float* m_torqueZ;

		
		float* m_sample_heights;

		bool m_force_corrected;

		Vec3f m_force_corrector;
		Vec3f m_torque_corrector;

		float m_heightScale = 0.2f;

		Reduction<float>* m_reduce;
	};
	IMPLEMENT_TCLASS(Coupling, TDataType)
}
