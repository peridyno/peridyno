#pragma once
//#include "Framework/Node.h"
#include "ParticleSystem/ParticleSystem.h"
//#include "PointRenderModule.h"
#include "Quat.h"

namespace dyno
{
	template <typename TDataType> class TetrahedronSet;
	/*!
	*	\class	ParticleSystem
	*	\brief	Position-based fluids.
	*
	*	This class implements a position-based fluid solver.
	*	Refer to Macklin and Muller's "Position Based Fluids" for details
	*
	*/
	template<typename TDataType>
	class TetSystem : public Node
	{
		DECLARE_CLASS_1(TetSystem, TDataType)
	public:

		bool self_update = true;
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		TetSystem(std::string name = "default");
		virtual ~TetSystem();

		//void loadTets(Coord lo, Coord hi, Real distance);
		//void loadTets(Coord center, Real r, Real distance);
		void loadTets(std::string filename);

		virtual bool translate(Coord t);
		virtual bool scale(Real s);

		void updateTopology() override;
		bool resetStatus() override;

		//		std::shared_ptr<PointRenderModule> getRenderModule();

				
		DEF_EMPTY_CURRENT_ARRAY(Position, Coord, DeviceType::GPU, "Tet's center of mass");

		/**
		 * @brief Particle velocity
		 */
		DEF_EMPTY_CURRENT_ARRAY(Velocity, Coord, DeviceType::GPU, "Tet velocity");

		/**
		 * @brief Particle force
		 */
		DEF_EMPTY_CURRENT_ARRAY(Force, Coord, DeviceType::GPU, "Force on each Tet");

		/**
		 * @brief Particle angular velocity
		 */
		DEF_EMPTY_CURRENT_ARRAY(AngularVelocity, Coord, DeviceType::GPU, "AngularVelocity of each Tet");

		/**
		 * @brief Particle mass
		 */
		DEF_EMPTY_CURRENT_ARRAY(Mass, Real, DeviceType::GPU, "");

		DEF_EMPTY_CURRENT_ARRAY(Torque, Coord, DeviceType::GPU, "");

		DEF_EMPTY_CURRENT_ARRAY(AngularMass, Matrix, DeviceType::GPU, "");

		DEF_EMPTY_CURRENT_ARRAY(Orientation, Matrix, DeviceType::GPU, "");

		Quat<Real> m_quaternion;
		
		std::shared_ptr<TetrahedronSet<TDataType>> derivedTopology() {
			return m_tethedrons;
		}

	public:
		void advance(Real dt) override;
		void setCenter(Coord position);

		bool initialize() override;
		//		virtual void setVisible(bool visible) override;

	protected:
		std::shared_ptr<TetrahedronSet<TDataType>> m_tethedrons;
		//		std::shared_ptr<PointRenderModule> m_pointsRender;
	};
}
