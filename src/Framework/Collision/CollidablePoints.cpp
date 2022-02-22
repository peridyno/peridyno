#include "CollidablePoints.h"
#include "Module/DeviceContext.h"
#include "Node.h"
#include "Topology/PointSet.h"
#include "Mapping/FrameToPointSet.h"
#include "Mapping/PointSetToPointSet.h"

namespace dyno
{
	IMPLEMENT_TCLASS(CollidablePoints, TDataType)

	template<typename TDataType>
	CollidablePoints<TDataType>::CollidablePoints()
		: CollidableObject(CollidableObject::POINTSET_TYPE)
		, m_bUniformRaidus(true)
		, m_radius(Real(0.005))
	{
	}
	
	template<typename TDataType>
	CollidablePoints<TDataType>::~CollidablePoints()
	{
	}

	template<typename TDataType>
	void CollidablePoints<TDataType>::setPositions(DArray<Coord>& centers)
	{
	}

	template<typename TDataType>
	void CollidablePoints<TDataType>::setVelocities(DArray<Coord>& vel)
	{
	}

	template<typename TDataType>
	void CollidablePoints<TDataType>::setRadii(DArray<Coord>& radii)
	{

	}

	template<typename TDataType>
	void CollidablePoints<TDataType>::updateCollidableObject()
	{
/*		auto mstate = getParent()->getMechanicalState();
		auto mType = mstate->getMaterialType();
		if (mType == MechanicalState::ParticleSystem)
		{
			auto center = mstate->getField<HostVarField<Coord>>(MechanicalState::position())->getData();
			auto rotation = mstate->getField<HostVarField<Matrix>>(MechanicalState::rotation())->getData();

			auto pSet = TypeInfo::cast<PointSet<TDataType>>(getParent()->getTopologyModule());

			auto mp = std::dynamic_pointer_cast<FrameToPointSet<TDataType>>(m_mapping);

			Rigid tmp_rigid(center, Quat<Real>(rotation));
			mp->applyTransform(tmp_rigid, m_positions);
		}
		else
		{
			auto pBuf = mstate->getField<DeviceArrayField<Coord>>(MechanicalState::position());
			//std::shared_ptr<DeviceArrayField<Coord>> pBuf = TypeInfo::cast<DeviceArrayField<Coord>>(pos);

			auto vBuf = mstate->getField<DeviceArrayField<Coord>>(MechanicalState::velocity());
			//std::shared_ptr<DeviceArrayField<Coord>> vBuf = TypeInfo::cast<DeviceArrayField<Coord>>(vel);

			m_positions.assign(*(pBuf->getDataPtr()));
			m_velocities.assign(*(vBuf->getDataPtr()));
		}*/
	}

	template<typename TDataType>
	void CollidablePoints<TDataType>::updateMechanicalState()
	{
/*		auto mstate = getParent()->getMechanicalState();
		auto mType = mstate->getMaterialType();
		auto dc = getParent()->getMechanicalState();
		if (mType == MechanicalState::ParticleSystem)
		{
			auto center = mstate->getField<HostVarField<Coord>>(MechanicalState::position())->getData();
			auto rotation = mstate->getField<HostVarField<Matrix>>(MechanicalState::rotation())->getData();
			auto vel = mstate->getField<HostVarField<Coord>>(MechanicalState::velocity())->getData();
			
			CArray<Coord> hPos;
			CArray<Coord> hInitPos;
			DArray<Coord> dInitPos;
			hPos.resize(m_positions.size());
			hInitPos.resize(m_positions.size());
			dInitPos.resize(m_positions.size());

			auto mp = std::dynamic_pointer_cast<FrameToPointSet<TDataType>>(m_mapping);
			mp->applyTransform(Rigid(center, Quat<Real>(rotation)), dInitPos);

			Real dt = getParent()->getDt();

			hPos.assign(m_positions);
			hInitPos.assign(dInitPos);
			Coord displacement(0);
			Coord angularVel(0);
			int nn = 0;
			for (uint i = 0; i < hPos.size(); i++)
			{
				Coord r = hInitPos[i] - center;
				if ((hInitPos[i] - hPos[i]).norm() > EPSILON && r.norm() > EPSILON)
				{
					displacement += (hPos[i] - hInitPos[i]);
					Coord vel_i = (hPos[i] - hInitPos[i]) / dt;
					angularVel += r.cross(vel_i) / r.normSquared();
					nn++;
				}
			}
			if (nn > 0)
			{
				displacement /= nn;
				angularVel /= nn;
			}
			
			dc->getField<HostVarField<Coord>>(MechanicalState::position())->setValue(center + displacement);
			dc->getField<HostVarField<Coord>>(MechanicalState::velocity())->setValue(vel + displacement/ dt);
			dc->getField<HostVarField<Coord>>(MechanicalState::angularVelocity())->setValue(angularVel);

			hPos.clear();
			hInitPos.clear();
			dInitPos.clear();
		}
		else
		{
			auto posArr = dc->getField<DeviceArrayField<Coord>>(MechanicalState::position());
			auto velArr = dc->getField<DeviceArrayField<Coord>>(MechanicalState::velocity());

			posArr->getDataPtr()->assign(m_positions);
			velArr->getDataPtr()->assign(m_velocities);
		}*/
	}

	DEFINE_CLASS(CollidablePoints);
}