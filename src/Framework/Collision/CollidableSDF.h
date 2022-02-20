#pragma once
#include <vector>
#include "Array/Array.h"
#include "Topology/DistanceField3D.h"
#include "Module/CollidableObject.h"

namespace dyno {

	template<typename TDataType>
	class CollidableSDF : public CollidableObject
	{
		DECLARE_TCLASS(CollidableSDF, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		CollidableSDF();

		void setSDF(std::shared_ptr<DistanceField3D<TDataType>> sdf)
		{
			if (m_sdf != nullptr)
			{
				m_sdf->release();
			}

			m_sdf = sdf;
		}

		std::shared_ptr<DistanceField3D<TDataType>> getSDF() { return m_sdf; }

		~CollidableSDF() { m_sdf->release(); }

		void updateCollidableObject() override {};
		void updateMechanicalState() override {};

		Coord getTranslation() { return m_translation; }
		Matrix getRotationMatrix() { return m_rotation; }

		bool initializeImpl() override;

	private:
		Coord m_translation;
		Matrix m_rotation;

		Coord m_velocity;
		Coord m_angular_velocity;

		std::shared_ptr<DistanceField3D<TDataType>> m_sdf;
	};
}