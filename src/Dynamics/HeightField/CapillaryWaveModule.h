#pragma once
#include "Module/ConstraintModule.h"
#include "Peridynamics/NeighborData.h"
#include "types.h"
#include "Module/ComputeModule.h"
#include "DeclarePort.h"
#include <vector>
namespace dyno {

	/**
	  * @brief This is an implementation of elasticity based on projective peridynamics.
	  *		   For more details, please refer to[He et al. 2017] "Projective Peridynamics for Modeling Versatile Elastoplastic Materials"
	  */
	template<typename TDataType>
	class CapillaryWaveModule : public ComputeModule
	{
		DECLARE_CLASS_1(CapillaryWaveModule, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename Vector<Real, 2> Coord2D;
		typedef typename Vector<Real, 4> Coord4D;

		CapillaryWaveModule();
		CapillaryWaveModule(int size, float patchLength);

		~CapillaryWaveModule() override;

		void compute() override;

		DEF_ARRAY2D_STATE(Coord2D, Position, DeviceType::GPU, "Height field velocity");
	
	public:
		float m_patch_length;
		float m_realGridSize;

		float m_simulatedRegionWidth;
		float m_simulatedRegionHeight;

		Vec4f* m_device_grid;		//当前动态区域状态

		Vec4f* m_device_grid_next;

		Vec4f* m_height = nullptr;				//高度场

		size_t m_grid_pitch;

		float m_horizon = 2.0f;			//水面初始高度

		Vec2f* m_source;				//用于添加船与水交互
		float* m_weight;
	protected:
		void initialize();
		void initDynamicRegion();
		void initSource();
		void resetSource();
		void swapDeviceGrid();

		Vec4f* GetHeight() { return m_height; }
	};
}