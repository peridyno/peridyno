#pragma once
#include "Node.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <math_constants.h>

namespace dyno
{
	/*!
	*	\class	CapillaryWave
	*	\brief	Peridynamics-based CapillaryWave.
	*/
	template<typename TDataType>
	class CapillaryWave : public Node
	{
		//DECLARE_CLASS_1(CapillaryWave, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename Vector<Real, 2> Coord2D;
		typedef typename Vector<Real, 4> Coord4D;

		CapillaryWave(int size, std::string name = "default");
		virtual ~CapillaryWave();

	public:

		DEF_ARRAY2D_STATE(Coord2D, Position, DeviceType::GPU, "Height field velocity");

	protected:
		void resetStates() override;

		void updateStates() override;

		void updateTopology() override;


	public:
		int mResolution;

		float mChoppiness;  //设置浪尖的尖锐性，范围0~1

		Vec4f* m_displacement = nullptr;  // 位移场
	};
}