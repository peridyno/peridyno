#pragma once

#include "Algorithm/Reduction.h"
#include "Algorithm/Functional.h"
#include "Algorithm/Arithmetic.h"
#include "Module/ConstraintModule.h"

//#include "Collision/Attribute.h"
//#include "ParticleSystem/Module/Kernel.h"
#include "ParticleSystem/Module/SummationDensity.h"


namespace dyno {

	template<typename TDataType> class SummationDensity;

	template<typename TDataType>
	class PaticleUniformAnalysis : public ConstraintModule
	{
		DECLARE_TCLASS(PaticleUniformAnalysis, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		PaticleUniformAnalysis();
		~PaticleUniformAnalysis() override;

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Input particle position");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Input particle position");

		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "");

		void constrain() override;
		bool initializeImpl() override;

		void setNamePrefix(std::string prefix);
		void setOutputPath(std::string path);

		DEF_VAR_IN(Real, SmoothingLength, "Smoothing Length");
		DEF_VAR_IN(Real, SamplingDistance, "Particle sampling distance");

	private:
		int mFileIndex = 0;

		std::string mOutpuPath;
		std::string mOutputPrefix = "PaticleUniformAnalysis";

		DArray<Real> m_SurfaceEnergy;
		DArray<Real> m_DensityEnergy;
		DArray<Real> m_TotalEnergy;
		DArray<Real> m_Count;
		DArray<Real> m_Density;

		std::unique_ptr<std::fstream> m_output;
		bool initial = true;
		Reduction<Real>* m_reduce;
		Arithmetic<Real>* m_arithmetic;
		uint counter = 0;
		Reduction<float>* m_reduce_real;
		//std::fstream m_output;
		Real m_init_density = 0.0f;

		std::shared_ptr<SummationDensity<TDataType>> mSummation;
	};

	IMPLEMENT_TCLASS(PaticleUniformAnalysis, TDataType)
}