#pragma once

#include "Algorithm/Reduction.h"
#include "Algorithm/Functional.h"
#include "Algorithm/Arithmetic.h"
#include "Module/ConstraintModule.h"


namespace dyno {


	template<typename TDataType>
	class EnergyAnalysis : public ConstraintModule
	{
		DECLARE_TCLASS(EnergyAnalysis, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		EnergyAnalysis();
		~EnergyAnalysis() override;

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "Input particle position");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "Input particle position");

		DEF_ARRAYLIST_IN(int, NeighborIds, DeviceType::GPU, "");

		void constrain() override;
		bool initializeImpl() override;

		void setNamePrefix(std::string prefix);
		void setOutputPath(std::string path);

	private:
		int mFileIndex = 0;

		std::string mOutpuPath;
		std::string mOutputPrefix = "NeighborCountAnalysis";

		DArray<Real> m_Energy;
		DArray<Real> m_Count;

		std::unique_ptr<std::fstream> m_output;
		bool initial = true;
		Reduction<Real>* m_reduce;
		Arithmetic<Real>* m_arithmetic;
		uint counter = 0;
		Reduction<float>* m_reduce_real;
		//std::fstream m_output;
	};

	IMPLEMENT_TCLASS(EnergyAnalysis, TDataType)
}