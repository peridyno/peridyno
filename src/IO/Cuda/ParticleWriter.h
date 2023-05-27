#pragma once
#include "Module/OutputModule.h"
#include "Module/TopologyModule.h"
#include "Topology/PointSet.h"
#include <string>

namespace dyno
{
	template <typename TDataType> class TriangleSet;
	template<typename TDataType>
	class ParticleWriter : public OutputModule
	{
		DECLARE_TCLASS(ParticleWriter, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;
		ParticleWriter();
		virtual ~ParticleWriter();

		void setNamePrefix(std::string prefix);
		void setOutputPath(std::string path);


	protected:
		void updateImpl() override;

	public:
		//DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");
		DEF_INSTANCE_IN(PointSet<TDataType>, PointSet, "");
	private:
		int mFileIndex = 0;
		int time_idx = 0;
		std::string mOutpuPath;
		std::string mOutputPrefix;

	};
}
