#pragma once
#include "Framework/ModuleIO.h"
#include "Framework/ModuleTopology.h"

#include <string>

namespace dyno
{

	template <typename TDataType> class TriangleSet;
	template<typename TDataType>
	class ParticleWriter : public IOModule
	{
		DECLARE_CLASS_1(ParticleWriter, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;
		ParticleWriter();
		virtual ~ParticleWriter();

		void setNamePrefix(std::string prefix);
		void setOutputPath(std::string path);

		bool execute() override;

	public:
		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Real> m_color_mapping;

		DeviceArrayField<Triangle> m_triangle_index;
		DeviceArrayField<Coord> m_triangle_pos;

	private:
		int m_output_index = 0;
		std::string m_output_path;
		std::string m_name_prefix;
	};

#ifdef PRECISION_FLOAT
	template class ParticleWriter<DataType3f>;
#else
	template class ParticleWriter<DataType3d>;
#endif
}
