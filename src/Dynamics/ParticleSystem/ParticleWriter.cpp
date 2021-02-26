#include "ParticleWriter.h"
#include "Framework/MechanicalState.h"
#include "Framework/ModuleIO.h"
#include "Utility/Function1Pt.h"

#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>

namespace dyno
{
	IMPLEMENT_CLASS_1(ParticleWriter, TDataType)

	template<typename TDataType>
	ParticleWriter<TDataType>::ParticleWriter()
	: IOModule()
	{
		attachField(&m_position, MechanicalState::position(), "Storing the particle positions!", false);
		attachField(&m_color_mapping, "ColorMapping", "Storing the particle properties!", false);
	}

	template<typename TDataType>
	ParticleWriter<TDataType>::~ParticleWriter()
	{
	}

	template<typename TDataType>
	void ParticleWriter<TDataType>::setNamePrefix(std::string prefix)
	{
		m_name_prefix = prefix;
	}

	template<typename TDataType>
	void ParticleWriter<TDataType>::setOutputPath(std::string path)
	{
		m_output_path = path;
	}

	template<typename TDataType>
	bool ParticleWriter<TDataType>::execute()
	{
		printf("===========WRITER============\n");
		assert(m_position.getElementCount() == m_color_mapping.getElementCount());

		std::stringstream ss; ss << m_output_index;
		std::string filename = m_output_path + std::string("D:\\info\\0918\\test_") + ss.str() + std::string(".txt");
		std::ofstream output(filename.c_str(), std::ios::out | std::ios::binary);

		int total_num = m_position.getElementCount();

		output.write((char*)&total_num, sizeof(int));

		if (total_num == 0)
		{
			m_output_index++;
			output.close();
			return true;
		}


		CArray<Coord> host_position;
		CArray<Real> host_mapping;
		host_position.resize(total_num);
		host_mapping.resize(total_num);

		Function1Pt::copy(host_position, m_position.getValue());
		Function1Pt::copy(host_mapping, m_color_mapping.getValue());

		
		for (int i = 0; i < total_num; i++)
		{
			output.write((char*)&(host_position[i][0]), sizeof(Real));
			output.write((char*)&(host_position[i][1]), sizeof(Real));
			output.write((char*)&(host_position[i][2]), sizeof(Real));
			output.write((char*)&(host_mapping[i]), sizeof(Real));
		}

		output.close();

		host_position.release();
		host_mapping.release();


		m_output_index++;

		return true;
	}
}