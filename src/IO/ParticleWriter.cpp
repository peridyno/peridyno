#include "ParticleWriter.h"

#include <sstream>
#include <iostream>
#include <fstream>

namespace dyno
{
	IMPLEMENT_TCLASS(ParticleWriter, TDataType)

	template<typename TDataType>
	ParticleWriter<TDataType>::ParticleWriter()
	: OutputModule()
	{
	}

	template<typename TDataType>
	ParticleWriter<TDataType>::~ParticleWriter()
	{
	}

	template<typename TDataType>
	void ParticleWriter<TDataType>::setNamePrefix(std::string prefix)
	{
		mOutputPrefix = prefix;
	}

	template<typename TDataType>
	void ParticleWriter<TDataType>::setOutputPath(std::string path)
	{
		mOutpuPath = path;
	}

	template<typename TDataType>
	void ParticleWriter<TDataType>::updateImpl()
	{
		std::stringstream ss; ss << mFileIndex;
		std::string filename = mOutpuPath + ss.str() + mOutputPrefix + std::string(".txt");
		std::ofstream output(filename.c_str(), std::ios::out | std::ios::binary);

		int pNum = this->inPosition()->getElementCount();

		output.write((char*)&pNum, sizeof(int));

		CArray<Coord> hPosition;
		
		hPosition.resize(pNum);
		hPosition.assign(this->inPosition()->getData());
		
		for (int i = 0; i < pNum; i++)
		{
			output.write((char*)&(hPosition[i][0]), sizeof(Real));
			output.write((char*)&(hPosition[i][1]), sizeof(Real));
			output.write((char*)&(hPosition[i][2]), sizeof(Real));
		}

		output.close();

		mFileIndex++;
	}

	DEFINE_CLASS(ParticleWriter);
}