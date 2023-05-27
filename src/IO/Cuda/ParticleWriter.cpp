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
		//if (time_idx % 6 != 0)
		//{
		//	time_idx++;
		//	return;
		//}
		//time_idx++;


		std::stringstream ss;

		//**********渲染数据需要自动补零
		if (mFileIndex < 10) ss<<"000";
		else if (mFileIndex < 100) ss << "00";
		else if (mFileIndex < 1000) ss << "0";
		//**********
		ss << mFileIndex;
		
		//std::string filename = mOutpuPath + ss.str() + mOutputPrefix + std::string(".txt");
		std::string filename = mOutpuPath.c_str() + std::string("/") + mOutputPrefix.c_str() + ss.str() + std::string(".txt");
		
		//std::ofstream output(filename.c_str(), std::ios::out | std::ios::binary);//binary output
		std::fstream output(filename.c_str(), std::ios::out );

		//int pNum = this->inPosition()->size();
		auto& inPos = this->inPointSet()->getDataPtr()->getPoints();
		int pNum = inPos.size();

		output << pNum<<' ';
		//output.write((char*)&pNum, sizeof(int));

		CArray<Coord> hPosition;
		hPosition.resize(pNum);
		//hPosition.assign(this->inPosition()->getData());
		hPosition.assign(inPos);

		for (int i = 0; i < pNum; i++)
		{

			output << hPosition[i][0] << ' ' << hPosition[i][1] << ' ' << hPosition[i][2] << ' ';
			/*output.write((char*)&(hPosition[i][0]), sizeof(Real));
			output.write((char*)&(hPosition[i][1]), sizeof(Real));
			output.write((char*)&(hPosition[i][2]), sizeof(Real));*/
		}

		output.close();

		mFileIndex++;
	}

	DEFINE_CLASS(ParticleWriter);
}