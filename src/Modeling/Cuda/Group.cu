#include "Group.h"
#include "Topology/PointSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"
#include <sstream> 


namespace dyno
{
	template<typename TDataType>
	Group<TDataType>::Group()
	{
		auto callback = std::make_shared<FCallBackFunc>(std::bind(&Group<TDataType>::varChanged, this));

		this->varPrimitiveId()->attach(callback); 
		this->varEdgeId()->attach(callback);
		this->varPointId()->attach(callback);

		this->inPointId()->tagOptional(true);
		this->inEdgeId()->tagOptional(true);
		this->inPrimitiveId()->tagOptional(true);

	}

	template<typename TDataType>
	void Group<TDataType>::resetStates()
	{
		varChanged();
	}


	template<typename TDataType>
	void Group<TDataType>::varChanged()
	{		

		std::string currentString;

		if (this->varPointId()->getValue().size()) 
		{
			currentString = this->varPointId()->getData();
			updateID(currentString, selectedPointID);
		}

		if (this->varEdgeId()->getValue().size()) 
		{
			currentString = this->varEdgeId()->getData();
			updateID(currentString, selectedEdgeID);
		}

		if (this->varPrimitiveId()->getValue().size()) 
		{
			currentString = this->varPrimitiveId()->getData();
			updateID(currentString, selectedPrimitiveID);
		}


	}


	template<typename TDataType>
	void Group<TDataType>::substrFromTwoString(std::string& first, std::string& Second, std::string& line, std::string& myStr, int& index)
	{
		if (index < int(line.size()))
		{
			size_t posStart = line.find(first, index);
			size_t posEnd = line.find(Second, posStart + 1);


			myStr = line.substr(posStart, posEnd - posStart);
			index = posEnd - 1;

			std::stringstream ss2(line);

		}
		else
		{
			return;
		}

	}


	DEFINE_CLASS(Group);
}