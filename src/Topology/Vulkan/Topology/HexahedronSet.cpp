#include "HexahedronSet.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace dyno
{
	
    template<typename TDataType>
	HexahedronSet<TDataType>::HexahedronSet()
		: QuadSet<TDataType>(),mSort(getSpvFile("shaders/glsl/topology/HexKeySort.comp.spv"))
	{
        this->addKernel("SetupHexKey",
                        std::make_shared<VkProgram>(BUFFER(QKey), BUFFER(int), BUFFER(Quad), CONSTANT(uint)));
        this->addKernel("CountQKey", std::make_shared<VkProgram>(BUFFER(int), BUFFER(QKey), CONSTANT(uint)));
        this->addKernel("SetupQuad", std::make_shared<VkProgram>(BUFFER(Quad), BUFFER(Quad2Hex), BUFFER(QKey),
                                                                 BUFFER(int), BUFFER(int), CONSTANT(uint)));

        this->kernel("SetupHexKey")->load(getSpvFile("shaders/glsl/topology/SetupHexKey.comp.spv"));
        this->kernel("CountQKey")->load(getSpvFile("shaders/glsl/topology/CountQKey.comp.spv"));
        this->kernel("SetupQuad")->load(getSpvFile("shaders/glsl/topology/SetupQuad.comp.spv"));
	}

	
    template<typename TDataType>
	HexahedronSet<TDataType>::~HexahedronSet()
	{
	}

	
    template<typename TDataType>
	void HexahedronSet<TDataType>::setHexahedrons(std::vector<Hexahedron>& hexahedrons)
	{
		std::vector<Quad> quads;

		m_hexahedrons.resize(hexahedrons.size());
		m_hexahedrons.assign(hexahedrons);

		this->updateQuads();
	}

	
    template<typename TDataType>
	void HexahedronSet<TDataType>::setHexahedrons(DArray<Hexahedron>& hexahedrons)
	{
		if (hexahedrons.size() != m_hexahedrons.size())
		{
			m_hexahedrons.resize(hexahedrons.size());
		}

		m_hexahedrons.assign(hexahedrons);

		this->updateQuads();
	}
	
    template<typename TDataType>
	DArrayList<int>& HexahedronSet<TDataType>::getVer2Hex()
	{
		DArray<uint> counter;
		counter.resize(this->mCoords.size());
		counter.reset();
		VkConstant<uint> n {m_hexahedrons.size()};
		VkConstant<uint> shape_size {sizeof(Hexahedron)};
		this->kernel("CountShape")->flush(vkDispatchSize(n.getValue(), 64), counter.handle(), m_hexahedrons.handle(), &n, &shape_size);

		m_ver2Hex.resize(counter);
		counter.reset();

		VkConstant<VkDeviceAddress> vec2Hex {m_ver2Hex.lists().handle()->bufferAddress()};
		this->kernel("SetupShapeId")->flush(vkDispatchSize(n.getValue(), 64), m_hexahedrons.handle(), &n, &shape_size, &vec2Hex);

		counter.clear();
		return m_ver2Hex;
	}

	
    template<typename TDataType>
	void HexahedronSet<TDataType>::getVolume(DArray<Real>& volume)
	{

	}

	template<typename QKey>
	void printTKey(DArray<QKey> keys, int maxLength) {
		CArray<QKey> h_keys;
		h_keys.resize(keys.size());
		h_keys.assign(keys);

		int psize = std::min((int)h_keys.size(), maxLength);
		for (int i = 0; i < psize; i++)
		{
			printf("%d: %d %d %d %d \n", i, h_keys[i][0], h_keys[i][1], h_keys[i][2], h_keys[i][3]);
		}

		h_keys.clear();
	}

	/*void printCount(DArray<int> keys, int maxLength) {
		CArray<int> h_keys;
		h_keys.resize(keys.size());
		h_keys.assign(keys);

		int psize = minimum((int)h_keys.size(), maxLength);
		for (int i = 0; i < psize; i++)
		{
			printf("%d: %d \n", i, h_keys[i]);
		}

		h_keys.clear();
	}*/

	
    template<typename TDataType>
	void HexahedronSet<TDataType>::updateQuads()
	{
		uint hexSize = m_hexahedrons.size();

		DArray<QKey> keys;
		DArray<int> hexIds;

		keys.resize(6 * hexSize);
		hexIds.resize(6 * hexSize);
		VkConstant<uint> n {hexSize};
        this->kernel("SetupHexKey")
            ->flush(vkDispatchSize(hexSize, 64), keys.handle(), hexIds.handle(), m_hexahedrons.handle(), &n);
        mSort.sortByKey(keys, hexIds, SortParam::eUp);

		DArray<int> counter;
		counter.resize(6 * hexSize);

        n.setValue(keys.size());
        this->kernel("CountQKey")->flush(vkDispatchSize(keys.size(), 64), counter.handle(), keys.handle(), &n);

        int quadNum = this->reduce().reduce(*counter.handle());
        this->scan().scan(*counter.handle(), *counter.handle(), VkScan<int>::Type::Exclusive);

		quad2Hex.resize(quadNum);

		auto& pQuad = this->getQuads();
		pQuad.resize(quadNum);
	    n.setValue(keys.size());
        this->kernel("SetupQuad")
            ->flush(vkDispatchSize(keys.size(), 64), pQuad.handle(), quad2Hex.handle(), keys.handle(),
                    counter.handle(), hexIds.handle(), &n);
		counter.clear();

		hexIds.clear();
		keys.clear();

	     //this->updateTriangles();
		this->updateEdges();
	}
	
    template<typename TDataType>
	void HexahedronSet<TDataType>::copyFrom(HexahedronSet hexSet)
	{
		m_hexahedrons.resize(hexSet.m_hexahedrons.size());
		m_hexahedrons.assign(hexSet.m_hexahedrons);

		quad2Hex.resize(hexSet.quad2Hex.size());
		quad2Hex.assign(hexSet.quad2Hex);

		m_ver2Hex.assign(hexSet.m_ver2Hex);

		QuadSet<TDataType>::copyFrom(hexSet);
	}

	DEFINE_CLASS(HexahedronSet);
}