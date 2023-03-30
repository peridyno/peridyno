#include "ArrayTools.h"
#include "STL/List.h"
#include "STL/Map.h"

namespace dyno
{
	template<uint N>
	struct SpaceHolder
	{
		char data[N];
	};

	template struct SpaceHolder<1>;
	template struct SpaceHolder<2>;
	template struct SpaceHolder<3>;
	template struct SpaceHolder<4>;
	template struct SpaceHolder<5>;
	template struct SpaceHolder<6>;
	template struct SpaceHolder<7>;
	template struct SpaceHolder<8>;
	template struct SpaceHolder<9>;
	template struct SpaceHolder<10>;
	template struct SpaceHolder<11>;
	template struct SpaceHolder<12>;
	template struct SpaceHolder<13>;
	template struct SpaceHolder<14>;
	template struct SpaceHolder<15>;
	template struct SpaceHolder<16>;
	template struct SpaceHolder<17>;
	template struct SpaceHolder<18>;
	template struct SpaceHolder<19>;
	template struct SpaceHolder<20>;
	template struct SpaceHolder<21>;
	template struct SpaceHolder<22>;
	template struct SpaceHolder<23>;
	template struct SpaceHolder<24>;
	template struct SpaceHolder<25>;
	template struct SpaceHolder<26>;
	template struct SpaceHolder<27>;
	template struct SpaceHolder<28>;
	template struct SpaceHolder<29>;
	template struct SpaceHolder<30>;
	template struct SpaceHolder<31>;
	template struct SpaceHolder<32>;
	template struct SpaceHolder<33>;
	template struct SpaceHolder<34>;
	template struct SpaceHolder<35>;
	template struct SpaceHolder<36>;
	template struct SpaceHolder<37>;
	template struct SpaceHolder<38>;
	template struct SpaceHolder<39>;
	template struct SpaceHolder<40>;
	template struct SpaceHolder<41>;
	template struct SpaceHolder<42>;
	template struct SpaceHolder<43>;
	template struct SpaceHolder<44>;
	template struct SpaceHolder<45>;
	template struct SpaceHolder<46>;
	template struct SpaceHolder<47>;
	template struct SpaceHolder<48>;

	template<uint N>
	__global__ void AT_Allocate(
		void* lists,
		void* elements,
		size_t ele_size,
		DArray<uint> index)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= index.size()) return;

		List<SpaceHolder<N>>* listStartPtr = (List<SpaceHolder<N>>*)lists;
		SpaceHolder<N>* elementsPtr = (SpaceHolder<N>*)elements;

		uint count = tId == index.size() - 1 ? ele_size - index[index.size() - 1] : index[tId + 1] - index[tId];

		List<SpaceHolder<N>> list;
		list.reserve(elementsPtr + index[tId], count);

		listStartPtr[tId] = list;
	}

	template<uint N>
	void parallel_allocate_for_list(void* lists, void* elements, size_t ele_size, DArray<uint>& index)
	{
		uint pDims = cudaGridSize(index.size(), BLOCK_SIZE);
		AT_Allocate<N> << <pDims, BLOCK_SIZE >> > (
			lists,
			elements,
			ele_size,
			index);
		cuSynchronize();
	}

	template void parallel_allocate_for_list<1>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<2>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<3>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<4>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<5>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<6>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<7>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<8>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_allocate_for_list<9>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<10>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<11>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<12>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<13>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<14>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<15>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<16>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_allocate_for_list<17>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<18>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<19>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<20>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<21>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<22>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<23>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<24>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_allocate_for_list<25>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<26>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<27>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<28>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<29>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<30>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<31>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<32>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_allocate_for_list<33>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<34>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<35>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<36>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<37>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<38>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<39>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<40>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_allocate_for_list<41>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<42>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<43>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<44>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<45>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<46>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<47>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_list<48>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);

	template<uint N>
	__global__ void AT_Assign(
		void* lists,
		void* elements,
		size_t ele_size,
		DArray<uint> index)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= index.size()) return;

		List<SpaceHolder<N>>* listStartPtr = (List<SpaceHolder<N>>*)lists;
		SpaceHolder<N>* elementsPtr = (SpaceHolder<N>*)elements;

		uint count = tId == index.size() - 1 ? ele_size - index[index.size() - 1] : index[tId + 1] - index[tId];

		List<SpaceHolder<N>> list = *(listStartPtr + tId);
		list.reserve(elementsPtr + index[tId], count);

		listStartPtr[tId] = list;
	}

	template<uint N>
	void parallel_init_for_list(void* lists, void* elements, size_t ele_size, DArray<uint>& index)
	{
		uint pDims = cudaGridSize(index.size(), BLOCK_SIZE);
		AT_Assign<N> << <pDims, BLOCK_SIZE >> > (
			lists,
			elements,
			ele_size,
			index);
		cuSynchronize();
	}

	template void parallel_init_for_list<1>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<2>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<3>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<4>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<5>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<6>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<7>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<8>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_init_for_list<9>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<10>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<11>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<12>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<13>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<14>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<15>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<16>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_init_for_list<17>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<18>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<19>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<20>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<21>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<22>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<23>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<24>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_init_for_list<25>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<26>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<27>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<28>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<29>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<30>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<31>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<32>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_init_for_list<33>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<34>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<35>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<36>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<37>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<38>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<39>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<40>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_init_for_list<41>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<42>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<43>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<44>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<45>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<46>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<47>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_list<48>(void* lists, void* elements, size_t ele_size, DArray<uint>& index);



	template<uint N>
	__global__ void AT_Allocate_Map(
		void* maps,
		void* elements,
		size_t ele_size,
		DArray<uint> index)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= index.size()) return;

		Map<uint,SpaceHolder<N>>* mapPtr = (Map<uint,SpaceHolder<N>>*)maps;
		Pair<uint,SpaceHolder<N>>* elementsPtr = (Pair<uint,SpaceHolder<N>>*)elements;

		uint count = tId == index.size() - 1 ? ele_size - index[index.size() - 1] : index[tId + 1] - index[tId];

		Map<uint,SpaceHolder<N>> map;
		map.reserve(elementsPtr + index[tId], count);

		mapPtr[tId] = map;
	}

	template<uint N>
	void parallel_allocate_for_map(void* maps, void* elements, size_t ele_size, DArray<uint>& index)
	{
		uint pDims = cudaGridSize(index.size(), BLOCK_SIZE);
		AT_Allocate_Map<N> << <pDims, BLOCK_SIZE >> > (
			maps,
			elements,
			ele_size,
			index);
		cuSynchronize();
	}

	template void parallel_allocate_for_map<1>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<2>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<3>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<4>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<5>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<6>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<7>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<8>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_allocate_for_map<9>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<10>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<11>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<12>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<13>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<14>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<15>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<16>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_allocate_for_map<17>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<18>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<19>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<20>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<21>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<22>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<23>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<24>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_allocate_for_map<25>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<26>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<27>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<28>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<29>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<30>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<31>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<32>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_allocate_for_map<33>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<34>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<35>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<36>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<37>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<38>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<39>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<40>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_allocate_for_map<41>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<42>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<43>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<44>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<45>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<46>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<47>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_allocate_for_map<48>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);

	template<uint N>
	__global__ void AT_Assign_Map(
		void* maps,
		void* elements,
		size_t ele_size,
		DArray<uint> index)
	{
		uint tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= index.size()) return;

		Map<uint,SpaceHolder<N>>* mapStartPtr = (Map<uint,SpaceHolder<N>>*)maps;
		Pair<uint,SpaceHolder<N>>* elementsPtr = (Pair<uint,SpaceHolder<N>>*)elements;

		uint count = tId == index.size() - 1 ? ele_size - index[index.size() - 1] : index[tId + 1] - index[tId];

		Map<uint,SpaceHolder<N>> map = *(mapStartPtr + tId);
		map.reserve(elementsPtr + index[tId], count);

		mapStartPtr[tId] = map;

	    //printf("parallel_init_for_map: ArrayTools the ptrpair is: %x \n\n", elementsPtr + index[tId]);
	}

	template<uint N>
	void parallel_init_for_map(void* maps, void* elements, size_t ele_size, DArray<uint>& index)
	{
		uint pDims = cudaGridSize(index.size(), BLOCK_SIZE);
		AT_Assign_Map<N> << <pDims, BLOCK_SIZE >> > (
			maps,
			elements,
			ele_size,
			index);
		cuSynchronize();
	}

	template void parallel_init_for_map<1>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<2>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<3>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<4>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<5>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<6>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<7>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<8>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_init_for_map<9>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<10>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<11>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<12>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<13>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<14>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<15>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<16>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_init_for_map<17>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<18>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<19>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<20>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<21>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<22>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<23>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<24>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_init_for_map<25>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<26>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<27>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<28>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<29>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<30>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<31>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<32>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_init_for_map<33>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<34>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<35>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<36>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<37>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<38>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<39>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<40>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);

	template void parallel_init_for_map<41>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<42>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<43>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<44>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<45>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<46>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<47>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
	template void parallel_init_for_map<48>(void* maps, void* elements, size_t ele_size, DArray<uint>& index);
}