#pragma once
#include "Platform.h"
#include "Array/Array.h"
#include "STL/List.h"

namespace dyno
{
	template<int N>
	struct PlaceHolder
	{
		char data[N];
	};


	template<int N>
	void parallel_allocate_for_list(void* lists, void* elements, int ele_size, GArray<int> index);

	template struct PlaceHolder<1>;
	template struct PlaceHolder<2>;
	template struct PlaceHolder<3>;
	template struct PlaceHolder<4>;
	template struct PlaceHolder<5>;
	template struct PlaceHolder<6>;
	template struct PlaceHolder<7>;
	template struct PlaceHolder<8>;

	template void parallel_allocate_for_list<1>(void* lists, void* elements, int ele_size, GArray<int> index);
	template void parallel_allocate_for_list<2>(void* lists, void* elements, int ele_size, GArray<int> index);
	template void parallel_allocate_for_list<3>(void* lists, void* elements, int ele_size, GArray<int> index);
	template void parallel_allocate_for_list<4>(void* lists, void* elements, int ele_size, GArray<int> index);
	template void parallel_allocate_for_list<5>(void* lists, void* elements, int ele_size, GArray<int> index);
	template void parallel_allocate_for_list<6>(void* lists, void* elements, int ele_size, GArray<int> index);
	template void parallel_allocate_for_list<7>(void* lists, void* elements, int ele_size, GArray<int> index);
	template void parallel_allocate_for_list<8>(void* lists, void* elements, int ele_size, GArray<int> index);
}