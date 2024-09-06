#include "gtest/gtest.h"
#include "STL/PriorityQueue.h"
#include "STL/RandomAccessContainer.h"
#include "STL/Heap.h"
#include <stdio.h>
#include <string>

using namespace std;
using namespace dyno;

#define TEST_CASE (16)

using Real = float;
typedef dyno::priority_queue<Real> this_type;
typedef this_type::container_type container_type;
typedef this_type::size_type size_type;
typedef container_type::iterator iterator;
typedef this_type::value_type value_type;

Real hostBuffer0[TEST_CASE] = { 0 };

__shared__ Real deviceBuffer[TEST_CASE];

DYN_FUNC void printf_helper0() {
	printf("This is print all helper\n");
#ifndef __CUDA_ARCH__
	string ss;
#endif

	for (int i = 0; i < TEST_CASE; ++i) {
#ifdef __CUDA_ARCH__ 
		//begin your device code
		//printf("This is device\n"); 
		printf("%f ", deviceBuffer[i]);
#else 
		//begin your host code
		//printf(" This is host\n"); 
		ss += to_string(hostBuffer0[i]) + " ";
#endif
	}
#ifdef __CUDA_ARCH__
	printf("\n");
#else
	printf("%s\n", ss.c_str());
#endif
}

DYN_FUNC void printf_helper0(const this_type& c) {
	printf("This is print size helper\n");
	this_type::size_type _len = c.size();

#ifndef __CUDA_ARCH__
	string ss;
#endif

	for (int i = 0; i < _len; ++i) {
#ifdef __CUDA_ARCH__ 
		//begin your device code
		//printf("This is device\n"); 
		printf("%f ", c.container[i]);
#else 
		//begin your host code
		//printf(" This is host\n"); 
		ss += to_string(c.container[i]) + " ";
#endif
	}
#ifdef __CUDA_ARCH__
	printf("\n");
#else
	printf("%s\n", ss.c_str());
#endif
}

DYN_FUNC void printf_PQ(this_type& pq) {
	printf("This is print priority sort helper\n");
	while (!pq.isEmpty()) {
		auto ele = pq.top();
		printf("%f ", ele);
		pq.pop();
	}
	printf("\n");
}

DYN_FUNC void push_PQ(this_type& pq) {

	for (int i = 0; i < TEST_CASE; ++i) {
		auto tmp = (Real)(i + sinf(i) * 10) * 0.1f;
		pq.push(tmp);
	}
}


__global__ void PQ_test_GPU() {
	//init
	for (int i = 0; i < TEST_CASE; ++i)deviceBuffer[i] = 0.0f;
	printf("======================================== Hello GPU ===================================\n");
	container_type deviceViewer = container_type();
	deviceViewer.reserve(deviceBuffer, TEST_CASE);
	this_type devicePQ(deviceViewer);
	push_PQ(devicePQ);
	printf_helper0(devicePQ);
	printf_PQ(devicePQ);

	push_PQ(devicePQ);
	for (int i = 0; i < (TEST_CASE / 2); ++i) {
		if (i % 2) {
			devicePQ.remove(i);
		}
	}
	printf_helper0(devicePQ);
	printf_helper0();
	printf_PQ(devicePQ);

	push_PQ(devicePQ);
	devicePQ.container[10] *= 10.0f;
	devicePQ.change(10);
	printf_helper0(devicePQ);
	printf_PQ(devicePQ);
}

TEST(PriorityQueue, func)
{
	this_type hostPQ;
	hostPQ.container.reserve(hostBuffer0, TEST_CASE);
	printf("Data: ");
	for (int i = 0; i < TEST_CASE; ++i) {
		auto tmp = (Real)(i + sinf(i) * 10) * 0.1f;
		printf("%f ", tmp);
	}
	printf("\n");


	printf("======================================== Hello CPU ===================================\n");

	push_PQ(hostPQ);
	printf_helper0(hostPQ);
	printf_PQ(hostPQ);

	push_PQ(hostPQ);
	for (int i = 0; i < (TEST_CASE/2); ++i) {
		if (i % 2) {
			hostPQ.remove(i);
		}
	}
	printf_helper0(hostPQ);
	printf_helper0();
	printf_PQ(hostPQ);

	push_PQ(hostPQ);
	hostPQ.container[10] *= 10.0f;
	hostPQ.change(10);
	printf_helper0(hostPQ);
	printf_PQ(hostPQ);

	PQ_test_GPU << <1, 1 >> > ();
	cudaDeviceSynchronize();
}


