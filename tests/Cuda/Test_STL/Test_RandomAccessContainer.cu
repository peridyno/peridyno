#include "gtest/gtest.h"
#include "STL/RandomAccessContainer.h"
#include <string>
#include <stdio.h>
#define TEST_CASE (16)

using namespace std;
using namespace dyno;
using Real = float;
typedef dyno::RandomAccessContainer<Real> this_type;

Real hostBuffer[TEST_CASE] = {0};
Real hostInsertBuffer[4] = { 1.1f, 1.2f, 1.3f, 1.4f };
__shared__ Real deviceBuffer[TEST_CASE];
__shared__ Real deviceInsertBuffer[4];
DYN_FUNC void printf_helper() {
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
		ss += to_string(hostBuffer[i]) + " ";
	#endif
	}
#ifdef __CUDA_ARCH__
	printf("\n");
#else
	printf("%s\n",ss.c_str());
#endif
}

DYN_FUNC void printf_helper(const this_type& c) {
	printf("This is print size helper\n");
	this_type::size_type _len = c.size();

#ifndef __CUDA_ARCH__
	string ss;
#endif

	for (int i = 0; i < _len; ++i) {
#ifdef __CUDA_ARCH__ 
		//begin your device code
		//printf("This is device\n"); 
		printf("%f ", c[i]);
#else 
		//begin your host code
		//printf(" This is host\n"); 
		ss += to_string(c[i]) + " ";
#endif
	}
#ifdef __CUDA_ARCH__
	printf("\n");
#else
	printf("%s\n", ss.c_str());
#endif
}

DYN_FUNC void test_reserve(this_type& container) {
#ifdef __CUDA_ARCH__
	container.reserve(deviceBuffer, TEST_CASE);
	printf("IN DEVICE: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf("IN DEVICE: is empty: %d\n", container.empty());
#else
	container.reserve(hostBuffer, TEST_CASE);
	printf("IN HOST: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf("IN HOST: is empty: %d\n", container.empty());
#endif
}

DYN_FUNC void test_resize(this_type& container) {
#ifdef __CUDA_ARCH__
	container.resize(TEST_CASE/2);
	printf("IN DEVICE: capacity: %d, size: %d\n", container.capacity(), container.size());
#else
	container.resize(TEST_CASE/2);
	printf("IN HOST: capacity: %d, size: %d\n", container.capacity(), container.size());
#endif
}

DYN_FUNC void test_assign(this_type& container, this_type::value_type val, this_type::size_type size_) {
#ifdef __CUDA_ARCH__
	container.assign(size_, val);
	printf("IN DEVICE: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#else
	container.assign(size_, val);
	printf("IN HOST: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#endif
}

DYN_FUNC void test_assign_buffer(this_type& container, this_type::iterator beg, this_type::size_type size_) {
#ifdef __CUDA_ARCH__
	container.assign(beg, (this_type::iterator)(beg + size_));
	printf("IN DEVICE: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#else
	container.assign(beg, (this_type::iterator)(beg + size_));
	printf("IN HOST: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#endif
}

DYN_FUNC void test_push_back(this_type& container) {
#ifdef __CUDA_ARCH__
	container.push_back();
	printf("IN DEVICE: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#else
	container.push_back();
	printf("IN HOST: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#endif
}

DYN_FUNC void test_push_back(this_type& container, this_type::value_type val) {
#ifdef __CUDA_ARCH__
	container.push_back(val);
	printf("IN DEVICE: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#else
	container.push_back(val);
	printf("IN HOST: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#endif
}

DYN_FUNC void test_clear(this_type& container) {
#ifdef __CUDA_ARCH__
	container.clear();
	bool isEmpty = container.begin() == container.end();
	printf("IN DEVICE: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#else
	container.clear();
	bool isEmpty = container.begin() == container.end();
	printf("IN HOST: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#endif
}

DYN_FUNC void test_erase_first(this_type& container, this_type::value_type val) {
#ifdef __CUDA_ARCH__
	container.erase_first(val);
	printf("IN DEVICE: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#else
	container.erase_first(val);
	printf("IN HOST: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#endif
}

DYN_FUNC void test_erase_first_unsorted(this_type& container, this_type::value_type val) {
#ifdef __CUDA_ARCH__
	container.erase_first_unsorted(val);
	printf("IN DEVICE: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#else
	container.erase_first_unsorted(val);
	printf("IN HOST: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#endif
}

DYN_FUNC void test_erase(this_type& container, this_type::iterator ite) {
#ifdef __CUDA_ARCH__
	container.erase(ite);
	printf("IN DEVICE: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#else
	container.erase(ite);
	printf("IN HOST: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#endif
}

DYN_FUNC void test_erase(this_type& container, this_type::iterator beg, this_type::iterator end) {
#ifdef __CUDA_ARCH__
	container.erase(beg, end);
	printf("IN DEVICE: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#else
	container.erase(beg, end);
	printf("IN HOST: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#endif
}

DYN_FUNC void test_erase_unsorted(this_type& container, this_type::iterator ite) {
#ifdef __CUDA_ARCH__
	container.erase_unsorted(ite);
	printf("IN DEVICE: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#else
	container.erase_unsorted(ite);
	printf("IN HOST: capacity: %d, size: %d\n", container.capacity(), container.size());
	printf_helper(); printf_helper(container);
#endif
}

__global__ void RAC_test_GPU() {
	printf("======================================== Hello GPU ===================================\n");
	//init
	for (int i = 0; i < TEST_CASE; ++i) deviceBuffer[i] = 0;

	
	for (int i = 0; i < 4; ++i) deviceInsertBuffer[i] = 1.1f + 0.1f * Real(i);

	this_type deviceViewer = this_type();

	test_reserve(deviceViewer);
	test_resize(deviceViewer);
	test_assign(deviceViewer, 1.0f, TEST_CASE /2);
	test_assign(deviceViewer, 2.0f, TEST_CASE /4);
	test_push_back(deviceViewer);
	test_push_back(deviceViewer, 3.0f);
	test_clear(deviceViewer);
	test_assign_buffer(deviceViewer, deviceInsertBuffer, 4);
	deviceViewer.insert(deviceViewer.end(), 2, 1.5f);
	deviceViewer.insert(deviceViewer.begin(), 2, 1.0f);
	deviceViewer.insert(deviceViewer.begin() + 2, &deviceInsertBuffer[0], &deviceInsertBuffer[2]);
	deviceViewer.pop_back();
	deviceViewer[0] = 0.9f;
	deviceViewer.front() = 0.8f;
	test_push_back(deviceViewer);
	deviceViewer.back() = 1.6f;
	test_erase_first(deviceViewer, 1.1f);
	test_erase_first_unsorted(deviceViewer, 1.2f);
	test_erase(deviceViewer, deviceViewer.begin() + 2);
	test_erase(deviceViewer, deviceViewer.begin(), deviceViewer.begin() + 2);
	test_erase_unsorted(deviceViewer, deviceViewer.begin() + 2);
}

TEST(RandomAccessContainer, func)
{
	//test_cpu[i] = (float)(i+sinf(i) * 10)* 0.1f;
	//init
	for (int i = 0; i < TEST_CASE; ++i) hostBuffer[i] = 0;
	printf("======================================== Hello CPU ===================================\n");
	this_type hostViewer = this_type();
	test_reserve(hostViewer);
	test_resize(hostViewer);
	test_assign(hostViewer, 1.0f, TEST_CASE / 2);
	test_assign(hostViewer, 2.0f, TEST_CASE / 4);
	test_push_back(hostViewer);
	test_push_back(hostViewer, 3.0f);
	test_clear(hostViewer);
	test_assign_buffer(hostViewer, hostInsertBuffer, 4);
	hostViewer.insert(hostViewer.end(), 2, 1.5f);
	hostViewer.insert(hostViewer.begin(), 2, 1.0f);
	hostViewer.insert(hostViewer.begin() + 2, &hostInsertBuffer[0], &hostInsertBuffer[2]);
	hostViewer.pop_back();
	hostViewer[0] = 0.9f;
	hostViewer.front() = 0.8f;
	test_push_back(hostViewer);
	hostViewer.back() = 1.6f;
	test_erase_first(hostViewer, 1.1f);
	test_erase_first_unsorted(hostViewer, 1.2f);
	test_erase(hostViewer, hostViewer.begin() + 2);
	test_erase(hostViewer, hostViewer.begin(), hostViewer.begin() + 2);
	test_erase_unsorted(hostViewer, hostViewer.begin() + 2);
	RAC_test_GPU <<< 1, 1 >>> ();
	cudaDeviceSynchronize();
}


