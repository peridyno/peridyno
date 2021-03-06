namespace dyno
{
	template<typename T, DeviceType dstType, DeviceType srcType>
	void arryCpy(Array<T, dstType>& dst, Array<T, srcType>& src)
	{
		assert(dst.size() == src.size());
		size_t totalNum = dst.size();

		if (dst.isGPU() && src.isGPU())			cudaMemcpy(dst.begin(), src.begin(), totalNum * sizeof(T), cudaMemcpyDeviceToDevice);
		else if (dst.isCPU() && src.isGPU())	cudaMemcpy(dst.begin(), src.begin(), totalNum * sizeof(T), cudaMemcpyDeviceToHost);
		else if (dst.isGPU() && src.isCPU())	cudaMemcpy(dst.begin(), src.begin(), totalNum * sizeof(T), cudaMemcpyHostToDevice);
		else if (dst.isCPU() && src.isCPU())	memcpy(dst.begin(), src.begin(), totalNum * sizeof(T));
	}

	template<typename T, DeviceType dstType, DeviceType srcType>
	void arryCpy(ArrayList<T, dstType>& dst, ArrayList<T, srcType>& src)
	{

	}
}
