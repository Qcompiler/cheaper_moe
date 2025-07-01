#include <cuda_runtime.h>
#include <iostream>

void getNvidiaGPUArchitecture() {
 
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        std::cout << "Device " << 0 << ": " << prop.name << std::endl;
        std::cout << "Compute Capability: " << int(prop.major) * 10 << "." << prop.minor << std::endl;
        std::cout << "Architecture: ";
        
        // 根据compute capability判断架构
        if (prop.major == 8 && prop.minor == 0) std::cout << "Ampere (GA100)";
        else if (prop.major == 8 && prop.minor == 6) std::cout << "Ampere (GA102)";
        else if (prop.major == 7 && prop.minor == 5) std::cout << "Turing (TU1xx)";
        else if (prop.major == 7 && prop.minor == 0) std::cout << "Volta (GV1xx)";
        else if (prop.major == 6 && prop.minor == 1) std::cout << "Pascal (GP10x)";
        else if (prop.major == 6 && prop.minor == 0) std::cout << "Pascal (GP100)";
        else if (prop.major == 5 && prop.minor == 2) std::cout << "Maxwell (GM20x)";
        else std::cout << "Unknown";
        
        std::cout << std::endl << std::endl;
    
}

int main() {
    getNvidiaGPUArchitecture();
    return 0;
}