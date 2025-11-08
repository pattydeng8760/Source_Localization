#include "H5Cpp.h"
#include <iostream>
#include <vector>
#include <complex>
#include <rarray>

#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

using complex_d = std::complex<double>;

void checkData(const std::string& filename) {
    try {
        H5File file(filename, H5F_ACC_RDONLY);
        std::cout << "File opened successfully: " << filename << std::endl;
        
        // List all datasets in the file
        hsize_t numObjects = file.getNumObjs();
        for (hsize_t i = 0; i < numObjects; ++i) {
            std::string objName = file.getObjnameByIdx(i);
            H5G_obj_t objType = file.getObjTypeByIdx(i);
            
            std::cout << "Object " << i << ": " << objName;
            if (objType == H5G_DATASET) {
                // Get dataset info
                DataSet dataset = file.openDataSet(objName);
                DataSpace dataspace = dataset.getSpace();
                int ndims = dataspace.getSimpleExtentNdims();
                
                std::vector<hsize_t> dims(ndims);
                dataspace.getSimpleExtentDims(dims.data(), nullptr);
                
                std::cout << " (Dataset, " << ndims << "D, shape: ";
                for (int j = 0; j < ndims; ++j) {
                    std::cout << dims[j];
                    if (j < ndims - 1) std::cout << "x";
                }
                std::cout << ")";
            }
            std::cout << std::endl;
        }
    }
    catch (Exception& e) {
        std::cerr << "Error opening file: " << e.getCDetailMsg() << std::endl;
    }
}

// Read 1D Integer dataset
rvector<int> loadFrequencyData(const std::string& filename, const std::string& datasetName) {
    try {
        H5File file(filename, H5F_ACC_RDONLY);
        DataSet dataset = file.openDataSet(datasetName);
        DataSpace dataspace = dataset.getSpace();
        
        hsize_t dims[1];
        int ndims = dataspace.getSimpleExtentDims(dims, nullptr);
        
        if (ndims != 1) {
            throw std::runtime_error("Expected 1D frequency dataset, got " + std::to_string(ndims) + "D");
        }
        
        // Initialize the vector with the correct size
        rvector<int> freq(dims[0]);
        freq.fill(0);
        
        // Read the dataset into the vector
        dataset.read(freq.data(), PredType::NATIVE_INT);
        
        std::cout << "Successfully loaded " << dims[0] << " frequency points." << std::endl;
        return freq;
    }
    catch (Exception& e) {
        std::cerr << "Error loading frequency data: " << e.getCDetailMsg() << std::endl;
        return rvector<int>(); // Return an empty vector on error
    }
}

// Improved complex data loading with better error handling
rvector<complex_d> loadComplexFFTData(const std::string& filename, const std::string& datasetName) {
    try {
        H5File file(filename, H5F_ACC_RDONLY);
        DataSet dataset = file.openDataSet(datasetName);
        DataSpace dataspace = dataset.getSpace();
        DataType datatype = dataset.getDataType();
        
        int ndims = dataspace.getSimpleExtentNdims();
        std::vector<hsize_t> dims(ndims);
        dataspace.getSimpleExtentDims(dims.data(), nullptr);
        
        std::cout << "FFT dataset dimensions: " << ndims << "D, shape: ";
        for (int i = 0; i < ndims; ++i) {
            std::cout << dims[i];
            if (i < ndims - 1) std::cout << "x";
        }
        std::cout << std::endl;
        
        // Calculate total number of elements
        hsize_t total_elements = 1;
        for (int i = 0; i < ndims; ++i) {
            total_elements *= dims[i];
        }
        
        // Check if it's a compound type (common for complex numbers)
        if (datatype.getClass() == H5T_COMPOUND) {
            std::cout << "Detected compound datatype (likely complex numbers)" << std::endl;
            std::cout << "Total complex elements to read: " << total_elements << std::endl;
            
            // For 2D data, we'll read it as a flat array and then can reshape if needed
            rvector<complex_d> fft_data(total_elements);
            fft_data.fill(complex_d(0.0, 0.0));
            
            // Create a memory datatype for complex<double>
            CompType complex_type(sizeof(complex_d));
            complex_type.insertMember("real", 0, PredType::NATIVE_DOUBLE);
            complex_type.insertMember("imag", sizeof(double), PredType::NATIVE_DOUBLE);
            
            dataset.read(fft_data.data(), complex_type);
            
            std::cout << "Successfully loaded " << total_elements << " complex FFT points." << std::endl;
            if (ndims == 2) {
                std::cout << "Data layout: " << dims[0] << " spatial points × " << dims[1] << " frequencies" << std::endl;
            }
            return fft_data;
        }
        // Check if it's a 3D array with shape [N, M, 2] (real, imag as last dimension)
        else if (ndims == 3 && dims[2] == 2) {
            std::cout << "Detected 3D array format [N, M, 2] for complex numbers" << std::endl;
            
            // Read as flat double array then convert to complex
            hsize_t complex_elements = dims[0] * dims[1];
            std::vector<double> temp_data(complex_elements * 2);
            dataset.read(temp_data.data(), PredType::NATIVE_DOUBLE);
            
            rvector<complex_d> fft_data(complex_elements);
            for (size_t i = 0; i < complex_elements; ++i) {
                fft_data[i] = complex_d(temp_data[2*i], temp_data[2*i + 1]);
            }
            
            std::cout << "Successfully loaded " << complex_elements << " complex FFT points." << std::endl;
            std::cout << "Data layout: " << dims[0] << " × " << dims[1] << " spatial-frequency grid" << std::endl;
            return fft_data;
        }
        // Check if it's a 2D array with shape [N, 2] (real, imag)
        else if (ndims == 2 && dims[1] == 2) {
            std::cout << "Detected 2D array format [N, 2] for complex numbers" << std::endl;
            
            // Read as flat double array then convert to complex
            std::vector<double> temp_data(dims[0] * 2);
            dataset.read(temp_data.data(), PredType::NATIVE_DOUBLE);
            
            rvector<complex_d> fft_data(dims[0]);
            for (size_t i = 0; i < dims[0]; ++i) {
                fft_data[i] = complex_d(temp_data[2*i], temp_data[2*i + 1]);
            }
            
            std::cout << "Successfully loaded " << dims[0] << " complex FFT points." << std::endl;
            return fft_data;
        }
        // 1D case - might be interleaved real/imag
        else if (ndims == 1) {
            std::cout << "Detected 1D array - assuming interleaved complex data" << std::endl;
            
            if (dims[0] % 2 != 0) {
                throw std::runtime_error("1D interleaved complex array should have even number of elements");
            }
            
            // Read as double array, then convert to complex
            std::vector<double> temp_data(dims[0]);
            dataset.read(temp_data.data(), PredType::NATIVE_DOUBLE);
            
            rvector<complex_d> fft_data(dims[0] / 2);
            for (size_t i = 0; i < dims[0] / 2; ++i) {
                fft_data[i] = complex_d(temp_data[2*i], temp_data[2*i + 1]);
            }
            
            std::cout << "Successfully loaded " << dims[0]/2 << " complex FFT points." << std::endl;
            return fft_data;
        }
        else {
            throw std::runtime_error("Unsupported FFT data format: " + std::to_string(ndims) + "D array with shape that doesn't match expected complex patterns");
        }
    }
    catch (Exception& e) {
        std::cerr << "Error loading FFT data: " << e.getCDetailMsg() << std::endl;
        return rvector<complex_d>(); // Return an empty vector on error
    }
}

int main() {
    const std::string filename = "../pressure_airfoil.hdf5";
    const std::string filename2 = "../pressure_airfoil_fft.hdf5";
    std::cout << "=== Checking HDF5 file structure ===" << std::endl;
    checkData(filename);
    
    std::cout << "\n=== Loading frequency data ===" << std::endl;
    rvector<int> freq = loadFrequencyData(filename2, "frequency");
    
    std::cout << "\n=== Loading FFT data ===" << std::endl;
    rvector<complex_d> fft = loadComplexFFTData(filename, "pressure");
    
    if (freq.empty()) {
        std::cerr << "Failed to load frequency data." << std::endl;
        return 1;
    }
    
    if (fft.empty()) {
        std::cerr << "Failed to load FFT data." << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Data Summary ===" << std::endl;
    std::cout << "Frequency array size: " << freq.size() << " points" << std::endl;
    std::cout << "FFT data size: " << fft.size() << " complex values" << std::endl;
    
    // Calculate expected relationship
    if (fft.size() % freq.size() == 0) {
        size_t spatial_points = fft.size() / freq.size();
        std::cout << "Interpreted as: " << spatial_points << " spatial points × " 
                  << freq.size() << " frequencies" << std::endl;
    }
    
    std::cout << "\n=== Sample data ===" << std::endl;
    // Show first few frequency points
    size_t freq_sample_size = std::min<size_t>(5, static_cast<size_t>(freq.size()));
    std::cout << "First " << freq_sample_size << " frequencies:" << std::endl;
    for (size_t i = 0; i < freq_sample_size; ++i) {
        std::cout << "  freq[" << i << "] = " << freq[i] << std::endl;
    }
    
    // Show first few FFT values
    size_t fft_sample_size = std::min<size_t>(5, static_cast<size_t>(fft.size()));
    std::cout << "\nFirst " << fft_sample_size << " FFT values:" << std::endl;
    for (size_t i = 0; i < fft_sample_size; ++i) {
        std::cout << "  fft[" << i << "] = " << fft[i] << std::endl;
    }
    
    // Show FFT values at the end of first frequency band (if data is structured as expected)
    if (fft.size() >= freq.size()) {
        std::cout << "\nLast " << freq_sample_size << " FFT values of first spatial point:" << std::endl;
        size_t start_idx = freq.size() - freq_sample_size;
        for (size_t i = 0; i < freq_sample_size; ++i) {
            size_t idx = start_idx + i;
            std::cout << "  fft[" << idx << "] = " << fft[idx] << " (freq " << freq[start_idx + i] << ")" << std::endl;
        }
    }
    
    return 0;
}