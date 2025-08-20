#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {
    try {
        std::cout << "測試 ONNX Runtime 初始化..." << std::endl;
        
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        std::cout << "✓ ONNX Runtime 環境初始化成功" << std::endl;
        
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        std::cout << "✓ Session 選項設定成功" << std::endl;
        
        std::string model_path = "/circ330/forgithub/VisualFusion_libtorch/Onnx/model/onnxModel/SemLA_onnx_320x240_opencv_compatible.onnx";
        std::cout << "嘗試載入模型: " << model_path << std::endl;
        
        auto session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        std::cout << "✓ ONNX 模型載入成功！" << std::endl;
        
        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t num_input_nodes = session->GetInputCount();
        size_t num_output_nodes = session->GetOutputCount();
        
        std::cout << "模型有 " << num_input_nodes << " 個輸入，" << num_output_nodes << " 個輸出" << std::endl;
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session->GetInputNameAllocated(i, allocator);
            std::cout << "輸入 " << i << ": " << input_name.get() << std::endl;
        }
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            std::cout << "輸出 " << i << ": " << output_name.get() << std::endl;
        }
        
        std::cout << "✓ 測試完成，ONNX Runtime 工作正常！" << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "✗ ONNX Runtime 錯誤: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "✗ 一般錯誤: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
