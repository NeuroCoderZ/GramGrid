#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

// Узел исполняет наши скомпилированные ADD_I32 и MATMUL16_I32, не сторонний код.
namespace {
thread_local std::string last_error;
constexpr std::size_t tile_elements = 1024;
enum class Operation { Add, Matrix16, SramBatch4 };

class Executor {
public:
    xrt::device device;
    xrt::xclbin image;
    std::unique_ptr<xrt::hw_context> context;
    std::unique_ptr<xrt::kernel> kernel;
    std::unique_ptr<xrt::bo> instructions, a, b, c;
    std::size_t instruction_count = 0;
    std::string device_name;
    Operation operation;
    std::mutex execution_lock;
    bool resident_weights = false;
    std::uint64_t weight_loads = 0;
    std::uint64_t mesh_executions = 0;

    Executor(const char* image_path, const char* instruction_path, Operation op = Operation::Add)
        : device(0u), image(std::string(image_path)), operation(op) {
        std::ifstream file(instruction_path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("Не найден файл инструкций NPU");
        auto size = file.tellg();
        if (size <= 0 || size > 1048576 || static_cast<std::size_t>(size) % 4)
            throw std::runtime_error("Недопустимая длина инструкций NPU");
        std::vector<std::uint32_t> words(static_cast<std::size_t>(size) / 4);
        file.seekg(0); file.read(reinterpret_cast<char*>(words.data()), size);
        if (!file) throw std::runtime_error("Не удалось прочитать инструкции NPU");
        instruction_count = words.size();
        auto kernels = image.get_kernels();
        if (kernels.size() != 1) throw std::runtime_error("Ожидается одно проверенное ядро NPU");
        device.register_xclbin(image);
        context = std::make_unique<xrt::hw_context>(device, image.get_uuid());
        kernel = std::make_unique<xrt::kernel>(*context, kernels.front().get_name());
        instructions = std::make_unique<xrt::bo>(device, words.size()*4,
            XCL_BO_FLAGS_CACHEABLE, kernel->group_id(1));
        a = std::make_unique<xrt::bo>(device, tile_elements*4, XRT_BO_FLAGS_HOST_ONLY, kernel->group_id(3));
        b = std::make_unique<xrt::bo>(device, tile_elements*4, XRT_BO_FLAGS_HOST_ONLY, kernel->group_id(4));
        c = std::make_unique<xrt::bo>(device, tile_elements*4, XRT_BO_FLAGS_HOST_ONLY, kernel->group_id(5));
        std::memcpy(instructions->map<void*>(), words.data(), words.size()*4);
        instructions->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        device_name = device.get_info<xrt::info::device::name>();
    }

    void add(const std::int32_t* input_a, const std::int32_t* input_b,
             std::int32_t* output, std::uint32_t count) {
        if (operation != Operation::Add || !input_a || !input_b || !output || count == 0 || count > 65536)
            throw std::runtime_error("Буферы или ядро ADD_I32 вне допустимых границ");
        for (std::size_t start = 0; start < count; start += tile_elements) {
            auto length = std::min(tile_elements, static_cast<std::size_t>(count)-start);
            auto aa = a->map<std::int32_t*>(); auto bb = b->map<std::int32_t*>();
            std::fill_n(aa, tile_elements, 0); std::fill_n(bb, tile_elements, 0);
            std::copy_n(input_a+start, length, aa); std::copy_n(input_b+start, length, bb);
            a->sync(XCL_BO_SYNC_BO_TO_DEVICE); b->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto run = (*kernel)(3u, *instructions, instruction_count, *a, *b, *c);
            if (run.wait(10000u) != ERT_CMD_STATE_COMPLETED)
                throw std::runtime_error("Аппаратное исполнение NPU не завершено");
            c->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            std::copy_n(c->map<std::int32_t*>(), length, output+start);
        }
    }

    void load_resident_weights(const std::int32_t* weights, std::uint32_t count) {
        std::lock_guard<std::mutex> lock(execution_lock);
        if ((operation != Operation::Matrix16 && operation != Operation::SramBatch4) || !weights || count != 256)
            throw std::runtime_error("Недопустимый профиль резидентных весов NPU");
        resident_weights = false;
        std::copy_n(weights, 256, b->map<std::int32_t*>());
        b->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        resident_weights = true;
        ++weight_loads;
    }

    void forward_resident(const std::int32_t* input, std::int32_t* output, std::uint32_t count) {
        std::lock_guard<std::mutex> lock(execution_lock);
        if (operation != Operation::Matrix16 || !resident_weights || !input || !output || count != 256)
            throw std::runtime_error("Резидентный этап NPU не готов или профиль входа неверен");
        std::copy_n(input, 256, a->map<std::int32_t*>());
        a->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run = (*kernel)(3u, *instructions, instruction_count, *a, *b, *c);
        if (run.wait(2000u) != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("Резидентное исполнение NPU не завершено");
        c->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::copy_n(c->map<std::int32_t*>(), 256, output);
        ++mesh_executions;
    }

    void forward_sram4(const std::int32_t* input, std::int32_t* output, std::uint32_t count) {
        std::lock_guard<std::mutex> lock(execution_lock);
        if (operation != Operation::SramBatch4 || !resident_weights || !input || !output || count != 1024)
            throw std::runtime_error("Неверный пакетный профиль SRAM NPU");
        std::copy_n(input, 1024, a->map<std::int32_t*>());
        a->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run = (*kernel)(3u, *instructions, instruction_count, *a, *b, *c);
        if (run.wait(2000u) != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("Пакетное исполнение SRAM NPU не завершено");
        c->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::copy_n(c->map<std::int32_t*>(), 1024, output);
        ++mesh_executions;
    }

    void matmul16(const std::int32_t* input_a, const std::int32_t* input_b, std::int32_t* output) {
        std::lock_guard<std::mutex> lock(execution_lock);
        resident_weights = false;
        if (operation != Operation::Matrix16 || !input_a || !input_b || !output)
            throw std::runtime_error("Буферы или ядро MATMUL16_I32 недопустимы");
        std::copy_n(input_a, 256, a->map<std::int32_t*>());
        std::copy_n(input_b, 256, b->map<std::int32_t*>());
        a->sync(XCL_BO_SYNC_BO_TO_DEVICE); b->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run = (*kernel)(3u, *instructions, instruction_count, *a, *b, *c);
        if (run.wait(10000u) != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("Матричное исполнение NPU не завершено");
        c->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::copy_n(c->map<std::int32_t*>(), 256, output);
    }
};
}

extern "C" {
const char* gramgrid_sram_error() { return last_error.c_str(); }
void* gramgrid_sram_create(const char* image, const char* instructions) {
    try {
        if (!image || !instructions) throw std::runtime_error("Не заданы артефакты SRAM NPU");
        return new Executor(image, instructions, Operation::SramBatch4);
    } catch (const std::exception& e) { last_error=e.what(); return nullptr; }
}
void gramgrid_sram_destroy(void* handle) { delete static_cast<Executor*>(handle); }
const char* gramgrid_sram_device_name(void* handle) {
    return handle ? static_cast<Executor*>(handle)->device_name.c_str() : "";
}
int gramgrid_sram_load_weights(void* handle, const std::int32_t* weights, std::uint32_t count) {
    try {
        if (!handle) throw std::runtime_error("SRAM NPU не инициализирован");
        static_cast<Executor*>(handle)->load_resident_weights(weights,count); return 0;
    } catch (const std::exception& e) { last_error=e.what(); return 1; }
}
int gramgrid_sram_forward4(void* handle, const std::int32_t* input, std::int32_t* output, std::uint32_t count) {
    try {
        if (!handle) throw std::runtime_error("SRAM NPU не инициализирован");
        static_cast<Executor*>(handle)->forward_sram4(input,output,count); return 0;
    } catch (const std::exception& e) { last_error=e.what(); return 1; }
}

// Separate symbol names prevent old v2 handles from entering the mesh ABI.
const char* gramgrid_mesh_error() { return last_error.c_str(); }
void* gramgrid_mesh_create(const char* image, const char* instructions) {
    try {
        if (!image || !instructions) throw std::runtime_error("Не заданы артефакты резидентного NPU");
        return new Executor(image, instructions, Operation::Matrix16);
    } catch (const std::exception& e) { last_error=e.what(); return nullptr; }
}
void gramgrid_mesh_destroy(void* handle) { delete static_cast<Executor*>(handle); }
const char* gramgrid_mesh_device_name(void* handle) {
    return handle ? static_cast<Executor*>(handle)->device_name.c_str() : "";
}
int gramgrid_mesh_load_weights(void* handle, const std::int32_t* weights, std::uint32_t count) {
    try {
        if (!handle) throw std::runtime_error("Резидентный NPU не инициализирован");
        static_cast<Executor*>(handle)->load_resident_weights(weights,count);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return 1; }
}
int gramgrid_mesh_forward(void* handle, const std::int32_t* input, std::int32_t* output, std::uint32_t count) {
    try {
        if (!handle) throw std::runtime_error("Резидентный NPU не инициализирован");
        static_cast<Executor*>(handle)->forward_resident(input,output,count);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return 1; }
}
std::uint64_t gramgrid_mesh_weight_loads(void* handle) {
    if (!handle) return 0;
    auto* executor=static_cast<Executor*>(handle);
    std::lock_guard<std::mutex> lock(executor->execution_lock);
    return executor->weight_loads;
}
std::uint64_t gramgrid_mesh_executions(void* handle) {
    if (!handle) return 0;
    auto* executor=static_cast<Executor*>(handle);
    std::lock_guard<std::mutex> lock(executor->execution_lock);
    return executor->mesh_executions;
}

const char* gramgrid_last_error() { return last_error.c_str(); }
void* gramgrid_create(const char* image, const char* instructions) {
    try {
        if (!image || !instructions) throw std::runtime_error("Не заданы артефакты ядра NPU");
        return new Executor(image, instructions);
    } catch (const std::exception& e) { last_error=e.what(); return nullptr; }
}
void* gramgrid_create_matmul16(const char* image, const char* instructions) {
    try {
        if (!image || !instructions) throw std::runtime_error("Не заданы артефакты матрицы NPU");
        return new Executor(image, instructions, Operation::Matrix16);
    } catch (const std::exception& e) { last_error=e.what(); return nullptr; }
}
int gramgrid_matmul16_i32(void* handle, const std::int32_t* a, const std::int32_t* b, std::int32_t* out) {
    try {
        if (!handle) throw std::runtime_error("Матричный узел NPU не инициализирован");
        static_cast<Executor*>(handle)->matmul16(a,b,out); return 0;
    } catch (const std::exception& e) { last_error=e.what(); return 1; }
}
void gramgrid_destroy(void* handle) { delete static_cast<Executor*>(handle); }
const char* gramgrid_device_name(void* handle) {
    return handle ? static_cast<Executor*>(handle)->device_name.c_str() : "";
}
int gramgrid_add_i32(void* handle, const std::int32_t* a, const std::int32_t* b,
                    std::int32_t* out, std::uint32_t count) {
    try {
        if (!handle) throw std::runtime_error("Узел NPU не инициализирован");
        static_cast<Executor*>(handle)->add(a,b,out,count); return 0;
    } catch (const std::exception& e) { last_error=e.what(); return 1; }
}
}
